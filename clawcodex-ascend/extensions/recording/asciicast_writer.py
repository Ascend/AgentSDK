"""Asciicast v2 writer.

Owns one ``.cast`` file. The writer:

* writes the JSON header on :meth:`open`,
* converts :class:`AsciicastEvent` instances into JSONL lines,
* flushes after every frame (per-frame flush keeps ``tail -f`` working
  and matches the recording-strategy decision documented in the
  recording plan),
* serializes concurrent emits via a ``threading.Lock`` so multiple
  :class:`ProgressSink` instances can fan into one writer.

The :class:`AsciicastCapture` is the thin handle adapters call into —
it hides the file I/O and JSON encoding. Calling :meth:`close` on a
capture flushes any buffered data and closes the file descriptor;
calling :meth:`close` again is a no-op so adapters can run their own
cleanup without worrying about ordering.

ANSI escape sequences are written verbatim inside the JSON string — no
parsing, no rewriting. This means the writer works for both the
structured-event-projection mode (orchestrator, query, cron) and the
rendered-output-capture mode (SOP, visualizer).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import TracebackType

from extensions.capabilities.recorder import (
    AsciicastCapture,
    AsciicastEvent,
    AsciicastHeader,
)

__all__ = ["AsciicastWriter", "AsciicastCapture"]


# U+2028 / U+2029 are valid JSON but break NDJSON receivers that split
# on Unicode line terminators. Mirror the upstream behaviour from
# clawcodex_ext/cli_core/ndjson.py without taking a layer-crossing dep.
_LINE_SEPARATORS = {
    "\u2028": "\\u2028",
    "\u2029": "\\u2029",
}


def _safe_dumps(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if "\u2028" not in encoded and "\u2029" not in encoded:
        return encoded
    return "".join(_LINE_SEPARATORS.get(ch, ch) for ch in encoded)


class AsciicastWriter:
    """Owns the ``.cast`` file for one capture.

    Use as a context manager or call :meth:`open` / :meth:`close`
    directly. Frames are appended via the returned :class:`AsciicastCapture`.
    """

    def __init__(self, path: Path, header: AsciicastHeader) -> None:
        self._path = Path(path)
        self._header = header
        self._fp: object | None = None  # the open file object
        self._lock = threading.Lock()
        self._start_monotonic: float = 0.0
        self._frame_count = 0
        self._capture: AsciicastCaptureImpl | None = None

    # -- lifecycle --------------------------------------------------------

    def __enter__(self) -> AsciicastCapture:
        self.open()
        assert self._capture is not None
        return self._capture

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def open(self) -> AsciicastCapture:
        """Open the file, write the header, return a capture handle."""
        if self._fp is not None:
            raise RuntimeError(f"AsciicastWriter already open: {self._path}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # The writer owns this persistent handle until ``close()``; a local
        # context manager would close it before capture frames are appended.
        # pylint: disable-next=consider-using-with
        fp = open(self._path, "w", encoding="utf-8", newline="\n")  # noqa: SIM115
        # Newline="\n" + manual JSON encoding — we never want the
        # platform's default line ending to leak into the .cast file.
        try:
            fp.write(_safe_dumps(self._header.to_dict()))
            fp.write("\n")
            fp.flush()
        except Exception:
            fp.close()
            raise
        self._fp = fp
        self._start_monotonic = time.monotonic()
        self._capture = AsciicastCaptureImpl(self)
        return self._capture

    def close(self) -> None:
        """Close the underlying file. Idempotent."""
        with self._lock:
            fp = self._fp
            self._fp = None
            self._capture = None
        if fp is not None:
            try:
                fp.flush()
            except Exception:  # nosec B110
                # Closing must still proceed if a final best-effort flush fails.
                pass
            fp.close()

    # -- accessors --------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def capture(self) -> AsciicastCapture:
        """Return the capture handle (must call :meth:`open` first)."""
        if self._capture is None:
            raise RuntimeError(f"AsciicastWriter not open: {self._path}")
        return self._capture

    # -- internal ---------------------------------------------------------

    def _write_frame(self, event: AsciicastEvent) -> None:
        with self._lock:
            fp = self._fp
            if fp is None:
                # Capture was closed mid-emit; drop the frame silently
                # rather than crashing the upstream caller.
                return
            payload = [event.t, event.kind, event.data]
            line = _safe_dumps(payload)
            fp.write(line)
            fp.write("\n")
            # Per-frame flush: keeps tail -f honest and bounds memory
            # growth in long-running captures. The lock ensures frames
            # remain in monotonic order even under concurrent emit.
            fp.flush()
            self._frame_count += 1


class AsciicastCaptureImpl:
    """Concrete :class:`AsciicastCapture` bound to one :class:`AsciicastWriter`."""

    def __init__(self, writer: AsciicastWriter) -> None:
        self._writer = writer

    def emit(self, event: AsciicastEvent) -> None:
        """Append one frame. Caller supplies the event timestamp."""
        self._writer._write_frame(event)

    def marker(self, label: str, text: str = "") -> None:
        """Emit a navigation marker.

        If ``text`` is provided, also emits an ``"o"`` frame right
        after the marker so the marker has visible context when
        replayed.
        """
        t = self._now()
        self._writer._write_frame(AsciicastEvent(t=t, kind="m", data=label))
        if text:
            self._writer._write_frame(AsciicastEvent(t=t, kind="o", data=text))

    def resize(self, cols: int, rows: int) -> None:
        """Record a terminal resize (``SIGWINCH``-equivalent)."""
        self._writer._write_frame(AsciicastEvent(t=self._now(), kind="r", data=f"{cols}x{rows}"))

    def close(self) -> None:
        """No-op on a shared writer; individual sources should not call
        :meth:`AsciicastWriter.close` directly — that is owned by the
        caller that opened the capture (typically the CLI).
        """
        return

    def _now(self) -> float:
        return time.monotonic() - self._writer._start_monotonic
