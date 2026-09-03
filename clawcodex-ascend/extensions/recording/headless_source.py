"""Headless (non-interactive) agent-loop recorder for asciicast.

Provides :class:`HeadlessRecorder` — a context manager that opens an
:class:`AsciicastWriter`, injects a capture handle into
:class:`HeadlessOptions`, and translates raw :class:`ToolEvent` and streamed
text chunks into asciicast ``"o"`` / ``"i"`` / ``"m"`` frames.

This module lives in ``extensions/recording/`` (Layer 2). It imports
``clawcodex_ext.tool_system.renderers.ToolEvent`` and the public summarizers
from ``extensions.recording.query_forwarder`` to keep tool rendering
consistent with the orchestrator/query-loop recording path.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

from clawcodex_ext.tool_system.renderers import ToolEvent
from extensions.capabilities.recorder import (
    AsciicastCapture,
    AsciicastEvent,
    AsciicastHeader,
)
from extensions.recording.asciicast_writer import AsciicastWriter
from extensions.recording.query_forwarder import (
    summarize_tool_output,
    summarize_tool_params,
)

logger = logging.getLogger(__name__)

__all__ = ["HeadlessRecorder", "open_headless_recorder"]


def _default_terminal_size() -> tuple[int, int]:
    """Return a sensible (cols, rows) tuple for headless recordings."""
    try:
        size = shutil.get_terminal_size((120, 36))
        return max(size.columns, 80), max(size.lines, 24)
    except Exception:
        return 120, 36


def _summarize_tool_use(event: ToolEvent) -> str:
    params = summarize_tool_params(event.tool_input)
    return f"Tool Use: {event.tool_name} {params}".rstrip()


def _summarize_tool_result(event: ToolEvent) -> str:
    marker = " [ERROR]" if event.is_error else ""
    output = summarize_tool_output(event.tool_output)
    return f"Tool Result: {event.tool_name}{marker} {output}".rstrip()


@dataclass
class HeadlessRecorder:
    """Context manager that captures a headless agent run to a ``.cast`` file.

    Use via :func:`open_headless_recorder` so width/height defaults are
    resolved. The recorder swallows all emit errors so recording can never
    abort the headless run.
    """

    path: Path
    width: int
    height: int
    command: str
    writer: AsciicastWriter | None = field(default=None, init=False)
    capture: AsciicastCapture | None = field(default=None, init=False)
    _start: float = field(default=0.0, init=False)

    def __enter__(self) -> AsciicastCapture | None:
        """Open the writer, emit ``session:start`` marker, return capture."""
        try:
            self.writer = AsciicastWriter(
                self.path,
                AsciicastHeader(
                    width=self.width,
                    height=self.height,
                    command=self.command,
                    title="ClawCodex headless session",
                ),
            )
            self.capture = self.writer.open()
            self._start = time.monotonic()
            self._safe_marker("session:start", text="Session started")
            return self.capture
        except Exception:
            logger.exception("headless recording: failed to open %s", self.path)
            self.writer = None
            self.capture = None
            return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Emit ``session:end`` marker and close the writer."""
        if self.capture is not None:
            label = "session:end"
            try:
                if exc is not None:
                    text = f"exception={type(exc).__name__}"
                else:
                    text = "Session ended"
                self.capture.marker(label, text=text)
            except Exception as marker_exc:
                logger.debug("headless recording: failed to emit final marker: %s", marker_exc)
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                logger.exception("headless recording: failed to close writer")

    # -- safe emit helpers ----------------------------------------------------

    def _now(self) -> float:
        return time.monotonic() - self._start

    def _safe_marker(self, label: str, text: str = "") -> None:
        if self.capture is None:
            return
        try:
            self.capture.marker(label, text=text)
        except Exception:
            logger.debug("headless recording: failed to emit marker %s", label)

    def _safe_emit(self, kind: str, data: str) -> None:
        if self.capture is None or not data:
            return
        try:
            self.capture.emit(AsciicastEvent(t=self._now(), kind=kind, data=data))  # type: ignore[arg-type]
        except Exception:
            logger.debug("headless recording: failed to emit %s frame", kind)

    def emit_input(self, text: str) -> None:
        """Record user (or cron) input as an ``"i"`` frame."""
        if text and not text.endswith("\n"):
            text = text + "\n"
        self._safe_emit("i", text)

    def emit_text(self, text: str) -> None:
        """Record streamed assistant text as an ``"o"`` frame."""
        self._safe_emit("o", text)

    def emit_tool_event(self, event: ToolEvent) -> None:
        """Translate a :class:`ToolEvent` into an ``"o"`` frame."""
        if event.kind == "tool_use":
            line = _summarize_tool_use(event)
        elif event.kind in ("tool_result", "tool_error"):
            line = _summarize_tool_result(event)
        else:
            return
        self._safe_emit("o", line + "\n")


def open_headless_recorder(
    path: str | Path,
    *,
    width: int | None = None,
    height: int | None = None,
    command: str = "clawcodex",
) -> HeadlessRecorder:
    """Factory for :class:`HeadlessRecorder` with sensible terminal-size defaults.

    Args:
        path: Destination ``.cast`` file.
        width: Terminal columns override. Defaults to ``shutil.get_terminal_size``
            or 120.
        height: Terminal rows override. Defaults to
            ``shutil.get_terminal_size`` or 36.
        command: Value for the asciicast header ``command`` field.
    """
    path = Path(path).expanduser().resolve()
    default_cols, default_rows = _default_terminal_size()
    cols = width if width is not None else default_cols
    rows = height if height is not None else default_rows
    if cols <= 0:
        cols = 120
    if rows <= 0:
        rows = 36
    return HeadlessRecorder(
        path=path,
        width=cols,
        height=rows,
        command=command,
    )
