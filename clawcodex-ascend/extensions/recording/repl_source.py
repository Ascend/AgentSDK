"""Capture the default prompt_toolkit + Rich REPL into a ``.cast``.

This module provides ``install_repl_capture(repl, ctx)`` which is called
from ``clawcodex_ext/frontend/repl.py`` right after
``install_repl_extensions``. It swaps the REPL's :class:`rich.console.Console`
with one whose ``file`` is a tee into an :class:`AsciicastCapture`, and wraps
``repl.prompt_session`` so that user input is recorded as asciicast ``"i"``
events.

Important limitation:

* The prompt_toolkit prompt bar itself (the ``❯`` glyph, line editing,
  completion popups) is rendered by prompt_toolkit, **not** Rich, so it does
  not flow through ``repl.console`` and is **not** captured as video. We
  emit ``m`` markers at prompt start / submit boundaries so the playback
  can still pause at the right spots.

Layer rule (CLAUDE.md): this module lives in ``extensions/recording/``
(Layer 2). It depends on public attributes of the Layer-1 REPL
(``repl.console``, ``repl.prompt_session``) but does not import private
upstream internals.
"""

from __future__ import annotations

import atexit
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import IO, Any

from extensions.capabilities.recorder import (
    AsciicastCapture,
    AsciicastEvent,
    AsciicastHeader,
)
from extensions.recording.asciicast_writer import AsciicastWriter

logger = logging.getLogger(__name__)

__all__ = [
    "ReplConsoleSource",
    "RichConsoleTeeWriter",
    "PromptSessionProxy",
    "install_repl_capture",
]


# ---------------------------------------------------------------------------
# File-like shim that forwards Rich Console writes to the capture
# ---------------------------------------------------------------------------


class RichConsoleTeeWriter:
    """A ``file=`` target for :class:`rich.console.Console`.

    Every ``write()`` forwards the raw chunk to two places:

    1. the original stream (so the user still sees the REPL on screen),
    2. an asciicast ``"o"`` frame through ``emit_o(data)``.

    Rich calls ``write`` with complete ANSI escape sequences and plain
    text, so the captured stream is what the user actually saw (minus the
    prompt_toolkit prompt bar). ``flush`` and ``isatty`` are also
    implemented so Rich does not complain.
    """

    def __init__(
        self,
        original: IO[str],
        emit_o: Any,
    ) -> None:
        self._original = original
        self._emit_o = emit_o

    # -- file-like protocol -------------------------------------------------

    def write(self, data: str) -> int:
        # Mirror to the real screen first, then to the recorder. If the
        # recorder side fails we still want the user to keep using the
        # REPL, so recorder errors are swallowed after logging.
        try:
            written = self._original.write(data)
        except Exception:
            written = 0
        try:
            if data:
                self._emit_o(data)
        except Exception:
            logger.exception("repl capture: failed to emit console chunk")
        return written

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception as exc:
            logger.debug("repl capture: failed to flush mirrored console: %s", exc)

    def isatty(self) -> bool:
        return getattr(self._original, "isatty", lambda: False)()


# ---------------------------------------------------------------------------
# Prompt session proxy: record user input as asciicast "i" events
# ---------------------------------------------------------------------------


class PromptSessionProxy:
    """Wraps a prompt_toolkit ``PromptSession`` and records input.

    ``prompt_async`` emits:

    * ``marker("repl:prompt:start")`` before waiting,
    * the actual user input as an ``"i"`` frame after submit,
    * ``marker("repl:prompt:submit")`` after submit.

    If the prompt returns ``None`` (Ctrl-C / Ctrl-D / EOF) no input frame
    is emitted, but the submit marker is still recorded.
    """

    def __init__(self, session: Any, capture: AsciicastCapture, *, emit_i: Any) -> None:
        self._session = session
        self._capture = capture
        self._emit_i = emit_i

    def __getattr__(self, name: str) -> Any:
        # Forward everything else (completer, key_bindings, history, ...)
        # to the real PromptSession unchanged.
        return getattr(self._session, name)

    async def prompt_async(self, *args: Any, **kwargs: Any) -> Any:
        try:
            self._capture.marker("repl:prompt:start")
        except Exception:
            logger.exception("repl capture: failed to emit prompt:start marker")
        try:
            value = await self._session.prompt_async(*args, **kwargs)
        finally:
            try:
                self._capture.marker("repl:prompt:submit")
            except Exception:
                logger.exception("repl capture: failed to emit prompt:submit marker")
        if value:
            try:
                self._emit_i(value + "\n")
            except Exception:
                logger.exception("repl capture: failed to emit input frame")
        return value


# ---------------------------------------------------------------------------
# RecordableSource wrapper (so this can also be wired via registry later)
# ---------------------------------------------------------------------------


class ReplConsoleSource:
    """A :class:`RecordableSource` that captures the default Rich REPL.

    This is primarily used through :func:`install_repl_capture`, but it
    also satisfies :class:`extensions.capabilities.recorder.RecordableSource`
    so it could be registered under a ``"repl"`` source_id in the future
    if we decide to expose ``clawcodex record --sources repl``.
    """

    source_id = "repl"

    def __init__(self, capture: AsciicastCapture) -> None:
        self._capture = capture
        self._repl: Any | None = None
        self._original_console: Any | None = None
        self._original_session: Any | None = None
        self._tee: RichConsoleTeeWriter | None = None

    def open(self, capture: AsciicastCapture) -> None:
        """No-op when used via install_repl_capture (which wires directly).

        Provided only for :class:`RecordableSource` compatibility.
        """
        return

    def close(self) -> None:
        """Restore the original console / prompt_session if we patched them."""
        if self._repl is None:
            return
        try:
            if self._original_console is not None:
                self._repl.console = self._original_console
        except Exception:
            logger.exception("repl capture: failed to restore console")
        try:
            if self._original_session is not None:
                self._repl.prompt_session = self._original_session
        except Exception:
            logger.exception("repl capture: failed to restore prompt_session")
        self._repl = None


# ---------------------------------------------------------------------------
# Public install hook
# ---------------------------------------------------------------------------


def install_repl_capture(
    repl: Any,
    ctx: Any,
    *,
    path: Path | str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> AsciicastWriter | None:
    """Wire the REPL's console + prompt_session into an asciicast writer.

    ``path`` defaults to ``ctx.options.record`` if available, so callers
    can simply do::

        if ctx.options.record:
            install_repl_capture(repl, ctx)

    Returns the opened :class:`AsciicastWriter` (so the caller can close
    it after ``repl.run()``), or ``None`` if recording is disabled.

    This function swallows import-time / construction errors and logs
    them; a missing recorder extension must not prevent the REPL from
    starting.
    """
    if path is None:
        path = getattr(ctx, "options", None) and getattr(getattr(ctx, "options", None), "record", None)
    if width is None:
        width = getattr(ctx, "options", None) and getattr(getattr(ctx, "options", None), "record_width", None)
    if height is None:
        height = getattr(ctx, "options", None) and getattr(getattr(ctx, "options", None), "record_height", None)
    if not path:
        return None

    path = Path(path).expanduser().resolve()

    # Determine terminal size. Fall back to shutil for the default, and
    # allow CLI overrides so CI / documentation examples are reproducible.
    term = shutil.get_terminal_size()
    cols = width if width is not None else term.columns
    rows = height if height is not None else term.lines
    if cols <= 0:
        cols = 120
    if rows <= 0:
        rows = 36

    header = AsciicastHeader(
        width=cols,
        height=rows,
        command="clawcodex --record " + str(path),
        title="ClawCodex REPL session",
    )

    writer: AsciicastWriter | None = None
    try:
        writer = AsciicastWriter(path, header)
        capture = writer.open()
    except Exception:
        logger.exception("repl capture: failed to open %s", path)
        return None

    # Capture timestamp origin. We use our own monotonic start so we can
    # emit frames without reaching into the writer's private state.
    start = time.monotonic()

    def emit_o(data: str) -> None:
        if not data:
            return
        t = time.monotonic() - start
        capture.emit(AsciicastEvent(t=t, kind="o", data=data))

    def emit_i(data: str) -> None:
        if not data:
            return
        t = time.monotonic() - start
        capture.emit(AsciicastEvent(t=t, kind="i", data=data))

    # Patch repl.console. We preserve as much of the original Console
    # configuration as possible (theme, highlight flag, force_terminal,
    # width) so the captured output matches what the user sees.
    try:
        from rich.console import Console

        original_console = repl.console
        original_file = getattr(original_console, "file", sys.stdout)
        tee = RichConsoleTeeWriter(original_file, emit_o)
        repl.console = Console(
            theme=getattr(original_console, "theme", None),
            highlight=getattr(original_console, "highlight", False),
            force_terminal=getattr(original_console, "force_terminal", True),
            width=getattr(original_console, "width", None) or cols,
            file=tee,
        )
    except Exception:
        logger.exception("repl capture: failed to patch console")
        writer.close()
        return None

    # Patch repl.prompt_session. The prompt bar itself is PTK-rendered,
    # but we can at least capture user input and prompt boundaries.
    try:
        original_session = getattr(repl, "prompt_session", None)
        if original_session is not None:
            repl.prompt_session = PromptSessionProxy(original_session, capture, emit_i=emit_i)
    except Exception:
        logger.exception("repl capture: failed to patch prompt_session")
        # Don't abort; console-only capture is still useful.

    # Ensure the writer is closed when the process exits (e.g. user hits
    # Ctrl-C or SIGTERM). The REPL frontend is synchronous from the
    # caller's perspective, so atexit is safer than relying on the caller
    # to remember writer.close().
    def _close_writer() -> None:
        try:
            writer.close()
        except Exception:
            logger.exception("repl capture: failed to close writer")

    atexit.register(_close_writer)
    return writer
