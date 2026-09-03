"""Query-loop event forwarder for asciicast recording.

Translates :class:`QueryEvent` instances
(``extensions/api/query.py:128-186``) into asciicast frames and pushes
them through an :class:`AsciicastCapture`. The forwarder is invoked
from :meth:`QueryRunner.stream` after each event drains from the
headless session queue.

The translation policy mirrors what the live REPL prints (see
``extensions/orchestrator/cli/issue.py:1521-1575`` for the transcript
dump format) so the recorded ``.cast`` reads identically to the live
console.

Why a separate module instead of inlining in ``query.py``: keeps
``extensions/api/query.py`` (layer 2) free of recording concerns
beyond a single ``if capture: forward(capture, event)`` line.
"""

from __future__ import annotations

import logging
from typing import Any

from extensions.api.query import (
    PhaseComplete,
    SessionComplete,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    TurnComplete,
)
from extensions.capabilities.recorder import (
    AsciicastCapture,
    AsciicastEvent,
)

logger = logging.getLogger(__name__)


def forward_event(capture: AsciicastCapture, event: Any) -> None:
    """Translate one :class:`QueryEvent` into an asciicast frame and emit.

    No-op on any exception so a recording failure never blocks the
    query loop.
    """
    try:
        frame = _translate(event)
    except Exception as exc:  # noqa: BLE001
        logger.debug("forward_event translate failed: %s", exc)
        return
    if frame is None:
        return
    try:
        capture.emit(frame)
    except Exception as exc:  # noqa: BLE001
        logger.warning("forward_event emit failed: %s", exc)


def _translate(event: Any) -> AsciicastEvent | None:
    """Pure function: QueryEvent → AsciicastEvent (or None to skip)."""
    if isinstance(event, TextDelta):
        # Streamed text content. Use t=0.0; the writer's monotonic
        # clock stamps the actual playback time on the line.
        if not event.content:
            return None
        return AsciicastEvent(t=0.0, kind="o", data=event.content)

    if isinstance(event, ToolCallEvent):
        # Two-space indent matches the orchestrator transcript dump.
        params = summarize_tool_params(event.params)
        line = f"  Tool Use: {event.tool_name} {params}".rstrip()
        return AsciicastEvent(t=0.0, kind="o", data=line + "\n")

    if isinstance(event, ToolResultEvent):
        is_error = bool(event.result.get("is_error"))
        output = event.result.get("output")
        summary = summarize_tool_output(output)
        marker = " [ERROR]" if is_error else ""
        line = f"  Tool Result: {event.tool_name}{marker} {summary}".rstrip()
        return AsciicastEvent(t=0.0, kind="o", data=line + "\n")

    if isinstance(event, PhaseComplete):
        return AsciicastEvent(
            t=0.0,
            kind="m",
            data=f"phase:{event.phase} (turns={event.turn_count})",
        )

    if isinstance(event, TurnComplete):
        # Skip per-turn frames to keep recordings compact — same
        # policy as AsciicastSink.on_turn_complete.
        return None

    if isinstance(event, SessionComplete):
        return AsciicastEvent(t=0.0, kind="m", data=f"session:{event.reason}")

    return None


def summarize_tool_params(params: dict[str, Any] | None) -> str:
    """Render a tool-call input dict as a compact one-liner."""
    if not params:
        return ""
    if "cmd" in params and isinstance(params["cmd"], str):
        return f"(cmd={params['cmd']!r})"
    if "file_path" in params and isinstance(params["file_path"], str):
        return f"(file_path={params['file_path']!r})"
    if "query" in params and isinstance(params["query"], str):
        q = params["query"]
        return f"(query={q[:80]!r}{'…' if len(q) > 80 else ''})"
    # Fallback: render the dict but cap length
    rendered = repr(params)
    return f"({rendered[:120]}{'…' if len(rendered) > 120 else ''})"


# Backward-compat aliases used by earlier recording adapters.
_summarize_params = summarize_tool_params


def summarize_tool_output(output: Any) -> str:
    """Render a tool-result output as a compact one-liner."""
    if output is None:
        return ""
    if isinstance(output, str):
        return f"({output[:80]}{'…' if len(output) > 80 else ''})"
    if isinstance(output, dict):
        rendered = repr(output)
        return f"({rendered[:120]}{'…' if len(rendered) > 120 else ''})"
    rendered = repr(output)
    return f"({rendered[:80]}{'…' if len(rendered) > 80 else ''})"


# Backward-compat aliases used by earlier recording adapters.
_summarize_output = summarize_tool_output


__all__ = [
    "forward_event",
    "summarize_tool_output",
    "summarize_tool_params",
]
