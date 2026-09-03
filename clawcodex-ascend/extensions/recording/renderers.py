"""Projection helpers and the ``TeeWriter`` for rendered-output capture.

These helpers turn ClawCodex-native event shapes
(:class:`~extensions.api.query.TextDelta`,
:class:`~extensions.api.query.ToolCallEvent`, etc.) and
:class:`CronRun` payloads into terminal-friendly text suitable for
``AsciicastCapture.emit`` / ``marker``.

Visual vocabulary mirrors what the orchestrator dashboard already
prints (see ``extensions/orchestrator/status_dashboard.py:253`` and
``clawcodex_ext/repl/live_status.py``) so the recorded ``.cast`` reads
identically to the live console output.
"""

from __future__ import annotations

import sys
from typing import IO, Any, Protocol, Self, runtime_checkable

__all__ = [
    "TeeWriter",
    "format_cron_event",
    "format_phase_marker",
    "format_tool_event",
]


@runtime_checkable
class _HasPhase(Protocol):
    phase: int


@runtime_checkable
class _HasTurn(Protocol):
    turn: int


@runtime_checkable
class _HasReason(Protocol):
    reason: str


def format_phase_marker(phase: int, total: int | None = None) -> str:
    """Render a phase marker line.

    Example: ``[phase 3/7] Completed``.
    """
    suffix = f"/{total}" if total else ""
    return f"[phase {phase}{suffix}]"


def format_tool_event(tool_name: str, is_error: bool = False, summary: str = "") -> str:
    """Render one tool-call line for the orchestrator / query-loop sinks.

    Mirrors the existing convention from
    ``extensions/orchestrator/cli/issue.py:1521-1575`` which prefixes
    tool activity with two-space indentation.
    """
    tag = "[ERROR]" if is_error else ""
    body = f"  Tool {tool_name}{(': ' + summary) if summary else ''}{tag}"
    return body


def format_cron_event(payload: dict[str, Any]) -> str:
    """Render a cron fire / miss / expire event.

    ``payload`` is expected to be a :class:`CronRun`-shaped dict (see
    ``clawcodex_ext/cron_system/runs.py:30-51``). Missing fields render
    as ``?`` so partial fixtures still produce readable output.
    """
    task_id = payload.get("task_id", "?")
    status = payload.get("status", "?")
    cron = payload.get("cron", "?")
    return f"[cron {status}] task={task_id} schedule='{cron}'"


class TeeWriter:
    """``sys.stdout`` mirror that forwards every write to a callback.

    Used by :class:`extensions.sop_converter.asciicast_projector.SopStageProjector`
    to capture the SOP CLI's existing ``print()`` calls without
    touching the upstream converter.

    Restores the original ``sys.stdout`` on :meth:`close`. Safe to use
    as a context manager.
    """

    def __init__(
        self,
        original: IO[str] | None = None,
        sink: Any | None = None,
    ) -> None:
        self._original = original if original is not None else sys.stdout
        self._sink = sink
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        sys.stdout = self  # type: ignore[assignment]
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        sys.stdout = self._original  # type: ignore[assignment]
        self._installed = False

    def __enter__(self) -> Self:
        self.install()
        return self

    def __exit__(self, *exc: object) -> None:
        self.restore()

    # -- file-like protocol ---------------------------------------------

    def write(self, data: str) -> int:
        if self._sink is not None:
            try:
                self._sink(data)
            except Exception:  # nosec B110
                # Never let a recorder-side failure break the upstream
                # print() chain — the orchestrator / SOP CLI must keep
                # working even if recording fails mid-run.
                pass
        return self._original.write(data)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:  # nosec B110
            # Console flushing is best-effort during recorder teardown.
            pass

    def isatty(self) -> bool:  # pragma: no cover - trivial
        return False
