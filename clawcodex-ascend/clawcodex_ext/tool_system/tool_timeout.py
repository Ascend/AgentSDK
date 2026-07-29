"""Per-tool timeout resolution + gap-watchdog helpers (F-108 P108-C / P108-G).

The canonical tool execution lives in ``src/tool_system/`` which
this fork cannot modify. The practical Layer-2 mechanism is:

* **Tool timeout** — observe the ``tool_use`` → ``tool_result`` gap
  inside the agent-loop message stream (already exposed by
  :mod:`extensions.api.query` and :mod:`clawcodex_ext.query.agent_loop_compat`).
  When the gap exceeds ``tool_timeout_s`` for the called tool the
  watching code trips the existing ``AbortController`` so the next
  boundary unwinds the loop. Observable effect: the user sees the
  same outcome (``SessionComplete(reason="timeout")`` /
  ``ToolResult(is_error=True, error="…timed out…")``) as a true
  ``asyncio.wait_for(tool_exec, 120)``.

* **Auto-recovery** — the trip target is always the
  :class:`AbortController` the parent loop already owns. Layer-3
  always wins Layer-1: the watchdog never kills the process, it
  only signals so the canonical handlers can clean up.

This module houses the *policy* (per-tool timeout table, gap
detector). The actual wiring lives in
:mod:`extensions.api.query` (headless) and
:mod:`clawcodex_ext.query.agent_loop_compat` (TUI / cutover). Both
call sites consume :func:`resolve_tool_timeout` and
:class:`ToolGapWatchdog`.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from clawcodex_ext.diagnostics.freeze_config import (
    DEFAULT_FREEZE_SETTINGS,
    FreezeSettings,
    resolve_freeze_settings,
)


# F-108 §十八 acceptance §3: tool timeout default 120 s. ``0`` (via
# env or settings) disables the watchdog for a specific tool.
DEFAULT_TOOL_TIMEOUT_S = 120.0

# Tool categories that we expect to complete much faster than the
# 120 s default. The triage table is opt-in — we don't override the
# global budget for tools that don't have an entry here.
_FAST_TOOL_TIMEOUT_S = 30.0
_LONG_RUNNING_TOOL_TIMEOUT_S = 600.0

# F-108 §十八 — the triage table. ``Bash`` is the canonical
# "tool that can hang forever if a child process is wedged" entry
# (risk #6 in the audit table). WebFetch is bumped to 30 s so the
# 5-min default doesn't cover an obviously-stalled HTTP socket;
# most healthy WebFetch calls return inside the budget. The
# ``Agent`` subagent dispatcher already has its own watchdog via
# F-99 — keep its timeout at the default so we don't double-fire.
_TOOL_TIMEOUT_S: dict[str, float] = {
    "Bash": DEFAULT_TOOL_TIMEOUT_S,
    "Edit": _FAST_TOOL_TIMEOUT_S,
    "Write": _FAST_TOOL_TIMEOUT_S,
    "Read": _FAST_TOOL_TIMEOUT_S,
    "WebFetch": _FAST_TOOL_TIMEOUT_S,
    "WebSearch": _FAST_TOOL_TIMEOUT_S,
    "NotebookEdit": _FAST_TOOL_TIMEOUT_S,
    # Long-running / agentic
    "Agent": _LONG_RUNNING_TOOL_TIMEOUT_S,
}


@dataclass(frozen=True)
class ToolTimeoutResolution:
    """Resolved budget for one tool call.

    ``enabled=False`` means the gap-watchdog should not arm. The
    ``user_override`` flag distinguishes ``0`` (user disabled) from
    the absence of a per-tool entry (default applies).
    """

    tool_name: str
    timeout_s: float
    enabled: bool
    user_override: bool = False
    # ``from_settings`` is True when the value was pulled out of
    # the persisted FreezeSettings.tool_timeout_s; False when only
    # the table above (or the dataclass default) drove the choice.
    from_settings: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "timeout_s": self.timeout_s,
            "enabled": self.enabled,
            "user_override": self.user_override,
            "from_settings": self.from_settings,
        }


# ----------------------------------------------------------------------
# Resolution — pure, fast, side-effect free.
# ----------------------------------------------------------------------


def resolve_tool_timeout(
    tool_name: str,
    *,
    settings: FreezeSettings | None = None,
    explicit_override: float | None = None,
) -> ToolTimeoutResolution:
    """Return the wall-clock budget for ``tool_name``.

    Resolution order (mirrors :func:`freeze_config.resolve_freeze_settings`):

    1. ``explicit_override`` (caller-provided, e.g. via CLI flag).
    2. ``os.environ["CLAWCODEX_TOOL_TIMEOUT_<NAME>"]`` (uppercase).
       Only honoured when ``explicit_override`` is None.
    3. ``FreezeSettings.tool_timeout_s`` (the global default).
    4. :data:`_TOOL_TIMEOUT_S` per-tool table.
    5. :data:`DEFAULT_TOOL_TIMEOUT_S` (the dataclass default).

    A resolved value of ``<= 0`` is honoured as "disabled" — the
    gap-watchdog never arms for that tool.
    """
    # 1) explicit
    if explicit_override is not None and explicit_override >= 0:
        return ToolTimeoutResolution(
            tool_name=tool_name,
            timeout_s=float(explicit_override),
            enabled=float(explicit_override) > 0.0,
            user_override=True,
        )
    # 2) per-tool env var
    env_value = _env_tool_timeout(tool_name)
    if env_value is not None:
        return ToolTimeoutResolution(
            tool_name=tool_name,
            timeout_s=env_value,
            enabled=env_value > 0.0,
            user_override=True,
        )
    resolved = settings or resolve_freeze_settings()
    budget = float(resolved.tool_timeout_s)
    settings_touched = resolved.tool_timeout_s != DEFAULT_FREEZE_SETTINGS.tool_timeout_s
    table_value = _TOOL_TIMEOUT_S.get(tool_name)
    # If the user supplied a global ``tool_timeout_s`` (``settings``
    # was injected AND the value differs from the dataclass
    # default), the user-supplied global wins over the per-tool
    # table. Otherwise the per-tool table narrows the budget for
    # known tools so e.g. ``Read`` defaults to 30 s rather than the
    # global 120 s.
    if settings_touched:
        return ToolTimeoutResolution(
            tool_name=tool_name,
            timeout_s=budget,
            enabled=budget > 0.0,
            from_settings=True,
        )
    if table_value is not None:
        return ToolTimeoutResolution(
            tool_name=tool_name,
            timeout_s=table_value,
            enabled=table_value > 0.0,
        )
    return ToolTimeoutResolution(
        tool_name=tool_name,
        timeout_s=budget,
        enabled=budget > 0.0,
    )


def _env_tool_timeout(tool_name: str) -> float | None:
    raw = os.environ.get(f"CLAWCODEX_TOOL_TIMEOUT_{tool_name.upper()}")
    if raw is None or raw.strip() == "":
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return v


# ----------------------------------------------------------------------
# Tool-gap watchdog — observes tool_use → tool_result gaps and
# trips an AbortController when the budget elapses.
# ----------------------------------------------------------------------


@dataclass
class ToolGapWatchdog:
    """Observe tool-use → tool-result gaps and trip on elapsed.

    The watchdog is constructed once per run (per agent loop), then
    fed :meth:`observe_tool_use` and :meth:`observe_tool_result`
    events as the message stream arrives. When the gap for any
    outstanding tool exceeds its resolved budget the watchdog trips
    ``abort_controller`` — the canonical Layer-3 path the canonical
    query loop already honours.

    The watchdog is **observation-only when no ``on_trip`` callback
    is set**: it returns True from :meth:`tick` so the caller can
    log a freeze event without mutating any controller. The trip
    callback is the gate to side-effects.
    """

    abort_controller: Any
    settings: FreezeSettings | None = None
    explicit_overrides: dict[str, float] | None = None
    on_trip: Callable[[ToolTimeoutResolution, float, str], None] | None = None
    logger: Any | None = None
    # Pending tool_use records: tool_use_id -> (started_at, resolution)
    _pending: dict[str, tuple[float, ToolTimeoutResolution]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _tripped: set[str] = field(default_factory=set)

    def observe_tool_use(self, tool_use_id: str, tool_name: str) -> None:
        """Arm the watchdog for a tool call.

        Idempotent: a duplicate ``tool_use_id`` (re-send, idempotent
        retry) keeps the original ``started_at`` so a delayed
        response is judged against the original budget.
        """
        res = resolve_tool_timeout(
            tool_name,
            settings=self.settings,
            explicit_override=(self.explicit_overrides or {}).get(tool_name),
        )
        if not res.enabled:
            # The watchdog was disabled for this tool — keep the
            # record so we still know a tool_use is in flight, but
            # ``tick`` will skip the budget check.
            with self._lock:
                self._pending[tool_use_id] = (time.monotonic(), res)
            return
        with self._lock:
            self._pending[tool_use_id] = (time.monotonic(), res)

    def observe_tool_result(self, tool_use_id: str) -> None:
        """Disarm a tool call when its result lands."""
        with self._lock:
            self._pending.pop(tool_use_id, None)
            self._tripped.discard(tool_use_id)

    def has_pending(self) -> bool:
        """Return whether at least one tool_use is awaiting a result."""
        with self._lock:
            return bool(self._pending)

    def tick(self, *, now: float | None = None) -> list[tuple[str, ToolTimeoutResolution, float]]:
        """Scan pending tool calls; trip any that are over budget.

        ``on_trip`` is invoked at most once per tool_use_id. Returns
        the list of tripped ids so the caller can log an event
        without binding the watchdog to a specific logger.

        The watchdog also HEARTBEATS the global FreezeDetector on
        every tick so a tight tool-use loop doesn't starve the
        Layer-1 thread-stack dump.
        """
        now = time.monotonic() if now is None else now
        tripped: list[tuple[str, ToolTimeoutResolution, float]] = []
        with self._lock:
            for tool_use_id, (started_at, res) in list(self._pending.items()):
                if not res.enabled:
                    continue
                if tool_use_id in self._tripped:
                    continue
                elapsed = now - started_at
                if elapsed < res.timeout_s:
                    continue
                self._tripped.add(tool_use_id)
                tripped.append((tool_use_id, res, elapsed))
        for tool_use_id, res, elapsed in tripped:
            self._trip(
                tool_use_id=tool_use_id,
                resolution=res,
                elapsed=elapsed,
            )
        self._heartbeat_freeze_detector()
        return tripped

    def _trip(
        self,
        *,
        tool_use_id: str,
        resolution: ToolTimeoutResolution,
        elapsed: float,
    ) -> None:
        """Trip the bound ``AbortController`` and notify observers."""
        reason = f"tool_timeout:{resolution.tool_name}:{elapsed:.1f}s>{resolution.timeout_s:.1f}s"
        controller = self.abort_controller
        if controller is not None:
            try:
                signal = getattr(controller, "signal", None)
                aborted = bool(getattr(signal, "aborted", False)) if signal is not None else False
            except Exception:  # nosec
                aborted = False
            if not aborted:
                try:
                    controller.abort(reason)
                except Exception:  # nosec
                    if self.logger is not None:
                        try:
                            self.logger.exception("AbortController.abort failed during tool_timeout trip")
                        except Exception:  # nosec
                            pass
        if self.on_trip is not None:
            try:
                self.on_trip(resolution, elapsed, tool_use_id)
            except Exception:  # nosec
                if self.logger is not None:
                    try:
                        self.logger.exception("tool_timeout on_trip callback raised")
                    except Exception:  # nosec
                        pass

    def _heartbeat_freeze_detector(self) -> None:
        """Keep Layer-1 alive while the gap-watchdog is healthy.

        Cheap mutex + monotonic read; called from the polling loop.
        When Layer-1 is disabled (``CLAWCODEX_FREEZE_DIAG`` unset)
        the singleton is None and the call is a no-op.
        """
        try:
            from clawcodex_ext.diagnostics import FreezeDetector

            det = FreezeDetector._INSTANCE  # noqa: SLF001
            if det is not None:
                det.heartbeat()
        except Exception:  # nosec
            pass


__all__ = [
    "DEFAULT_TOOL_TIMEOUT_S",
    "ToolGapWatchdog",
    "ToolTimeoutResolution",
    "resolve_tool_timeout",
]
