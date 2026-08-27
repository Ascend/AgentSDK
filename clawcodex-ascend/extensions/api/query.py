#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Public Python API for running a single query.

Wraps the headless entrypoint for programmatic use.
"""

from __future__ import annotations

import asyncio
import io
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

from .debug_log import append_debug_event

# Per-tool gap watchdog. Imported at module load to
# keep ``QueryConfig`` construction cheap; the actual ``ToolGapWatchdog``
# is built inside ``stream`` so we can plumb the user's QConfig knobs.
from clawcodex_ext.tool_system.tool_timeout import ToolGapWatchdog  # pylint: disable=import-error

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


# Wall-clock budget for a single headless query run.
# The runner spawns ``run_headless_session`` on the default executor and
# awaits the resulting future; without a bound that future can hang
# forever (see §十八 risk #5). ``asyncio.wait_for`` below cuts the
# wait at ``QueryConfig.timeout_s`` seconds (default 1800, configured
# via workflow.md ``agent.run_timeout_ms``), yielding
# ``SessionComplete(reason="exit_code=124")`` so callers can detect the
# timeout via the conventional GNU exit code.
#
# ``asyncio.wait_for`` does NOT cancel an executor future — Python's
# executor futures cannot be killed from the event loop — so the
# underlying thread may keep running until ``run_headless_session``
# returns naturally. The headless session is expected to honour
# ``AbortController`` for cooperative cancellation;
# until then this is a known limitation called out in §十八 risk
# table.
#
# Set to ``0`` to disable the timeout (§十八 design decision #5:
# every Layer-2 budget has a ``0`` escape hatch).


@dataclass
class QueryConfig:
    """Configuration for a single query run."""

    prompt: str
    workspace: str | Path
    provider: str | None = None
    model: str | None = None
    # NOTE: tools filter not yet forwarded to headless session (currently unused)
    # tools: list[str] | None = None
    permission_mode: str = "dontAsk"
    max_turns: int = 20
    run_id: str | None = None
    debug_log_path: str | Path | None = None
    # Environment variables merged into the headless session's Bash
    # subprocess env. Values override inherited daemon env.
    env: dict[str, str] | None = None
    # Prompt split: when set, this text is appended to the
    # ``effective_system_prompt`` built by ``run_headless``. The daemon's
    # ``agent_runner`` uses this to keep the constant workflow background
    # (project, conventions, decoupling principles) in the system prompt
    # across every turn, while the per-issue data (identifier, description,
    # labels) lives in ``prompt`` as the user message.
    append_system_prompt: str | None = None
    # Agent identity for pending_messages drain.
    # When set, the headless session's ToolContext gets this agent_id
    # and runtime_tasks, enabling real-time inject at ToolResult
    # boundaries.
    agent_id: str | None = None
    runtime_tasks: Any | None = None
    # When set, the headless session loads the existing transcript via
    # Session.resume(resume_session_id) instead of starting with an empty
    # Conversation. Used by the CLI/REPL --resume path.
    resume_session_id: str | None = None
    # Per-run wall-clock timeout in seconds. When > 0, the event-drain
    # loop in ``stream_events`` enforces this budget and yields
    # ``SessionComplete(reason="exit_code=124")`` on expiry.  Default
    # 1800s (30 min) matches the orchestrator's ``run_timeout_ms``.
    # Set to 0 to disable (§十八 design decision #5).
    timeout_s: float = 1800.0
    # Stream-stall watchdog. When > 0, the event-drain
    # loop aborts the run once the headless session shows NO activity
    # (no tool events AND no stdout growth) for this many consecutive
    # seconds, yielding ``SessionComplete(reason="exit_code=125")``.
    # Provider-agnostic backstop for the observed failure mode where an
    # LLM request returns zero chunks forever: the per-provider idle
    # watchdog (WI-5.2) only covers the Anthropic SDK stream, so
    # OpenAI-compatible paths used to burn the whole wall-clock budget
    # (20 min observed live) doing nothing. Detection is activity-based,
    # not first-token-based — any tool event or stdout growth resets the
    # deadline. Set to 0 to disable (design decision #5).
    #
    # Default rationale (measured over 16 archived production runs):
    # healthy runs showed dual-silence gaps up to 240 s (long LLM turns
    # whose text is not streamed to stdout), while the two genuine
    # hangs sat silent for 949 s / 1140 s. 300 s clears the healthy
    # maximum with margin and still cuts hang recovery from the 1800 s
    # budget to ~5 min. The 30 s requirement is met by the WARN tier
    # below, which is diagnosis-only and safe at low thresholds.
    stall_timeout_s: float = 300.0
    # Early-diagnosis tier: after this many silent seconds a
    # ``query_runner.stall_suspected`` debug event + WARNING log fire
    # (once per silence episode; re-armed when activity resumes). This
    # is what guarantees "clear diagnosis within 30 s of a hang"
    # without any false-kill risk. 0 disables.
    stall_warn_s: float = 30.0
    # Per-tool budget policy. When > 0 the agent loop
    # aborts any tool call whose tool_use → tool_result gap exceeds
    # the resolved budget (see ``tool_timeout.resolve_tool_timeout``).
    # Per-tool overrides via ``tool_timeout_overrides`` take priority.
    # Set to 0 to disable the gap-watchdog entirely.
    tool_timeout_s: float = 120.0
    tool_timeout_overrides: dict[str, float] | None = None
    # Outer agent-loop wall-clock budget. The runner
    # enforces this in the polling loop (mirrors ``timeout_s``) — when
    # the run outlives the budget it yields
    # ``SessionComplete(reason="exit_code=124")`` and signals the
    # abort controller. Set to 0 to disable.
    #
    # Defaults: ``timeout_s`` already enforces a 1800 s budget; this
    # second-tier value matches ``FreezeSettings.agent_loop_timeout_s``
    # (600 s) so either layer hitting the budget surfaces as
    # ``exit_code=124``. The narrower 600 s lets long plugin installs
    # (which the agent really shouldn't run inline) abort early.
    agent_loop_timeout_s: float = 600.0
    # F-REC: optional asciicast capture. When set, every event drained
    # from the headless session queue is forwarded to ``capture.emit``
    # via ``extensions.recording.query_forwarder.forward_event``.
    # ``None`` (the default) keeps the query loop allocation-free for
    # the common non-recording path.
    capture: Any = None
    # Callback invoked by the stream's polling loop (every ~60ms)
    # to drain control-socket commands. Returns a non-None signal
    # (e.g. ``"stop"``) when the stream should abort and break; ``None``
    # to continue normally. When ``None`` (the default), the stream does
    # not drain — preserving backward compatibility for non-orchestrator
    # callers.
    control_drain_fn: Callable[[], str | None] | None = None
    # Async callback invoked by the stream's polling loop after
    # the drain. When the session is paused, this callback blocks
    # (internally running a drain-and-wait loop) so that resume/stop
    # commands are still processed. When not paused, it returns
    # immediately. When ``None`` (the default), the stream does not
    # pause-check — preserving backward compatibility.
    pause_wait_fn: Callable[[], Awaitable[None]] | None = None
    # threading.Event used as a "pause gate" — when cleared,
    # the headless session's on_event callback blocks before putting
    # events into event_queue, preventing further LLM API calls.
    # Set = running (default), clear = paused. ``None`` = no pause
    # gate (backward compatibility for non-orchestrator callers).
    pause_gate: threading.Event | None = None


@dataclass
class TextDelta:
    """Streaming text chunk."""

    content: str


@dataclass
class ToolCallEvent:
    """Tool call event from the agent."""

    tool_name: str
    params: dict[str, Any]
    tool_use_id: str | None = None
    _approved: bool | None = None
    _deny_reason: str | None = None

    @property
    def is_approved(self) -> bool | None:
        return self._approved


@dataclass
class ToolResultEvent:
    """Tool result event from the agent."""

    tool_name: str
    result: dict[str, Any]
    # Pair this result with the originating ToolCallEvent.
    # Populated by convert_tool_event from the bridge dict; defaults to
    # None for events that lack an id.
    tool_use_id: str | None = None


@dataclass
class PhaseComplete:
    """One phase (multiple turns) finished."""

    phase: int
    turn_count: int


@dataclass
class TurnComplete:
    """One turn finished."""

    turn: int


@dataclass
class SessionComplete:
    """Session finished."""

    reason: str


QueryEvent = TextDelta | ToolCallEvent | ToolResultEvent | TurnComplete | PhaseComplete | SessionComplete


class QueryRunner:
    """Execute a single prompt through ClawCodex query engine."""

    def __init__(self, config: QueryConfig) -> None:
        self.config = config

    async def stream(self) -> AsyncIterator[QueryEvent]:
        """Yield query events as they occur.

        Uses the headless runner registry under the hood, which dispatches
        to the configured backend (default: upstream headless entrypoint).
        The caller observes tool events via ``on_event`` without needing
        to import from upstream.
        """
        # Import the headless session runner — this stays off the upstream
        # import path at module-load time; the concrete implementation is
        # loaded lazily inside run_headless_session.
        from ..capabilities.headless_runner import (
            HeadlessSessionOptions,
            make_abort_controller,
            run_headless_session,
        )

        debug_log_path = self.config.debug_log_path
        append_debug_event(
            debug_log_path,
            "query_runner.start",
            run_id=self.config.run_id,
            provider=self.config.provider,
            model=self.config.model,
            permission_mode=self.config.permission_mode,
            prompt_len=len(self.config.prompt),
            workspace=str(self.config.workspace),
            max_turns=self.config.max_turns,
            stall_timeout_s=self.config.stall_timeout_s,
        )

        event_queue: queue.Queue[Any] = queue.Queue()
        tool_event_count = 0
        last_event_at = time.monotonic()
        tool_names_by_id: dict[str, str] = {}

        # Tool-gap watchdog. Constructed lazily on
        # the first tool event (so we can plumb the abort_controller
        # that the headless session owns). ``tool_watchdog_state``
        # holds the watchdog + the bookkeeping flags the polling
        # loop checks per tick.
        tool_watchdog_state: dict[str, Any] = {"wd": None, "tripped": False, "last_trip": None}

        # Capture pause_gate and abort_controller references for
        # the on_event closure. Both are defined here so the closure
        # captures them by reference (late binding).
        _pause_gate = self.config.pause_gate
        abort_controller = make_abort_controller()

        def on_event(tool_event: Any) -> None:
            nonlocal tool_event_count, last_event_at
            try:
                tool_event_count += 1
                last_event_at = time.monotonic()
                kind = getattr(tool_event, "kind", None)
                tool_use_id = getattr(tool_event, "tool_use_id", None)
                tool_name = getattr(tool_event, "tool_name", "")
                if kind == "tool_use" and tool_use_id and tool_name:
                    tool_names_by_id[str(tool_use_id)] = str(tool_name)
                elif not tool_name and tool_use_id:
                    tool_name = tool_names_by_id.get(str(tool_use_id), "")
                is_error = getattr(tool_event, "is_error", False)
                error = getattr(tool_event, "error", None)
                append_debug_event(
                    debug_log_path,
                    "headless.event",
                    run_id=self.config.run_id,
                    kind=kind,
                    tool=tool_name,
                    tool_use_id=tool_use_id,
                    is_error=is_error,
                    error=str(error)[:500] if error is not None and is_error else None,
                )
                # Feed the gap-watchdog from the headless
                # ``on_event`` channel so the trip fires at the moment
                # the event arrives, not at the polling-loop tick.
                wd = tool_watchdog_state.get("wd")
                if wd is not None and tool_use_id:
                    if kind == "tool_use":
                        wd.observe_tool_use(str(tool_use_id), str(tool_name or ""))
                    elif kind in ("tool_result", "tool_error"):
                        wd.observe_tool_result(str(tool_use_id))
                # Block on pause_gate before queuing the event.
                # When paused, the headless session's thread blocks here,
                # preventing further LLM API calls. The abort check
                # ensures we unblock when stop/takeover is received.
                if _pause_gate is not None:
                    while not _pause_gate.is_set():
                        if abort_controller.signal.aborted:
                            return
                        _pause_gate.wait(timeout=1)
                event_queue.put(tool_event)
            except Exception:  # nosec B110
                pass

        stdout = io.StringIO()
        # abort_controller was created above (before on_event) so the
        # closure can reference it. The cooperative-cancellation handle
        # allows us to signal the headless session to unwind on
        # timeout / teardown / operator stop.

        # Build the gap watchdog now that the abort
        # controller exists. ``tool_watchdog_state`` captures the
        # watchdog handle + a trip flag so the polling loop can react.
        def _build_tool_watchdog() -> Any:
            try:
                from clawcodex_ext.diagnostics.freeze_config import (
                    FreezeSettings,
                    resolve_freeze_settings,
                )

                settings = FreezeSettings(
                    agent_loop_timeout_s=float(self.config.agent_loop_timeout_s),
                    tool_timeout_s=float(self.config.tool_timeout_s),
                )
                real_settings = resolve_freeze_settings()
                overrides = {}
                for name in (
                    "turn_timeout_s",
                    "permission_timeout_s",
                    "threshold_s",
                    "dump_dir",
                ):
                    if getattr(settings, name, None) in (None, 0.0):
                        val = getattr(real_settings, name, None)
                        if val not in (None, 0.0):
                            overrides[name] = val
                if overrides:
                    import dataclasses as _dc

                    settings = _dc.replace(settings, **overrides)
            except Exception:
                logger.warning("tool-gap watchdog config build failed", exc_info=True)
                settings = None  # type: ignore[assignment]

            def _on_trip(res: Any, elapsed_s: float, tool_use_id: str) -> None:
                tool_watchdog_state["tripped"] = True
                tool_watchdog_state["last_trip"] = {
                    "tool_name": res.tool_name,
                    "timeout_s": res.timeout_s,
                    "elapsed_s": round(elapsed_s, 3),
                    "tool_use_id": tool_use_id,
                }
                append_debug_event(
                    debug_log_path,
                    "query_runner.tool_timeout",
                    run_id=self.config.run_id,
                    tool=res.tool_name,
                    tool_use_id=tool_use_id,
                    timeout_s=res.timeout_s,
                    elapsed_s=round(elapsed_s, 3),
                    reason=f"tool_timeout:{res.tool_name}",
                )

            try:
                return ToolGapWatchdog(
                    abort_controller=abort_controller,
                    settings=settings,  # type: ignore[arg-type]
                    explicit_overrides=self.config.tool_timeout_overrides,
                    on_trip=_on_trip,
                    logger=logger,
                )
            except Exception:
                return None

        tool_watchdog_state["wd"] = _build_tool_watchdog()
        session_opts = HeadlessSessionOptions(
            prompt=self.config.prompt,
            workspace_root=Path(self.config.workspace),
            provider_name=self.config.provider,
            model=self.config.model,
            max_turns=self.config.max_turns,
            permission_mode=self.config.permission_mode,
            stdout=stdout,
            stderr=stdout,
            on_event=on_event,
            env=self.config.env or {},
            append_system_prompt=self.config.append_system_prompt,
            abort_controller=abort_controller,
            agent_id=self.config.agent_id,
            runtime_tasks=self.config.runtime_tasks,
            resume_session_id=self.config.resume_session_id,
        )

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, run_headless_session, session_opts)
        next_heartbeat_at = time.monotonic() + 30.0

        def convert_tool_event(ev: Any) -> QueryEvent | None:
            kind = getattr(ev, "kind", None)
            tool_name = getattr(ev, "tool_name", "")
            tool_input = getattr(ev, "tool_input", None)
            tool_use_id = getattr(ev, "tool_use_id", None)
            if kind == "tool_use" and tool_use_id and tool_name:
                tool_names_by_id[str(tool_use_id)] = str(tool_name)
            elif not tool_name and tool_use_id:
                tool_name = tool_names_by_id.get(str(tool_use_id), "")
            tool_output = getattr(ev, "tool_output", None)
            is_error = getattr(ev, "is_error", False)
            error = getattr(ev, "error", None)

            if kind == "tool_use":
                return ToolCallEvent(
                    tool_name=tool_name,
                    params=tool_input or {},
                    tool_use_id=tool_use_id,
                )
            if kind in {"tool_result", "tool_error"}:
                result = {
                    "output": tool_output,
                    "is_error": bool(is_error) or kind == "tool_error",
                }
                if error is not None:
                    result["error"] = error
                return ToolResultEvent(
                    tool_name=tool_name,
                    result=result,
                    tool_use_id=tool_use_id,
                )
            return None

        # Drain the event queue while the headless session runs in the background.
        # A short timeout lets us poll for completion without busy-waiting.
        # F-REC: when a capture handle is wired, forward each drained
        # :class:`QueryEvent` to the recorder. The forwarder is a pure
        # function that returns ``None`` for events it cannot translate,
        # so a recording-disabled path pays only one ``if`` check per
        # event.
        capture = getattr(self.config, "capture", None)
        if capture is not None:
            try:
                from extensions.recording.query_forwarder import forward_event  # pylint: disable=import-error
            except Exception:  # pragma: no cover - import failure fallback
                forward_event = None  # type: ignore[assignment]
        else:
            forward_event = None
        #
        # The polling loop also enforces ``timeout_s`` from
        # ``QueryConfig`` (default 1800 s). ``asyncio.wait_for`` cannot cancel an executor
        # future — ``future.done()`` would stay False even after
        # ``wait_for`` raised — so the budget check lives INSIDE the
        # loop instead of wrapping the final ``await future``. On
        # timeout we break out and surface ``exit_code=124``.
        timeout_s = self.config.timeout_s
        loop_started_at = time.monotonic()
        timed_out = False
        # Stall-watchdog bookkeeping. ``stdout.tell()`` is
        # O(1) (write position), unlike ``getvalue()`` which copies the
        # whole buffer — safe to poll every loop iteration.
        stall_timeout_s = self.config.stall_timeout_s
        stall_warn_s = self.config.stall_warn_s
        stalled = False
        stall_warned_at: float | None = None  # activity mark of the warned episode
        last_stdout_pos = 0
        last_stdout_change_at = loop_started_at
        try:
            while True:
                # Wall-clock budget check at every iteration so
                # timeout_s / agent_loop_timeout_s are enforced even
                # when events are continuously flowing (the except
                # queue.Empty branch is only hit when the queue is
                # idle).  GNU exit code 124 = timeout.
                _now = time.monotonic()
                if timeout_s > 0 and (_now - loop_started_at) >= timeout_s:  # pylint: disable=chained-comparison
                    timed_out = True
                    break
                _agent_loop_budget = self.config.agent_loop_timeout_s
                if _agent_loop_budget > 0 and (_now - loop_started_at) >= _agent_loop_budget:  # pylint: disable=chained-comparison
                    timed_out = True
                    break
                # Drain control-socket commands at every polling
                # iteration so pause/stop/inject/resume take effect within
                # ~60ms (the queue-timeout + sleep interval) instead of
                # waiting for the next tool event. The callback is
                # non-blocking (uses ``get_nowait`` internally).
                if self.config.control_drain_fn is not None:
                    _drain_signal = self.config.control_drain_fn()
                    if _drain_signal is not None:
                        abort_controller.abort(f"operator_{_drain_signal}")
                        break
                # If the session is paused, block here in a
                # drain-and-wait loop (the callback handles resume/stop
                # delivery while paused). This prevents the stream from
                # yielding events while paused AND keeps the command
                # queue draining so resume is not deadlocked.
                if self.config.pause_wait_fn is not None:
                    await self.config.pause_wait_fn()
                try:
                    ev: Any = event_queue.get(timeout=0.05)
                    event = convert_tool_event(ev)
                    if event is not None:
                        if forward_event is not None:
                            forward_event(capture, event)
                        yield event
                except queue.Empty:
                    if future.done():
                        while True:
                            try:
                                ev = event_queue.get_nowait()
                            except queue.Empty:
                                break
                            event = convert_tool_event(ev)
                            if event is not None:
                                if forward_event is not None:
                                    forward_event(capture, event)
                                yield event
                        break
                    now = time.monotonic()
                    # Tick the tool-gap watchdog. Trip
                    # is callback-driven (sets ``tool_watchdog_state``
                    # + signals abort), so the loop just needs to
                    # honour the tripped flag once the headless
                    # session has had a chance to unwind.
                    wd = tool_watchdog_state.get("wd")
                    if wd is not None:
                        try:
                            wd.tick(now=now)
                        except Exception:  # nosec B110
                            pass
                    # Budget enforcement inside the polling
                    # loop. Conventional GNU ``timeout`` exit code (124)
                    # distinguishes "wall-clock budget exhausted" from
                    # other non-zero exits.
                    if timeout_s > 0 and (now - loop_started_at) >= timeout_s:  # pylint: disable=chained-comparison
                        timed_out = True
                        break
                    # Outer agent-loop budget
                    # (``agent_loop_timeout_s``). Surfaces as
                    # ``exit_code=124`` so callers can tell "outer
                    # budget" from "stall" (125) and "tool timeout"
                    # (126). The narrower 600 s default lets long
                    # plugin installs abort before the 1800 s
                    # ``timeout_s`` retroactively kicks in.
                    agent_loop_budget = self.config.agent_loop_timeout_s
                    if agent_loop_budget > 0 and (now - loop_started_at) >= agent_loop_budget:  # pylint: disable=chained-comparison
                        timed_out = True
                        break
                    # Stream-stall watchdog. "Activity" is
                    # any tool event (``last_event_at``, updated by
                    # ``on_event``) or stdout growth (streamed text).
                    # When both signals have been flat for
                    # ``stall_timeout_s`` we stop waiting: the observed
                    # hang mode (provider accepted the request, then
                    # never sent a single chunk) otherwise burns the
                    # entire wall-clock budget doing nothing.
                    stdout_pos = stdout.tell()
                    if stdout_pos != last_stdout_pos:
                        last_stdout_pos = stdout_pos
                        last_stdout_change_at = now
                    last_activity_at = max(last_event_at, last_stdout_change_at)
                    tool_in_flight = bool(wd is not None and wd.has_pending())
                    # WARN tier: a clear diagnosis within stall_warn_s of
                    # the silence starting — long before the abort tier
                    # would consider acting. One event per silence
                    # episode; activity re-arms it.
                    if tool_in_flight:
                        # The per-tool watchdog owns tool_use -> tool_result
                        # gaps. The stream-stall budget must not preempt it.
                        stall_warned_at = None
                    elif stall_warn_s > 0:
                        if stall_warned_at is not None and last_activity_at > stall_warned_at:
                            stall_warned_at = None  # activity resumed — re-arm
                        if stall_warned_at is None and (now - last_activity_at) >= stall_warn_s:
                            stall_warned_at = last_activity_at
                            logger.warning(
                                "query stall suspected run_id=%s: no activity for %.0fs (abort tier at %.0fs)",
                                self.config.run_id,
                                now - last_activity_at,
                                stall_timeout_s,
                            )
                            append_debug_event(
                                debug_log_path,
                                "query_runner.stall_suspected",
                                run_id=self.config.run_id,
                                stall_warn_s=stall_warn_s,
                                stall_timeout_s=stall_timeout_s,
                                seconds_since_last_event=round(now - last_event_at, 3),
                                seconds_since_stdout_change=round(now - last_stdout_change_at, 3),
                                stdout_len=stdout_pos,
                                tool_events=tool_event_count,
                            )
                    if stall_timeout_s > 0 and not tool_in_flight:
                        if (now - last_activity_at) >= stall_timeout_s:
                            stalled = True
                            append_debug_event(
                                debug_log_path,
                                "query_runner.stall_detected",
                                run_id=self.config.run_id,
                                stall_timeout_s=stall_timeout_s,
                                seconds_since_last_event=round(now - last_event_at, 3),
                                seconds_since_stdout_change=round(now - last_stdout_change_at, 3),
                                seconds_since_start=round(now - loop_started_at, 3),
                                stdout_len=stdout_pos,
                                tool_events=tool_event_count,
                            )
                            break
                    if now >= next_heartbeat_at:
                        append_debug_event(
                            debug_log_path,
                            "query_runner.heartbeat",
                            run_id=self.config.run_id,
                            future_done=future.done(),
                            seconds_since_last_event=round(now - last_event_at, 3),
                            stdout_len=len(stdout.getvalue()),
                            tool_events=tool_event_count,
                        )
                        next_heartbeat_at = now + 30.0
                    await asyncio.sleep(0.01)
        finally:
            # Zombie-run fix: if the consumer goes away while the headless
            # future is still running — wall-clock budget break above, the
            # orchestrator's outer ``asyncio.wait_for`` timeout, or an
            # ``issue stop`` task.cancel() (both surface here as generator
            # teardown) — trip the abort controller so the session unwinds
            # at its next cancellation point instead of running (and
            # spawning workers / burning tokens) unsupervised for minutes.
            if not future.done():
                try:
                    abort_controller.abort("query_runner teardown (timeout/cancel)")
                    append_debug_event(
                        debug_log_path,
                        "query_runner.abort_signalled",
                        run_id=self.config.run_id,
                        seconds_since_start=round(time.monotonic() - loop_started_at, 3),
                    )
                except Exception:  # nosec B110
                    pass
            # Unblock the pause gate so the headless session's
            # on_event callback can exit its wait loop and check the
            # abort signal. Without this, a paused session would
            # deadlock on teardown.
            if _pause_gate is not None:
                _pause_gate.set()

        if timed_out:
            # The abort controller (finally above) asks the still-running
            # headless future to unwind cooperatively; the caller sees a
            # definitive ``SessionComplete`` right away and the debug log
            # carries enough context for postmortem.
            append_debug_event(
                debug_log_path,
                "query_runner.timeout",
                run_id=self.config.run_id,
                timeout_s=timeout_s,
                seconds_since_start=round(time.monotonic() - loop_started_at, 3),
                stdout_len=len(stdout.getvalue()),
            )
            exit_code = 124
        elif tool_watchdog_state.get("tripped"):
            # Tool-level auto-recovery. The
            # ``tool_watchdog_state['last_trip']`` payload is the
            # most-recent trip so postmortem tooling can attribute
            # the run outcome to a specific tool call. 126 is
            # distinct from 124 / 125.
            last_trip = tool_watchdog_state.get("last_trip") or {}
            append_debug_event(
                debug_log_path,
                "query_runner.tool_timeout_final",
                run_id=self.config.run_id,
                tool=last_trip.get("tool_name"),
                tool_use_id=last_trip.get("tool_use_id"),
                timeout_s=last_trip.get("timeout_s"),
                elapsed_s=last_trip.get("elapsed_s"),
            )
            exit_code = 126
        elif stalled:
            # Distinct from 124 so downstream can tell "budget spent
            # while working" from "provider went silent". The finally
            # block above already tripped the abort controller; the run
            # fails fast and the orchestrator's normal retry machinery
            # takes over — recovery in ~stall_timeout_s instead of the
            # full wall-clock budget.
            exit_code = 125
        elif abort_controller.signal.aborted:
            # Operator stop/takeover via control_drain_fn. The
            # abort was tripped by the drain callback, not by a timeout
            # or stall. Don't block on ``await future`` — the headless
            # session is unwinding cooperatively in the background.
            append_debug_event(
                debug_log_path,
                "query_runner.operator_abort",
                run_id=self.config.run_id,
                reason=abort_controller.signal.reason,
                seconds_since_start=round(time.monotonic() - loop_started_at, 3),
            )
            exit_code = -1
        else:
            try:
                exit_code = await future
            except SystemExit as exc:
                code = exc.code
                exit_code = code if isinstance(code, int) else 1
        result_text = stdout.getvalue()
        if result_text:
            if forward_event is not None:
                forward_event(capture, TextDelta(content=result_text))
            yield TextDelta(content=result_text)

        reason = "success" if exit_code == 0 else f"exit_code={exit_code}"
        if forward_event is not None:
            forward_event(capture, SessionComplete(reason=reason))
        yield SessionComplete(reason=reason)

    async def run(self) -> dict[str, Any]:
        """Run to completion, return final result."""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        async for event in self.stream():
            if isinstance(event, TextDelta):
                text_parts.append(event.content)
            elif isinstance(event, ToolCallEvent):
                tool_calls.append({"name": event.tool_name, "params": event.params})
            elif isinstance(event, SessionComplete):
                return {
                    "text": "".join(text_parts),
                    "reason": event.reason,
                    "tool_calls": tool_calls,
                }
        return {"text": "".join(text_parts), "reason": "unknown", "tool_calls": tool_calls}
