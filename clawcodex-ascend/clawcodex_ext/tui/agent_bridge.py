# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Bridge between the agent loop and the Textual UI.

The canonical agent loop (:func:`src.query.query.query`) is async and
performs blocking HTTP calls under the hood, so this module runs it on
a worker thread via the F.1 adapter
(:func:`src.query.agent_loop_compat.run_query_as_agent_loop`). The
worker thread owns its own fresh asyncio loop (NOT Textual's main
loop, which would block UI rendering during model streams). This
module owns that thread plus the translation layer that marshals
events back to the Textual screen:

* ``on_event(ToolEvent)``   → :class:`ToolEventMessage`.
* ``on_text_chunk(str)``    → :class:`AssistantChunk` (live streaming).
* permission request        → :class:`PermissionRequested` + blocking
  wait on a :class:`threading.Event`, letting the worker thread unblock
  only when the user has interacted with
  :class:`src.tui.screens.permission_modal.PermissionModal`.

Keeping this logic out of :class:`src.tui.app.ClawCodexTUI` lets unit
tests drive :class:`AgentBridge` with a fake agent loop (see
``tests/tui/test_agent_bridge.py``).
"""

from __future__ import annotations
# pylint: disable=C0302,C0412,E0611,W0621

import threading
import time
from typing import Any, Callable

from src.agent import Session
from src.tool_system.renderers import AgentLoopResult, ToolEvent
from clawcodex_ext.diagnostics.freeze_config import (
    DEFAULT_FREEZE_SETTINGS,
    resolve_freeze_settings,
)
from clawcodex_ext.query.agent_loop_compat import (
    build_effective_system_prompt,
    run_query_as_agent_loop,
)
from clawcodex_ext.goal.evaluator import GoalEvaluationError
from src.tool_system.context import ToolContext
from src.tool_system.registry import ToolRegistry
from src.utils.abort_controller import AbortController, AbortError
from clawcodex_ext.types.messages import SystemMessage as QuerySystemMessage

from .messages import (
    AdvisorEventMessage,
    AgentRunFinished,
    AgentRunStarted,
    AskUserQuestionRequested,
    AssistantChunk,
    AssistantMessage,
    MultiModelEvent,
    PermissionRequested,
    SystemNotice,
    ToolEventMessage,
)
from .state import AppState


def _resolve_permission_timeout_s() -> float:
    """Layer-0 modal deadline (F-108 P108-A / P108-E).

    Resolved honouring ``FreezeSettings.permission_timeout_s`` →
    ``$CLAWCODEX_PERMISSION_TIMEOUT`` → the dataclass default
    (30 s). ``0`` disables the timeout entirely (legacy unbounded
    ``done.wait()``). Kept as a function so the bridge re-resolves
    on every prompt (cheap; settings can change between runs).
    """
    try:
        return float(resolve_freeze_settings().permission_timeout_s)
    except Exception:
        return DEFAULT_FREEZE_SETTINGS.permission_timeout_s


class AgentBridge:
    """Owns the agent-loop worker thread on behalf of the TUI."""

    def __init__(
        self,
        *,
        post_message: Callable[[Any], None],
        session: Session,
        provider: Any,
        tool_registry: ToolRegistry,
        tool_context: ToolContext,
        app_state: AppState,
        run_worker: Callable[..., Any],
        max_turns: int = 20,
        stream: bool = True,
        tail_follower: Any | None = None,
        append_system_prompt: str = "",
        runtime_permission_controller: Any | None = None,
        reset_goal_progress: bool = False,
    ) -> None:
        self._post = post_message
        self._session = session
        self._provider = provider
        self._tool_registry = tool_registry
        self._tool_context = tool_context
        self._state = app_state
        self._run_worker = run_worker
        self._max_turns = max_turns
        self._stream = stream
        self._append_system_prompt = append_system_prompt
        self._goal_unsubscribe: Callable[[], None] | None = None
        self._goal_bound_service: Any | None = None
        self._goal_bound_thread_id: str | None = None
        self._goal_runtime: Any | None = None
        self._goal_active_id: str | None = None
        self._goal_active_since: float | None = None
        self._busy_lock = threading.Lock()
        self._busy = False
        # Runtime permission controller — the unified chokepoint for
        # Shift+Tab cycles. Stored on the bridge so ``replace_runtime``
        # can rewire it if the tool context is swapped mid-session.
        # ``None`` is tolerated: callers that pre-date the unified
        # controller (mostly older tests) get the legacy direct-mutation
        # path through ``action_cycle_permission_mode``.
        self._runtime_permission_controller = runtime_permission_controller
        # Per-run abort controller. Created fresh in :meth:`submit` and
        # tripped by :meth:`cancel` (ESC from the prompt). The agent
        # loop checks the signal at safe boundaries; the streaming
        # callback also raises :class:`AbortError` on the worker thread
        # to tear down an in-flight HTTP stream cleanly.
        self._abort_controller: AbortController | None = None
        # Wire permission handler: the tool dispatcher calls this from
        # the worker thread, we post to the UI and block on an Event.
        tool_context.permission_handler = self._permission_handler
        # Also stamp the controller's ``default_permission_handler`` on
        # the tool context so a later cycle-out-of-bypass restores this
        # exact callable. Idempotent with the TUI app's __init__ patch
        # — the bridge is the authoritative owner of the handler, so
        # any post-construction wiring wins.
        if tool_context.default_permission_handler is None:
            tool_context.default_permission_handler = self._permission_handler
        # Wire AskUserQuestion handler: the AskUserQuestion tool calls
        # ``context.ask_user(questions)`` from the worker thread. We
        # mirror the permission pattern — post a request to the UI,
        # block on an Event, return the answer dict when the modal
        # resolves. Without this wiring the tool would receive an
        # empty ``{}`` (or nothing at all) and the agent loop would
        # never see the user's choices.
        tool_context.ask_user = self._ask_user_handler
        self._bind_goal_state(
            reset_baseline=True,
            reset_progress=reset_goal_progress,
        )
        # Advisor IDs we've already mirrored to the UI. The high-level
        # SDK stream doesn't give us per-event hooks for server tools,
        # so the bridge inspects the assembled conversation after each
        # turn — without dedup, every turn would re-emit the same
        # events for blocks that landed in earlier turns.
        #
        # ``_last_scanned_msg_index`` is an optimization: iterating the
        # full message list on every turn is O(N*B). Tracking how far
        # we've scanned lets us start at the new tail instead. Reset
        # together with ``_emitted_advisor_ids`` via
        # ``reset_advisor_dedup`` whenever the message list is
        # truncated or replaced.
        self._emitted_advisor_ids: set[str] = set()
        self._last_scanned_msg_index: int = 0
        # Tail-follower for --resume: watches the backgrounded agent's
        # transcript and posts new lines to the UI in real time.
        self._tail_follower = tail_follower
        self._multimodel_listener = None
        self._attach_multimodel_listener(provider)
        self._tail_thread: threading.Thread | None = None
        if self._tail_follower is not None:
            self._start_tail_follower()

    def replace_runtime(self, *, provider: Any, tool_registry: ToolRegistry, tool_context: ToolContext) -> None:
        self._unbind_goal_state()
        old_provider = self._provider
        if self._multimodel_listener is not None and hasattr(old_provider, "remove_event_listener"):
            old_provider.remove_event_listener(self._multimodel_listener)
        self._provider = provider
        self._attach_multimodel_listener(provider)
        self._tool_registry = tool_registry
        self._tool_context = tool_context
        tool_context.permission_handler = self._permission_handler
        if tool_context.default_permission_handler is None:
            tool_context.default_permission_handler = self._permission_handler
        tool_context.ask_user = self._ask_user_handler
        self._bind_goal_state(reset_baseline=False, reset_progress=False)

    def replace_session(self, session: Session | None) -> bool:
        """Rebind session state when idle; return ``False`` while busy."""

        if session is None:
            return False
        with self._busy_lock:
            if self._busy:
                return False
            previous_id = getattr(self._session, "session_id", None)
            next_id = getattr(session, "session_id", None)
            self._unbind_goal_state()
            self._session = session
            if str(previous_id or "") != str(next_id or ""):
                self._goal_active_id = None
                self._goal_active_since = None
            self.reset_advisor_dedup()
            self._bind_goal_state(reset_baseline=True, reset_progress=True)
            return True

    def _bind_goal_state(self, *, reset_baseline: bool, reset_progress: bool) -> None:
        """Bind the live session id and mirror GoalService snapshots to AppState."""

        from clawcodex_ext.runtime.tool_context_binding import bind_tool_context_runtime

        bind_tool_context_runtime(
            self._tool_context,
            tool_registry=self._tool_registry,
            session=self._session,
            provider=self._provider,
        )
        # Claude Code goals are session-scoped. A stale protocol/test override
        # must not pin a resumed TUI to the previous session's goal row.
        self._tool_context.goal_thread_id = None

        thread_id = getattr(self._tool_context, "session_id", None)
        if thread_id is None:
            self._state.set_goal_status(None)
            return
        thread_id = str(thread_id)

        service = getattr(self._tool_context, "goal_service", None)
        if service is None:
            try:
                from clawcodex_ext.goal.service import GoalService

                service = GoalService()
                self._tool_context.goal_service = service
            except Exception:  # nosec B110
                self._state.set_goal_status(None)
                return

        try:
            from clawcodex_ext.goal.runtime import (
                goal_runtime_for_context,
                restore_goal_runtime_after_session_resume,
            )

            if reset_progress:
                runtime = restore_goal_runtime_after_session_resume(
                    self._tool_context,
                )
            else:
                runtime = goal_runtime_for_context(self._tool_context)
                if runtime is not None:
                    runtime.restore_after_resume()
        except Exception:  # nosec B110
            runtime = None

        if reset_baseline:
            self._goal_active_id = None
            self._goal_active_since = None

        self._goal_bound_service = service
        self._goal_bound_thread_id = thread_id
        self._goal_runtime = runtime

        def _sync(goal: Any | None) -> None:
            if self._goal_bound_service is service and self._goal_bound_thread_id == thread_id:
                self._sync_goal_snapshot(goal)

        try:
            self._goal_unsubscribe = service.subscribe(
                thread_id,
                _sync,
                emit_current=True,
            )
        except Exception:
            self._goal_unsubscribe = None
            self._state.set_goal_status(None)

    def _unbind_goal_state(self) -> None:
        unsubscribe = self._goal_unsubscribe
        self._goal_unsubscribe = None
        if unsubscribe is not None:
            try:
                unsubscribe()
            except Exception:  # nosec B110
                pass

        runtime = self._goal_runtime
        service = self._goal_bound_service
        if runtime is not None and service is not None:
            try:
                service.unregister_runtime(runtime)
            except Exception:  # nosec B110
                pass
        if getattr(self._tool_context, "goal_runtime", None) is runtime:
            self._tool_context.goal_runtime = None

        self._goal_runtime = None
        self._goal_bound_service = None
        self._goal_bound_thread_id = None

    def _sync_goal_snapshot(self, goal: Any | None) -> None:
        if goal is None:
            self._goal_active_id = None
            self._goal_active_since = None
            self._state.set_goal_status(None)
            return

        try:
            from clawcodex_ext.goal.model import ThreadGoalStatus
            from clawcodex_ext.goal.protocol import ThreadGoalDTO

            payload = ThreadGoalDTO.from_model(goal).to_dict()
            payload["goalId"] = goal.goal_id
            if goal.status is ThreadGoalStatus.ACTIVE:
                if self._goal_active_id != goal.goal_id or self._goal_active_since is None:
                    self._goal_active_id = goal.goal_id
                    self._goal_active_since = time.time() - max(goal.time_used_seconds, 0)
                payload["activeSince"] = self._goal_active_since
            else:
                self._goal_active_id = None
                self._goal_active_since = None
            self._state.set_goal_status(payload)
        except Exception:  # nosec B110
            self._state.set_goal_status(None)

    def _attach_multimodel_listener(self, provider: Any) -> None:
        if not hasattr(provider, "add_event_listener") or not hasattr(provider, "slots"):
            self._multimodel_listener = None
            return
        slots = [slot.name for slot in provider.slots]

        def listener(kind: str, payload: dict[str, Any]) -> None:
            if kind == "progress":
                self._post(MultiModelEvent("progress", payload["slot"], payload.get("chunk", "")))
            elif kind == "complete":
                self._post(MultiModelEvent("complete", result=payload["result"]))
            elif kind == "aggregated":
                self._post(MultiModelEvent("aggregated", results=payload.get("results")))

        self._multimodel_listener = listener
        provider.add_event_listener(listener)
        self._post(MultiModelEvent("start", results=slots))

    def reset_advisor_dedup(self) -> None:
        """Drop the advisor-dedup state.

        Call this when conversation history is wiped or summarized so
        the IDs we tracked no longer correspond to anything in the
        live message list — keeping them around leaks memory (bounded:
        one UUID per advisor call ever made) and risks suppressing a
        legitimate re-render if a post-compact replay reuses an ID.
        The TUI wires this on ``/clear`` (``tui/app.py:__clear__`` and
        the idle-return "clear" choice). Other reset paths
        (``/compact``, programmatic message wipe) don't yet trigger it;
        the leak is harmless until they do.
        """
        self._emitted_advisor_ids.clear()
        self._last_scanned_msg_index = 0

    # ---- tail-follower for --resume ----
    def _start_tail_follower(self) -> None:
        """Spawn a daemon thread that watches the transcript for new lines."""
        self._tail_thread = threading.Thread(
            target=self._run_tail_follower,
            name="tail-follower",
            daemon=True,
        )
        self._tail_thread.start()

    def _run_tail_follower(self) -> None:
        """Run the TailFollower in a dedicated thread with its own event loop."""
        import asyncio
        from .messages import AssistantMessage, ToolEventMessage

        follower = self._tail_follower
        if follower is None:
            return

        async def _follow() -> None:
            try:
                await follower.start(from_offset=follower._offset)
                async for msg_dict in follower:
                    if msg_dict is None:
                        continue
                    # Detect background agent completion marker
                    if msg_dict.get("role") == "system" and msg_dict.get("content") == "__background_complete__":
                        self._post(
                            AgentRunFinished(
                                response_text="",
                                num_turns=0,
                                usage=None,
                                error=None,
                            )
                        )
                        break
                    role = msg_dict.get("role", "")
                    content = msg_dict.get("content", "")
                    if role == "assistant":
                        # Streaming chunk or full message.
                        text = content if isinstance(content, str) else ""
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text += item.get("text", "")
                        if text:
                            self._post(AssistantMessage(text=text))
                    elif role in ("tool_use", "tool_result"):
                        self._post(
                            ToolEventMessage(
                                kind=role,
                                tool_name=msg_dict.get("name", ""),
                                tool_input=msg_dict.get("input"),
                                tool_output=msg_dict.get("content"),
                                tool_use_id=msg_dict.get("id") or msg_dict.get("tool_use_id"),
                            )
                        )
            except Exception:  # nosec B110
                pass

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_follow())
        finally:
            loop.close()

    # ---- public API ----
    @property
    def busy(self) -> bool:
        return self._busy

    def _set_in_agent_loop(self, value: bool) -> None:
        """Keep the cron scheduler's mutable busy gate in sync."""

        flag = self._tool_context._in_agent_loop
        if flag is not None:
            flag.value = value

    def submit(self, prompt: str) -> bool:
        """Queue ``prompt`` for the agent. Returns False if busy."""
        ## _log(f'[agent_bridge] submit called: {prompt}')

        with self._busy_lock:
            if self._busy:
                ## _log(f'[agent_bridge] busy, returning False')
                return False
            self._busy = True
            self._set_in_agent_loop(True)
            self._abort_controller = AbortController()
            ## _log(f'[agent_bridge] acquired busy lock')
            # Plumb the controller onto the tool context BEFORE we spawn
            # the worker. Tools (Bash supervisor, Agent subagents, the
            # streaming executor, tool hooks) read
            # ``context.abort_controller`` to learn whether the user has
            # asked to interrupt; without this assignment they see
            # ``None`` and run to completion regardless of ESC. The Agent
            # subagent path is the worst case — ``run_agent`` inherits
            # the parent's controller via ``parent_context.abort_controller``,
            # so a missing field here forces the subagent to mint a fresh
            # controller that is never tripped by ESC.
            self._tool_context.abort_controller = self._abort_controller

        self._session.conversation.add_user_message(prompt)
        ## _log(f'[agent_bridge] posted AgentRunStarted, calling run_worker')
        self._post(AgentRunStarted(prompt=prompt))
        self._state.set_thinking(True, verb="Synthesizing")
        ## _log(f'[agent_bridge] about to call self._run_worker')
        self._run_worker(
            self._run_agent_in_thread,
            thread=True,
            exclusive=True,
            name="agent-loop",
        )
        ## _log(f'[agent_bridge] run_worker called')
        return True

    def continue_goal_if_idle(self) -> bool:
        """Claim and start the active session goal without a second user prompt."""

        try:
            from clawcodex_ext.goal.runtime import goal_runtime_for_context

            runtime = goal_runtime_for_context(self._tool_context)
            continuation = runtime.continue_if_idle() if runtime is not None else None
        except Exception:
            return False
        if runtime is None or continuation is None:
            return False

        with self._busy_lock:
            if self._busy or not runtime.claim_continuation(continuation):
                return False
            self._busy = True
            self._set_in_agent_loop(True)
            self._abort_controller = AbortController()
            self._tool_context.abort_controller = self._abort_controller

        try:
            for message in continuation.messages:
                add_existing = getattr(self._session.conversation, "add_existing_message", None)
                if callable(add_existing):
                    add_existing(message)
                else:
                    self._session.conversation.add_message(message.role, message.content)

            goal = runtime.service.get_goal(runtime.thread_id)
            directive = goal.objective if goal is not None else "/goal"
            self._post(AgentRunStarted(prompt=directive))
            self._state.set_thinking(True, verb="Pursuing goal")
            self._run_worker(
                self._run_agent_in_thread,
                thread=True,
                exclusive=True,
                name="agent-loop",
            )
            return True
        except Exception:
            with self._busy_lock:
                self._busy = False
                self._set_in_agent_loop(False)
                self._abort_controller = None
            self._state.set_thinking(False)
            return False

    def _effective_max_turns(self) -> int:
        """Keep evaluator goals running until achieved or explicitly stopped."""

        try:
            from clawcodex_ext.goal.model import GoalCompletionMode, ThreadGoalStatus
            from clawcodex_ext.goal.runtime import goal_runtime_for_context

            runtime = goal_runtime_for_context(self._tool_context)
            goal = runtime.service.get_goal(runtime.thread_id) if runtime is not None else None
            if (
                goal is not None
                and goal.status is ThreadGoalStatus.ACTIVE
                and goal.completion_mode is GoalCompletionMode.EVALUATOR
            ):
                return 0
        except Exception:  # nosec B110
            pass
        return self._max_turns

    def cancel(self, reason: str = "user_interrupt") -> bool:
        """Trip the active run's abort signal. Returns True if a run was cancelled.

        Safe to call from any thread. The agent loop checks the signal
        at the next safe boundary (next turn, next tool call, next
        streaming chunk) and unwinds; ``_run_agent_in_thread`` then
        posts an ``AgentRunFinished`` and clears the busy flag.
        """

        # This is called from hot key handlers on Textual's UI thread.
        # Do not wait on ``_busy_lock`` here: the worker may briefly hold
        # it while finishing a run, and a contended lock makes Ctrl+C /
        # Ctrl+B feel ignored. Reading these references is atomic under
        # the GIL; a stale controller is harmless because abort() is
        # idempotent.
        controller = self._abort_controller if self._busy else None
        if controller is None:
            return False
        controller.abort(reason)
        return True

    # ---- worker implementation ----
    def _run_agent_in_thread(self) -> None:
        ## _log(f'[agent_bridge] _run_agent_in_thread STARTED')
        controller = self._abort_controller
        agent_type = self._tool_context.agent_type or ""

        def _on_event(event: ToolEvent) -> None:
            # Keep the app_state in sync so StatusLine / overlays can
            # observe in-progress tool ids.
            if event.kind == "tool_use" and event.tool_use_id:
                self._state.mark_tool_started(event.tool_use_id)
            elif event.kind in ("tool_result", "tool_error") and event.tool_use_id:
                self._state.mark_tool_finished(event.tool_use_id)
            self._post(
                ToolEventMessage(
                    kind=event.kind,
                    tool_name=event.tool_name,
                    tool_input=_safe_copy(event.tool_input),
                    tool_output=_safe_copy(event.tool_output),
                    tool_use_id=event.tool_use_id,
                    is_error=event.is_error,
                    error=event.error,
                )
            )

        def _on_text(chunk: str) -> None:
            # Bail out of the provider stream as soon as the user hits
            # ESC; raising from the callback breaks out of the
            # Anthropic SDK's ``with client.messages.stream(...)``
            # context manager and tears down the HTTP connection.
            if controller is not None and controller.signal.aborted:
                raise AbortError(controller.signal.reason or "user_interrupt")
            ## _log(f'[agent_bridge] _on_text: {chunk[:30] if chunk else "empty"}...')
            self._state.append_streaming_text(chunk)
            self._post(AssistantChunk(text=chunk, agent_name=agent_type))

        def _on_thinking(chunk: str) -> None:
            """Handle thinking content chunks from the provider."""
            ## _log(f'[agent_bridge] _on_thinking called with: {chunk[:50] if chunk else "empty"}...')
            from .messages import ThinkingChunk

            self._post(ThinkingChunk(text=chunk))

        try:
            # Ch5/F.3 cutover: route TUI through the canonical query()
            # loop via the F.1 adapter. The Textual ``@work(thread=True)``
            # worker doesn't have an asyncio loop, so we spin up a fresh
            # one INSIDE the worker thread (NOT on the main loop —
            # ``@work(thread=False)`` would block Textual's UI rendering
            # during model streams). Pre-build the effective system
            # prompt (CLAUDE.md, style, git status) — legacy
            # run_agent_loop did this internally; the adapter doesn't.
            import asyncio as _asyncio
            from src.outputStyles import resolve_output_style

            _style_prompt = resolve_output_style(
                getattr(self._tool_context, "output_style_name", None),
                getattr(self._tool_context, "output_style_dir", None),
            ).prompt
            effective_system_prompt = build_effective_system_prompt(
                _style_prompt,
                self._tool_context,
            )
            if self._append_system_prompt:
                effective_system_prompt = f"{effective_system_prompt}\n\n{self._append_system_prompt}"

            def _persist(msg: Any) -> None:
                # BLOCKING #2 fix: persist FULL message (tool_use /
                # tool_result blocks included) so subsequent turns can
                # pair tool_use IDs to results. Plain
                # ``add_assistant_message`` loses block structure.
                # Critic S3: log failures instead of swallowing — a
                # persist failure means the conversation is corrupted
                # for the next turn (tool_use without tool_result will
                # 400 at the API). Surface it now, not later.
                try:
                    add_existing = getattr(
                        self._session.conversation,
                        "add_existing_message",
                        None,
                    )
                    if callable(add_existing):
                        add_existing(msg)
                    else:
                        self._session.conversation.add_message(msg.role, msg.content)
                    if isinstance(msg, QuerySystemMessage) and getattr(msg, "subtype", None) in {
                        "goal_evaluation",
                        "goal_achieved",
                        "goal_evaluator_error",
                    }:
                        subtype = getattr(msg, "subtype", None)
                        self._post(
                            SystemNotice(
                                text=str(msg.content),
                                style=(
                                    "light"
                                    if subtype == "goal_achieved"
                                    else "error"
                                    if subtype == "goal_evaluator_error"
                                    else "muted"
                                ),
                            )
                        )
                except Exception:
                    import logging

                    logging.getLogger(__name__).exception(
                        "Failed to persist message into conversation: role=%s; next-turn API may reject the call.",
                        getattr(msg, "role", "?"),
                    )

            _loop = _asyncio.new_event_loop()
            ## _log(f'[agent_bridge] about to call run_query_as_agent_loop')
            ## _log(f'[agent_bridge] initial_messages count: {len(list(self._session.conversation.messages))}')
            ## _log(f'[agent_bridge] stream={self._stream}, _on_text={"_on_text" if self._stream else None}')
            try:
                compat_result = _loop.run_until_complete(
                    run_query_as_agent_loop(
                        initial_messages=list(self._session.conversation.messages),
                        provider=self._provider,
                        tool_registry=self._tool_registry,
                        tool_context=self._tool_context,
                        system_prompt=effective_system_prompt,
                        max_turns=self._effective_max_turns(),
                        on_event=_on_event,
                        on_text_chunk=_on_text if self._stream else None,
                        on_thinking_chunk=_on_thinking if self._stream else None,
                        on_message=_persist,
                        abort_controller=controller,
                    )
                )
                ## _log(f'[agent_bridge] run_query_as_agent_loop returned, response_text={compat_result.response_text[:100] if compat_result.response_text else "empty"}, num_turns={compat_result.num_turns}')
                ## _log(f'[agent_bridge] result terminal: {compat_result.terminal}')
            finally:
                _loop.close()
            result = AgentLoopResult(
                response_text=compat_result.response_text,
                usage=(compat_result.usage if compat_result.num_turns > 0 else None),
                num_turns=compat_result.num_turns,
            )

            # Per-turn save: persist JSONL transcript only (lightweight).
            try:
                self._session.save_transcript()
            except Exception:  # nosec B110
                pass
        except GoalEvaluationError as exc:
            # The explicit SystemNotice emitted above is the user-facing
            # failure. Preserve it and finish without a second generic error.
            aggregate_usage = getattr(exc, "aggregate_usage", None)
            if isinstance(aggregate_usage, dict):
                self._state.usage["input_tokens"] = self._state.usage.get("input_tokens", 0) + int(
                    aggregate_usage.get("input_tokens", 0) or 0
                )
                self._state.usage["output_tokens"] = self._state.usage.get("output_tokens", 0) + int(
                    aggregate_usage.get("output_tokens", 0) or 0
                )
            try:
                self._session.save_transcript()
            except Exception:  # nosec B110
                pass
            self._post(
                AgentRunFinished(
                    response_text="",
                    num_turns=int(getattr(exc, "num_turns", 0) or 0),
                    usage=(aggregate_usage if isinstance(aggregate_usage, dict) else None),
                    error=None,
                )
            )
            self._finish()
            return
        except AbortError:
            self._post(
                AgentRunFinished(
                    response_text="",
                    num_turns=0,
                    usage=None,
                    error="Cancelled by user",
                )
            )
            self._finish()
            return
        except Exception as exc:  # pragma: no cover — surfaced to UI
            self._post(
                AgentRunFinished(
                    response_text="",
                    num_turns=0,
                    usage=None,
                    error=str(exc),
                )
            )
            self._finish()
            return

        self._post(AssistantMessage(text=result.response_text, agent_name=agent_type))
        # Surface any advisor activity from this run as transcript rows.
        # The Python provider path doesn't emit per-event hooks for
        # server tools (the SDK's high-level ``messages.stream`` only
        # signals text deltas to ``on_text_chunk``), so the bridge
        # inspects the final assembled assistant content. This is
        # idempotent across turns: ``_emit_advisor_events`` only posts
        # events for advisor blocks added during the current run.
        try:
            self._emit_advisor_events()
        except Exception:  # nosec B110
            # Never let an advisor-rendering issue mask the model's
            # actual response.
            pass
        if result.usage:
            try:
                self._state.usage.update(
                    {
                        "input_tokens": self._state.usage.get("input_tokens", 0)
                        + int(result.usage.get("input_tokens", 0) or 0),
                        "output_tokens": self._state.usage.get("output_tokens", 0)
                        + int(result.usage.get("output_tokens", 0) or 0),
                    }
                )
            except Exception:  # nosec B110
                pass
        # Mirror the client-side advisor token counts from tool_context
        # into ``state.usage`` so the StatusLine widget can surface them
        # next to the worker tokens. ``tool_context.advisor_*`` is
        # accumulated by ``AdvisorTool._advisor_call`` (PR #190) on every
        # client-side consultation; this is the TUI-side parity of the
        # legacy REPL's ``_bottom_toolbar`` reading the same ctx fields.
        # NOTE: server-side advisor attribution (Anthropic API's
        # ``usage.iterations[]`` with ``type="advisor_message"``) is NOT
        # currently surfaced — the Python SDK 0.88.0 we depend on only
        # defines ``"message"`` and ``"compaction"`` discriminators
        # (see anthropic/types/beta/beta_message_iteration_usage.py).
        # Add a separate accumulator + parser when the SDK gains the
        # advisor discriminator OR when we move to direct HTTP parsing.
        try:
            adv_in = int(getattr(self._tool_context, "advisor_input_tokens", 0) or 0)
            adv_out = int(getattr(self._tool_context, "advisor_output_tokens", 0) or 0)
            if adv_in or adv_out:
                self._state.usage["advisor_input_tokens"] = adv_in
                self._state.usage["advisor_output_tokens"] = adv_out
        except Exception:  # nosec B110
            pass
        self._post(
            AgentRunFinished(
                response_text=result.response_text,
                num_turns=result.num_turns,
                usage=result.usage,
            )
        )
        self._finish()

    def _finish(self) -> None:
        self._state.set_thinking(False)
        self._state.clear_streaming_text()
        with self._busy_lock:
            self._busy = False
            self._set_in_agent_loop(False)
            # Replace the per-run controller on the shared tool context
            # with a fresh one so the next ``submit()`` starts from a
            # clean state. Leaving an aborted controller in place would
            # cause the next prompt's first tool dispatch to see
            # ``signal.aborted == True`` and short-circuit before the
            # user has even pressed ESC.
            #
            # The dataclass field is non-optional, so we can't simply
            # clear to ``None`` — we install an untripped controller
            # that mirrors the dataclass default.
            #
            # Safety note: ``_finish`` runs on the worker thread *after*
            # ``run_agent_loop`` has returned or raised, so no in-flight
            # tool can be reading ``context.abort_controller`` from this
            # thread at the moment we replace it. Detached background
            # processes (e.g. ``spawn_background_bash``) capture their
            # own controller reference at spawn time rather than re-
            # reading the context field, so replacing here doesn't
            # orphan them either.
            self._tool_context.abort_controller = AbortController()
        # Signal the UI to drain any queued prompts.  The check is a
        # best-effort filter — the UI handler re-verifies on the UI
        # thread, so a spurious post (e.g. the queue was cleared by
        # ESC between the check and the handler) is a safe no-op.
        if self._state.queued_prompts:
            from .messages import QueuedPromptReady

            self._post(QueuedPromptReady())

    # ---- advisor rendering ----
    def _emit_advisor_events(self) -> None:
        """Scan the conversation for advisor blocks and post UI events for new ones.

        Iterates the assembled assistant messages (which now persist
        advisor blocks via ``ChatResponse.raw_content_blocks`` — see
        ``src/providers/anthropic_provider.py:_build_chat_response``)
        and posts one ``AdvisorEventMessage(kind="start")`` and one
        ``AdvisorEventMessage(kind="result")`` per advisor pair. Both
        emit in order so the transcript can mount the row in its
        running state before flipping it to done.

        Starts iteration at ``self._last_scanned_msg_index`` and
        advances the cursor at the end. This avoids the O(N×B) per-
        turn cost that grows over a long session — we only scan
        messages that landed since the last scan, which on a healthy
        agentic loop is just the latest assistant turn.
        """
        from src.utils.advisor import (
            extract_advisor_error_code,
            extract_advisor_result_text,
        )

        messages = getattr(self._session.conversation, "messages", None)
        if not messages:
            return
        # Defend against a wipe (e.g. /clear): if the conversation
        # shrank since last scan, the index is stale. Start over.
        start_idx = self._last_scanned_msg_index
        if start_idx > len(messages):
            start_idx = 0
        # Read the active advisor model from settings; ``server_tool_use``
        # blocks don't include it (it's parameterized on the schema, not
        # echoed back), but the user-facing label is much friendlier
        # with the model name attached.
        try:
            from src.settings.settings import get_settings

            advisor_model = (get_settings().advisor_model or "") or None
        except Exception:
            advisor_model = None
        # Client-side advisor results land on the NEXT (user-role)
        # message after the assistant's tool_use, since the dispatcher
        # routes the call through the tool registry. Build a lookup
        # of ``tool_use_id → (text, is_error)`` from user messages so
        # the inner loop can pair without an O(N²) walk.
        client_side_results: dict[str, tuple[str, bool]] = {}
        for msg in messages[start_idx:]:
            mcontent = getattr(msg, "content", None)
            if not isinstance(mcontent, list):
                continue
            for blk in mcontent:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") != "tool_result":
                    continue
                use_id = blk.get("tool_use_id")
                if not isinstance(use_id, str) or not use_id:
                    continue
                rc = blk.get("content")
                # tool_result.content can be a string or a list of
                # content blocks (multimodal tools). The advisor only
                # ever emits string content (advisor's reply text).
                if isinstance(rc, str):
                    text = rc
                elif isinstance(rc, list):
                    parts: list[str] = []
                    for b in rc:
                        if isinstance(b, dict) and isinstance(b.get("text"), str):
                            parts.append(b["text"])
                    text = "\n".join(parts)
                else:
                    text = str(rc) if rc is not None else ""
                client_side_results[use_id] = (text, bool(blk.get("is_error")))

        for msg in messages[start_idx:]:
            content = getattr(msg, "content", None)
            if not isinstance(content, list):
                continue
            # Pair-finding pass: collect (use_id → use_block_index) and
            # (use_id → result_content) within this assistant message.
            for i, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                bname = block.get("name")
                bid = block.get("id") or block.get("tool_use_id")
                if not isinstance(bid, str) or not bid:
                    continue
                if bid in self._emitted_advisor_ids:
                    continue
                # Client-side advisor: regular tool_use(name="advisor")
                # on an assistant message; the result is on a later
                # user-role message. We've already indexed those above.
                if btype == "tool_use" and bname == "advisor":
                    self._post(
                        AdvisorEventMessage(
                            kind="start",
                            tool_use_id=bid,
                            advisor_model=advisor_model,
                        )
                    )
                    pair = client_side_results.get(bid)
                    if pair is None:
                        # The dispatcher hasn't produced a result yet
                        # (turn in flight, or interrupted). Synthesize
                        # the interrupted event so the row doesn't spin
                        # forever. If the result lands later, the
                        # ``_emitted_advisor_ids`` guard prevents
                        # double-rendering.
                        self._post(
                            AdvisorEventMessage(
                                kind="result",
                                tool_use_id=bid,
                                advisor_model=advisor_model,
                                error_code="interrupted",
                            )
                        )
                    else:
                        result_text, is_err = pair
                        self._post(
                            AdvisorEventMessage(
                                kind="result",
                                tool_use_id=bid,
                                advisor_model=advisor_model,
                                text=None if is_err else result_text,
                                error_code=(
                                    result_text[:120] if is_err and result_text else ("error" if is_err else None)
                                ),
                            )
                        )
                    self._emitted_advisor_ids.add(bid)
                    continue
                if btype == "server_tool_use" and bname == "advisor":
                    self._post(
                        AdvisorEventMessage(
                            kind="start",
                            tool_use_id=bid,
                            advisor_model=advisor_model,
                        )
                    )
                    # Look for a matching advisor_tool_result anywhere
                    # later in the same assistant message.
                    result_block = None
                    for later in content[i + 1 :]:
                        if (
                            isinstance(later, dict)
                            and later.get("type") == "advisor_tool_result"
                            and later.get("tool_use_id") == bid
                        ):
                            result_block = later
                            break
                    if result_block is None:
                        # No result on this assistant turn — the use was
                        # interrupted. Synthesize an error event so the
                        # UI doesn't leave the row spinning forever.
                        self._post(
                            AdvisorEventMessage(
                                kind="result",
                                tool_use_id=bid,
                                advisor_model=advisor_model,
                                error_code="interrupted",
                            )
                        )
                        self._emitted_advisor_ids.add(bid)
                        continue
                    rcontent = result_block.get("content")
                    text = extract_advisor_result_text(rcontent)
                    err_code = extract_advisor_error_code(rcontent)
                    self._post(
                        AdvisorEventMessage(
                            kind="result",
                            tool_use_id=bid,
                            advisor_model=advisor_model,
                            text=text,
                            error_code=err_code,
                        )
                    )
                    self._emitted_advisor_ids.add(bid)
        # Cursor forward so the next scan only inspects new messages.
        self._last_scanned_msg_index = len(messages)

    # ---- permission bridge ----
    def _permission_handler(
        self,
        tool_name: str,
        message: str,
        suggestion: str | None,
    ) -> tuple[bool, bool]:
        """Called from the worker thread whenever the tool dispatcher
        wants user approval.

        Posts a :class:`PermissionRequested` to the UI, which pushes a
        modal; blocks the worker until the modal resolves via
        :class:`PermissionResolved`.
        """

        done = threading.Event()
        outcome: dict[str, bool] = {"allowed": False, "enable": False}

        def _decide(allowed: bool, enable: bool) -> None:
            outcome["allowed"] = allowed
            outcome["enable"] = enable
            done.set()

        pending = self._state.enqueue_permission(
            tool_name=tool_name,
            message=message,
            suggestion=suggestion,
            tool_input=None,
            decide=_decide,
        )
        self._post(
            PermissionRequested(
                request_id=pending.request_id,
                tool_name=tool_name,
                message=message,
                suggestion=suggestion,
                tool_input=None,
            )
        )
        # F-108 P108-A: bound the wait so a stuck modal cannot hang the
        # worker thread indefinitely (risk #2). After the timeout we
        # fall back to the safest default — deny without remembering —
        # which mirrors the legacy ESC behaviour.
        timeout_s = _resolve_permission_timeout_s()
        if timeout_s > 0:
            done.wait(timeout=timeout_s)
            if not done.is_set():
                # F-108 P108-A: UI never responded within the budget;
                # auto-deny. The pending modal is still in state — the
                # eventual UI ``decide()`` call (or Escape) will call
                # ``resolve_permission`` and drop it; that is idempotent.
                outcome["allowed"] = False
                outcome["enable"] = False
        else:
            done.wait()
        # Remove the entry from the state queue; the modal already
        # dismissed itself and emitted ``PermissionResolved``.
        self._state.resolve_permission(pending.request_id)
        return outcome["allowed"], outcome["enable"]

    # ---- ask_user bridge ----
    def _ask_user_handler(self, questions: list[dict[str, Any]]) -> dict[str, str]:
        """Called from the worker thread when ``AskUserQuestion`` runs.

        Posts :class:`AskUserQuestionRequested` to the UI, which pushes
        :class:`~src.tui.screens.ask_user_question.AskUserQuestionModal`;
        blocks the worker until the modal resolves via
        :class:`AskUserQuestionResolved`. Mirrors
        :meth:`_permission_handler` so the failure modes (UI never
        responds → worker thread stuck) are identical to those of the
        permission flow.

        Returns a dict ``{question_text: chosen_label_or_free_text}``.
        On cancellation (Esc / Ctrl+C) returns an empty dict so the
        agent loop can proceed without a real answer.
        """

        done = threading.Event()
        outcome: dict[str, dict[str, str] | None] = {"answers": None}

        def _decide(answers: dict[str, str] | None) -> None:
            outcome["answers"] = answers or {}
            done.set()

        pending = self._state.enqueue_ask_user(
            questions=list(questions),
            decide=_decide,
        )
        self._post(
            AskUserQuestionRequested(
                request_id=pending.request_id,
                questions=list(questions),
            )
        )
        # F-108 P108-A: bound the wait so a stuck AskUserQuestion modal
        # cannot hang the worker thread indefinitely (risk #3). After
        # the timeout we return ``{}`` (parity with the Esc-cancel path)
        # so the agent loop can recover without a real answer.
        timeout_s = _resolve_permission_timeout_s()
        if timeout_s > 0:
            done.wait(timeout=timeout_s)
            if not done.is_set():
                # F-108 P108-A: UI never responded; return empty answers.
                # Same idempotency reasoning as ``_permission_handler`` —
                # the modal's eventual decide() will still drain state.
                outcome["answers"] = {}
        else:
            done.wait()
        # Remove the entry from the state queue; the modal already
        # dismissed itself and emitted ``AskUserQuestionResolved``.
        self._state.resolve_ask_user(pending.request_id)
        return outcome["answers"] or {}


def _safe_copy(value: Any) -> Any:
    """Best-effort clone so the UI thread doesn't mutate tool-thread memory."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _safe_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_copy(v) for v in value]
    return value


__all__ = ["AgentBridge"]
