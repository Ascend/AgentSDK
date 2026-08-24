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
"""Run one issue through the ClawCodex query engine."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .agent_completion import AgentCompletionMixin  # pylint: disable=import-error
from .agent_control import AgentControlMixin  # pylint: disable=import-error
from .agent_events import AgentEventMixin  # pylint: disable=import-error
from .agent_lifecycle import AgentLifecycleMixin  # pylint: disable=import-error
from .agent_session import AgentSession, RetryItem  # pylint: disable=import-error
from .agent_stream import AgentStreamMixin  # pylint: disable=import-error
from .agent_turn import AgentTurnMixin, TurnState  # pylint: disable=import-error

if TYPE_CHECKING:
    from .config.schema import AgentConfig, SandboxConfig, WorkflowConfig, WorkspaceConfig

logger = logging.getLogger(__name__)


class AgentRunner(
    AgentControlMixin,
    AgentEventMixin,
    AgentLifecycleMixin,
    AgentTurnMixin,
    AgentStreamMixin,
    AgentCompletionMixin,
):
    """Execute one issue with isolated coordinator and multi-turn state."""

    def __init__(
        self,
        agent_config: "AgentConfig",
        sandbox_config: "SandboxConfig",
        workspace_cfg: "WorkspaceConfig | None" = None,
        *,
        agent_runtime: Any | None = None,
        session_storage: Any | None = None,
        coordinator_provider: Any | None = None,
    ) -> None:
        from .approval_policy import get_approval_policy
        from .config.schema import WorkspaceConfig

        self.agent_config = agent_config
        self.sandbox_config = sandbox_config
        self.workspace_cfg = workspace_cfg or WorkspaceConfig()
        self.max_turns = agent_config.max_turns
        self.max_tools_per_turn = getattr(agent_config, "max_tools_per_turn", 50) or 50
        self._approval_policy = get_approval_policy(getattr(sandbox_config, "approval_policy", "never") or "never")
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        if any(item is not None for item in (agent_runtime, session_storage, coordinator_provider)):
            self._protocols_active = True
            self._agent_runtime = agent_runtime
            self._session_storage = session_storage
            self._coordinator = coordinator_provider
        else:
            self._protocols_active = False
            self._agent_runtime = None
            self._session_storage = None
            self._coordinator = None

    async def run(
        self,
        session: AgentSession,
        workflow: "WorkflowConfig",
        status_dashboard: Any | None = None,
        tracker: Any = None,
        comment_tracker: Any | None = None,
        clarification_resolver: Any | None = None,
        progress_reporter: Any | None = None,
        diagnostics_callback: Callable[[AgentSession], None] | None = None,
    ) -> None:
        """Execute one session with coordinator mode isolated per task."""
        self._resolve_protocols()
        explicit_mode = getattr(session, "coordinator_mode", None)
        coordinator_enabled = (
            bool(explicit_mode)
            if explicit_mode is not None
            else bool(getattr(self.agent_config, "coordinator_mode", False))
        )
        with self._coordinator.enter(coordinator_enabled):
            await self._run_impl(
                session,
                workflow,
                status_dashboard=status_dashboard,
                tracker=tracker,
                comment_tracker=comment_tracker,
                clarification_resolver=clarification_resolver,
                progress_reporter=progress_reporter,
                diagnostics_callback=diagnostics_callback,
            )

    async def _run_impl(
        self,
        session: AgentSession,
        workflow: "WorkflowConfig",
        status_dashboard: Any | None = None,
        tracker: Any = None,
        comment_tracker: Any | None = None,
        clarification_resolver: Any | None = None,
        progress_reporter: Any | None = None,
        diagnostics_callback: Callable[[AgentSession], None] | None = None,
    ) -> None:
        """Run QueryRunner turns until a terminal decision or budget limit."""
        run_state, session_context = await self._initialize_run(
            session,
            workflow,
            comment_tracker=comment_tracker,
            progress_reporter=progress_reporter,
            diagnostics_callback=diagnostics_callback,
        )
        try:
            while run_state.turn_number < self.max_turns:
                prompt = self._build_turn_prompt(
                    session,
                    run_state.turn_number,
                    clarification_resolver,
                )
                await self._ensure_turn_services(session, prompt)
                self._append_turn_start_debug(session, run_state, prompt)
                query_runner = self._build_query_runner(session, prompt)
                turn_state = TurnState()
                stream_iter = query_runner.stream()
                retry_turn = False
                natural_completion = False
                try:
                    while True:
                        try:
                            event = await anext(stream_iter)
                        except StopAsyncIteration:
                            break
                        self._record_stream_event(session, event, run_state)
                        action = await self._dispatch_stream_event(
                            session,
                            event,
                            run_state,
                            turn_state,
                            session_context,
                            tracker=tracker,
                            status_dashboard=status_dashboard,
                            stream_iter=stream_iter,
                        )
                        if action == "retry":
                            retry_turn = True
                            break
                        if action == "return":
                            return
                        if action == "break":
                            break
                        if action == "turn_complete":
                            natural_completion = True
                            break
                except Exception as exc:
                    recovery = await self._recover_rate_limit_exception(
                        exc,
                        session,
                        turn_state,
                        run_state,
                        status_dashboard,
                    )
                    if recovery == "raise":
                        raise
                    if recovery == "return":
                        return
                    retry_turn = True

                if retry_turn:
                    continue
                if session.session_end_reason in (
                    "operator_stop",
                    "operator_takeover",
                ):
                    await self._finalize_session(session, run_state, session.session_end_reason)
                    return
                if turn_state.cap_reached:
                    await self._finish_forced_turn_boundary(session, run_state)
                    continue
                if natural_completion:
                    continue
                if not turn_state.has_tool_calls and turn_state.output:
                    run_state.turn_number += 1
                    session.turn_count = run_state.turn_number

            await self._finish_budget_exhausted(session, run_state)
        finally:
            self._finalize_run_artifacts(session)

    async def _dispatch_stream_event(
        self,
        session: AgentSession,
        event: Any,
        run_state: Any,
        turn_state: TurnState,
        session_context: dict[str, Any],
        *,
        tracker: Any,
        status_dashboard: Any,
        stream_iter: Any,
    ) -> str:
        """Dispatch one query event to its focused handler."""
        from extensions.api.query import (
            SessionComplete,
            TextDelta,
            ToolCallEvent,
            ToolResultEvent,
        )

        if isinstance(event, TextDelta):
            await self._handle_text_delta_event(session, event, turn_state, run_state, status_dashboard)
            return "continue"
        if isinstance(event, ToolCallEvent):
            await self._handle_tool_call_event(
                session,
                event,
                turn_state,
                run_state,
                session_context,
                status_dashboard,
            )
            return "continue"
        if isinstance(event, ToolResultEvent):
            action = await self._handle_tool_result_event(
                session,
                event,
                turn_state,
                run_state,
                session_context,
                status_dashboard,
                stream_iter,
            )
            return "return" if action == "complete" else action
        if isinstance(event, SessionComplete):
            action = await self._handle_session_complete_event(
                session,
                event,
                run_state,
                turn_state,
                tracker=tracker,
                status_dashboard=status_dashboard,
            )
            if action == "continue":
                return "turn_complete"
            return action
        return "continue"

    async def _recover_rate_limit_exception(
        self,
        exc: Exception,
        session: AgentSession,
        turn_state: TurnState,
        run_state: Any,
        status_dashboard: Any,
    ) -> str:
        """Apply the normal 429 path when a provider raises a typed error."""
        try:
            from extensions.orchestrator_runtime.adapters.clawcodex_compat import (
                is_rate_limit_error,
            )

            if not is_rate_limit_error(exc):
                return "raise"
        except ImportError:
            return "raise"
        output = turn_state.output or f"Error code: 429 - {exc!s}"
        status = await self._handle_rate_limit(
            session,
            output,
            run_state.turn_number,
            status_dashboard,
        )
        if status == "rate_limit_circuit_open":
            await self._finalize_session(session, run_state, status)
            return "return"
        self._flush_turn_transcript(session)
        return "retry"

    @staticmethod
    def _append_turn_start_debug(session: AgentSession, run_state: Any, prompt: str) -> None:
        from .debug_log import append_debug_event

        append_debug_event(
            session.debug_log_path,
            "agent_runner.turn_start",
            issue_id=session.issue.id,
            run_id=session.run_id,
            turn=run_state.turn_number,
            prompt_len=len(prompt),
            output_len=len(session.output_text),
        )

    async def _finish_forced_turn_boundary(self, session: AgentSession, run_state: Any) -> None:
        """Advance after max_tools_per_turn consumed all pending results."""
        run_state.turn_number += 1
        session.turn_count = run_state.turn_number
        if session._transcript_storage is not None:
            self._flush_turn_transcript(session)
            session._transcript_storage.flush()

    async def _finish_budget_exhausted(self, session: AgentSession, run_state: Any) -> None:
        """Publish the synthetic terminal event for outer-loop exhaustion."""
        session.status = "max_turns_exceeded"
        session.session_end_reason = "budget_exhausted"
        session.session_end_summary = f"reached max_turns={self.max_turns} after {run_state.turn_number} turns"
        session.tool_count = run_state.tool_count
        await self._finalize_session(session, run_state, "budget_exhausted")

    def _finalize_run_artifacts(self, session: AgentSession) -> None:
        """Clear log context and write snapshots on every exit path."""
        try:
            from .logging_setup import clear_log_context

            clear_log_context()
        finally:
            session._save_json_snapshot()
            self._export_events_for_viz(session)


__all__ = ["AgentRunner", "AgentSession", "RetryItem"]
