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
"""SessionComplete state machine for the Orchestrator Agent Runtime."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .agent_session import AgentSession
    from .agent_turn import RunState, TurnState

logger = logging.getLogger(__name__)

MAX_READ_ONLY_TURNS = 4
NOOP_DETECTION_MAX_TURNS = 5
CompletionAction = Literal["continue", "retry", "return"]


class AgentCompletionMixin:
    """Turn SessionComplete events into continuation or terminal decisions."""

    async def _handle_session_complete_event(
        self,
        session: "AgentSession",
        event: Any,
        run_state: "RunState",
        turn_state: "TurnState",
        *,
        tracker: Any = None,
        status_dashboard: Any = None,
    ) -> CompletionAction:
        """Handle one natural turn boundary and return the outer-loop action."""
        if session.session_end_reason in ("operator_stop", "operator_takeover"):
            await self._finalize_session(session, run_state, session.session_end_reason)
            return "return"

        if self._is_429_response(turn_state.output):
            status = await self._handle_rate_limit(
                session,
                turn_state.output,
                run_state.turn_number,
                status_dashboard,
            )
            if status == "rate_limit_circuit_open":
                await self._finalize_session(session, run_state, status)
                return "return"
            self._flush_turn_transcript(session)
            return "retry"

        await self._finish_turn_boundary(session, event, run_state)
        if event.reason != "success":
            session.status = "failed"
            reason = str(event.reason)
            normalized_reason = reason if reason.startswith("exit_code=") else f"exit_code={reason}"
            session.session_end_reason = session.session_end_reason or normalized_reason
            session.session_end_summary = session.session_end_summary or f"QueryRunner ended with reason={event.reason}"
            await self._finalize_session(session, run_state, session.session_end_reason)
            return "return"

        session.consecutive_429_count = 0
        session.rate_limit_pending_turn = None
        if tracker is not None and session.issue.id:
            active, refreshed = await self._poll_tracker(session, tracker)
            if active and run_state.turn_number < self.max_turns:
                terminal_reason = await self._continuation_guard(session, run_state, turn_state)
                if terminal_reason is None:
                    return "continue"
                await self._finalize_session(session, run_state, terminal_reason)
                return "return"
            session.issue = refreshed or session.issue
        elif run_state.turn_number < self.max_turns:
            if self._complete_workflow_stage(session, run_state, turn_state):
                await self._finalize_session(
                    session,
                    run_state,
                    session.session_end_reason or "task_complete",
                )
                return "return"
            return "continue"

        if run_state.turn_number >= self.max_turns and session.status == "running":
            session.status = "max_turns_exceeded"
            session.session_end_reason = "budget_exhausted"
            session.session_end_summary = f"reached max_turns={self.max_turns} after {run_state.turn_number} turns"
        elif session.status == "running":
            await self._classify_success_without_active_issue(session, run_state)
        await self._finalize_session(
            session,
            run_state,
            session.session_end_reason or event.reason,
        )
        return "return"

    async def _finish_turn_boundary(
        self,
        session: "AgentSession",
        event: Any,
        run_state: "RunState",
    ) -> None:
        """Increment counters, flush transcript and publish phase/turn events."""
        from extensions.api.query import PhaseComplete, TurnComplete

        self._drain_control_commands(session)
        run_state.turn_number += 1
        session.turn_count = run_state.turn_number
        if session._transcript_storage is not None:
            try:
                self._flush_turn_transcript(session)
                session._transcript_storage.flush()
            except Exception:
                logger.exception("Failed to flush transcript run_id=%s", session.run_id)
        phase_event = PhaseComplete(phase=run_state.turn_number, turn_count=run_state.turn_number)
        turn_event = TurnComplete(turn=run_state.turn_number)
        await self._broadcast_to_socket(session, phase_event)
        await self._broadcast_to_socket(session, turn_event)
        self._dispatch_sink(run_state.sink, "on_phase_complete", phase_event, session)
        self._dispatch_sink(run_state.sink, "on_turn_complete", turn_event, session)
        run_state.update_diagnostics(session)

    async def _poll_tracker(self, session: "AgentSession", tracker: Any) -> tuple[bool, Any]:
        try:
            return await self._should_continue(session.issue, tracker, session)
        except Exception as exc:
            logger.warning(
                "Tracker poll failed for issue %s, assuming active: %s",
                session.issue.id,
                exc,
            )
            return True, session.issue

    async def _continuation_guard(
        self,
        session: "AgentSession",
        run_state: "RunState",
        turn_state: "TurnState",
    ) -> str | None:
        """Detect stagnation, read-only loops, repeated signatures and no-op work."""
        if not turn_state.has_tool_calls and not turn_state.output.strip():
            run_state.no_work_streak += 1
        else:
            run_state.no_work_streak = 0
        threshold = run_state.max_no_op_turns * (2 if session.has_made_progress else 1)
        if run_state.no_work_streak >= threshold:
            if run_state.tool_count == 0 and getattr(self.agent_config, "test_command", None):
                if await self._run_verification(session):
                    session.status = "completed"
                    session.session_end_reason = "already_completed"
                    session.session_end_summary = "work already implemented (verification passed)"
                    return "already_completed"
                session.status = "stagnation"
                session.session_end_reason = "llm_gave_up"
                session.session_end_summary = "LLM returned SessionComplete(success) after 0 tool calls"
                return "llm_gave_up"
            session.status = "stagnation"
            session.session_end_reason = "stagnation"
            session.session_end_summary = f"{run_state.no_work_streak} consecutive no-work turns"
            return "stagnation"

        workspace_dirty = await self._workspace_is_dirty(session)
        if (
            run_state.turn_number > 0
            and turn_state.has_tool_calls
            and not turn_state.has_modifying_tool
            and not workspace_dirty
        ):
            run_state.read_only_streak += 1
        else:
            run_state.read_only_streak = 0
        if run_state.read_only_streak >= MAX_READ_ONLY_TURNS:
            session.status = "read_only_loop"
            session.session_end_reason = "read_only_loop"
            session.session_end_summary = f"{run_state.read_only_streak} consecutive read-only turns"
            return "read_only_loop"

        signature = "|".join(sorted(turn_state.tool_names)) if turn_state.tool_names else "<empty>"
        run_state.tool_signature_history.append(signature)
        run_state.tool_signature_history[:] = run_state.tool_signature_history[-run_state.loop_window :]
        repeat_count = run_state.tool_signature_history.count(signature)
        if repeat_count >= run_state.loop_threshold:
            session.status = "loop_detected"
            session.session_end_reason = "loop_detected"
            session.session_end_summary = f"signature {signature!r} repeated {repeat_count} times"
            return "loop_detected"

        if workspace_dirty:
            session.has_made_progress = True
            run_state.consecutive_clean_turns = 0
        else:
            run_state.consecutive_clean_turns += 1
            if run_state.consecutive_clean_turns >= NOOP_DETECTION_MAX_TURNS:
                session.status = "completed"
                session.session_end_reason = "noop_completed"
                session.session_end_summary = f"{run_state.consecutive_clean_turns} consecutive clean turns"
                return "noop_completed"
        return None

    async def _workspace_is_dirty(self, session: "AgentSession") -> bool:
        try:
            from extensions.orchestrator_runtime.adapters.clawcodex_compat import (
                get_file_status,
            )

            return bool(get_file_status(str(session.workspace.path)))
        except Exception:
            return True

    def _complete_workflow_stage(
        self,
        session: "AgentSession",
        run_state: "RunState",
        turn_state: "TurnState",
    ) -> bool:
        """Classify tracker-less workflow stages such as Pipeline analysis."""
        if not turn_state.has_tool_calls and not turn_state.output.strip():
            run_state.no_work_streak += 1
        else:
            run_state.no_work_streak = 0
        if session.has_made_progress or run_state.tool_count > 0:
            session.status = "completed"
            session.session_end_reason = "task_complete"
            session.session_end_summary = (
                f"workflow stage completed after {run_state.turn_number} turns, {run_state.tool_count} tools"
            )
            return True
        if run_state.no_work_streak >= run_state.max_no_op_turns:
            session.status = "failed"
            session.session_end_reason = "rate_limited"
            session.session_end_summary = f"agent never started: {run_state.no_work_streak} empty turns"
            return True
        return False

    async def _classify_success_without_active_issue(self, session: "AgentSession", run_state: "RunState") -> None:
        if session.has_made_progress:
            completion = await self._workspace_completion_state(session)
            has_changes = completion is None or completion[0] or completion[2]
            if has_changes:
                session.status = "completed"
                session.session_end_reason = "task_complete"
                session.session_end_summary = "issue no longer active"
            else:
                session.status = "failed"
                session.session_end_reason = "no_changes_produced"
                session.session_end_summary = (
                    f"Agent called Write/Edit ({run_state.tool_count} tools) but workspace has no file changes"
                )
            return
        if await self._run_verification(session):
            session.status = "completed"
            session.session_end_reason = "already_completed"
            session.session_end_summary = "work already implemented (verification passed)"
        else:
            session.status = "failed"
            session.session_end_reason = "llm_gave_up"
            session.session_end_summary = f"LLM completed after {run_state.tool_count} read-only tool calls"

    async def _finalize_session(
        self,
        session: "AgentSession",
        run_state: "RunState",
        reason: str,
    ) -> None:
        """Publish the terminal event, flush transcript and close the socket."""
        from extensions.api.query import SessionComplete

        event = SessionComplete(reason=reason)
        await self._broadcast_to_socket(session, event)
        self._dispatch_sink(run_state.sink, "on_session_complete", event, session)
        if session._transcript_storage is not None:
            try:
                self._flush_turn_transcript(session)
                session._transcript_storage.flush()
            except Exception:
                logger.exception("Failed to final-flush transcript run_id=%s", session.run_id)
        if session.control_socket is not None:
            try:
                await session.control_socket.stop()
            except Exception:
                logger.exception("Failed to stop control_socket run_id=%s", session.run_id)
            session.control_socket = None


__all__ = ["AgentCompletionMixin", "CompletionAction"]
