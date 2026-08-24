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
"""Text, tool-call and tool-result stream handlers for AgentRunner."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .agent_session import AgentSession
    from .agent_turn import RunState, TurnState

logger = logging.getLogger(__name__)

MEGATURN_CHECK_EVERY_S = 60.0
MEGATURN_IDLE_STOP_S = 1800.0
MODIFYING_TOOL_NAMES = frozenset(
    {
        "Write",
        "Edit",
        "FileWrite",
        "FileWriteTool",
        "FileEdit",
        "FileEditTool",
        "WriteTool",
        "EditTool",
    }
)

StreamAction = Literal["continue", "break", "complete"]


class AgentStreamMixin:
    """Handle non-terminal events yielded by QueryRunner.stream()."""

    def _record_stream_event(
        self,
        session: "AgentSession",
        event: Any,
        run_state: "RunState",
    ) -> None:
        """Update last-event diagnostics and append the debug row."""
        from .debug_log import append_debug_event

        event_type = type(event).__name__
        session.last_agent_event_at = time.time()
        session.last_agent_event = event_type
        tool_name = getattr(event, "tool_name", None)
        if tool_name:
            session.last_tool_name = tool_name
        append_debug_event(
            session.debug_log_path,
            "agent_runner.event",
            issue_id=session.issue.id,
            run_id=session.run_id,
            type=event_type,
            tool=tool_name,
            turn=run_state.turn_number,
            tool_count=run_state.tool_count,
            output_len=len(session.output_text),
            reason=getattr(event, "reason", None),
        )

    async def _publish_stream_event(
        self,
        session: "AgentSession",
        event: Any,
        status_dashboard: Any = None,
    ) -> None:
        """Fan out one event to the dashboard, CLI queue and control socket."""
        if status_dashboard is not None:
            try:
                status_dashboard.on_event(event, session)
            except Exception:
                pass  # nosec B110 - optional dashboard is best effort
        if session.event_queue is not None:
            try:
                session.event_queue.put_nowait(event)
            except Exception:
                pass  # nosec B110 - optional CLI queue is best effort
        await self._broadcast_to_socket(session, event)

    async def _handle_text_delta_event(
        self,
        session: "AgentSession",
        event: Any,
        turn_state: "TurnState",
        run_state: "RunState",
        status_dashboard: Any = None,
    ) -> StreamAction:
        """Accumulate assistant text and publish it to live consumers."""
        session.output_text += event.content
        turn_state.output += event.content
        session._transcript_asst_text += event.content
        run_state.update_diagnostics(session)
        await self._publish_stream_event(session, event, status_dashboard)
        return "continue"

    async def _handle_tool_call_event(
        self,
        session: "AgentSession",
        event: Any,
        turn_state: "TurnState",
        run_state: "RunState",
        session_context: dict[str, Any],
        status_dashboard: Any = None,
    ) -> StreamAction:
        """Apply approval/audit, update progress and buffer a tool-use block."""
        turn_state.has_tool_calls = True
        turn_state.tool_count += 1
        turn_state.pending_tool_results += 1
        run_state.tool_count += 1
        if event.tool_name:
            turn_state.tool_names.append(event.tool_name)
        if event.tool_name in MODIFYING_TOOL_NAMES:
            session.has_made_progress = True
            turn_state.has_modifying_tool = True
        if turn_state.tool_count >= self.max_tools_per_turn:
            turn_state.cap_reached = True
            logger.info(
                "Turn %s reached max_tools_per_turn=%s for issue %s",
                run_state.turn_number,
                self.max_tools_per_turn,
                session.issue.id,
            )

        approved_event = self._handle_tool_call(event, session_context)
        session_context["turn"] = run_state.turn_number
        self._append_tool_event_log(approved_event, session_context)
        run_state.update_diagnostics(session)
        await self._publish_stream_event(session, approved_event, status_dashboard)
        self._buffer_tool_use(session, approved_event)
        return "continue"

    @staticmethod
    def _buffer_tool_use(session: "AgentSession", event: Any) -> None:
        if session._transcript_storage is None or not event.tool_use_id:
            return
        try:
            from extensions.orchestrator_runtime.utils.messages_impl import ToolUseBlock

            session._transcript_tool_uses.append(
                ToolUseBlock(
                    id=event.tool_use_id,
                    name=event.tool_name,
                    input=event.params,
                )
            )
        except Exception:
            logger.exception("Failed to buffer transcript tool_use run_id=%s", session.run_id)

    async def _handle_tool_result_event(
        self,
        session: "AgentSession",
        event: Any,
        turn_state: "TurnState",
        run_state: "RunState",
        session_context: dict[str, Any],
        status_dashboard: Any = None,
        stream_iter: Any = None,
    ) -> StreamAction:
        """Buffer a tool result and decide whether the stream should stop."""
        turn_state.pending_tool_results -= 1
        if event.tool_name == "Agent":
            self._append_agent_spawn_result_log(event, session_context)
        await self._publish_stream_event(session, event, status_dashboard)
        self._buffer_tool_result(session, event)
        run_state.update_diagnostics(session)

        if turn_state.cap_reached and turn_state.pending_tool_results <= 0:
            return "break"
        if turn_state.pending_tool_results <= 0 and self._drain_control_commands(session):
            return "break"
        await self._probe_megaturn_idle(session, turn_state)
        if turn_state.megaturn_stop and turn_state.pending_tool_results <= 0:
            await self._finish_megaturn_idle(session, run_state, stream_iter)
            return "complete"
        return "continue"

    def _buffer_tool_result(self, session: "AgentSession", event: Any) -> None:
        if session._transcript_storage is None or not event.tool_use_id:
            return
        try:
            from extensions.orchestrator_runtime.utils.messages_impl import ToolResultBlock

            output = event.result.get("output", "")
            session._transcript_pending_results[event.tool_use_id] = ToolResultBlock(
                tool_use_id=event.tool_use_id,
                content=output if isinstance(output, str) else str(output),
                is_error=event.result.get("is_error", False),
            )
            if event.tool_use_id not in session._transcript_result_order:
                session._transcript_result_order.append(event.tool_use_id)
            if len(session._transcript_result_order) >= len(session._transcript_tool_uses):
                self._flush_turn_transcript(session)
        except Exception:
            logger.exception("Failed to buffer transcript tool_result run_id=%s", session.run_id)

    async def _probe_megaturn_idle(self, session: "AgentSession", turn_state: "TurnState") -> None:
        """Check whether a productive mega-turn has stopped changing files."""
        now = time.monotonic()
        if now < turn_state.megaturn_next_check_at:
            return
        from .agent_session import (
            _is_orchestrator_internal_path,
            _megaturn_idle_stop_enabled,
        )

        if not _megaturn_idle_stop_enabled(session):
            return
        turn_state.megaturn_next_check_at = now + MEGATURN_CHECK_EVERY_S
        try:
            from extensions.orchestrator_runtime.adapters.clawcodex_compat import (
                get_file_status,
            )

            workspace_path = getattr(session.workspace, "path", None)
            if workspace_path is None:
                return
            entries = await asyncio.to_thread(get_file_status, str(workspace_path))
            user_entries = [
                entry for entry in entries if not _is_orchestrator_internal_path(getattr(entry, "path", str(entry)))
            ]
            signature = "|".join(sorted(str(getattr(entry, "path", entry)) for entry in user_entries))
            if signature != turn_state.megaturn_workspace_signature:
                turn_state.megaturn_workspace_signature = signature
                turn_state.megaturn_workspace_changed_at = now
            elif user_entries and (now - turn_state.megaturn_workspace_changed_at >= MEGATURN_IDLE_STOP_S):
                turn_state.megaturn_stop = True
        except Exception:
            pass  # nosec B110 - the idle probe must fail open

    async def _finish_megaturn_idle(
        self,
        session: "AgentSession",
        run_state: "RunState",
        stream_iter: Any,
    ) -> None:
        """Finish a mega-turn once changed files have remained idle."""
        from extensions.api.query import SessionComplete
        from .debug_log import append_debug_event

        session.has_made_progress = True
        session.status = "completed"
        session.session_end_reason = "megaturn_workspace_idle"
        session.session_end_summary = (
            "Workspace changes landed and stayed unchanged "
            f"for {int(MEGATURN_IDLE_STOP_S)}s inside a single turn; "
            "session ended early."
        )
        append_debug_event(
            session.debug_log_path,
            "agent_runner.megaturn_early_stop",
            issue_id=session.issue.id,
            run_id=session.run_id,
            turn=run_state.turn_number,
            tool_count=run_state.tool_count,
            idle_stop_s=MEGATURN_IDLE_STOP_S,
        )
        self._flush_turn_transcript(session)
        if stream_iter is not None:
            try:
                await stream_iter.aclose()
            except Exception:
                pass  # nosec B110 - stream may already be closed
        self._dispatch_sink(
            run_state.sink,
            "on_session_complete",
            SessionComplete(reason="megaturn_workspace_idle"),
            session,
        )


__all__ = ["AgentStreamMixin", "StreamAction"]
