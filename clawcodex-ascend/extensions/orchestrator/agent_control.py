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
"""Control-socket and operator-interaction helpers for AgentRunner."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from extensions.api.query import ToolCallEvent

    from .agent_session import AgentSession

logger = logging.getLogger(__name__)


class AgentControlMixin:
    """Provide approval, control-socket, pause and operator-hint behavior."""

    def _resolve_protocols(self) -> None:
        """Lazy-initialise Protocol defaults on first use."""
        if getattr(self, "_protocols_active", False):
            return
        from extensions.orchestrator_runtime.adapters import (
            build_default_agent_runtime,
            build_default_bootstrap_state,
            build_default_coordinator_provider,
            build_default_session_storage,
        )

        self._agent_runtime = build_default_agent_runtime()
        self._session_storage = build_default_session_storage()
        self._coordinator = build_default_coordinator_provider()
        self._bootstrap_state = build_default_bootstrap_state()
        self._protocols_active = True

    def _handle_tool_call(
        self,
        event: "ToolCallEvent",
        session_context: dict[str, Any],
    ) -> "ToolCallEvent":
        """Intercept a tool call and mirror its approval decision."""
        from .approval_policy import (
            ToolCallEvent as PolicyToolCallEvent,
            apply_approval_decision,
            read_approval_decision,
        )

        policy_event = PolicyToolCallEvent(
            tool_name=event.tool_name,
            params=event.params,
            tool_use_id=event.tool_use_id,
        )
        self._approval_policy.evaluate(policy_event, session_context)
        apply_approval_decision(event, read_approval_decision(policy_event))
        return event

    @staticmethod
    def _event_to_broadcast_dict(event: Any) -> dict:
        """Return a JSON-safe representation of a query event."""
        from extensions.api.query import (
            PhaseComplete,
            SessionComplete,
            TextDelta,
            ToolCallEvent,
            ToolResultEvent,
            TurnComplete,
        )

        if isinstance(event, TextDelta):
            return {"content": str(getattr(event, "content", ""))}
        if isinstance(event, ToolCallEvent):
            from .approval_policy import read_approval_decision

            decision = read_approval_decision(event)
            return {
                "tool_name": str(getattr(event, "tool_name", "")),
                "tool_use_id": getattr(event, "tool_use_id", None),
                "params": dict(getattr(event, "params", {}) or {}),
                "approved": decision.is_approved,
            }
        if isinstance(event, ToolResultEvent):
            return {
                "tool_name": str(getattr(event, "tool_name", "")),
                "tool_use_id": getattr(event, "tool_use_id", None),
                "result": dict(getattr(event, "result", {}) or {}),
            }
        if isinstance(event, PhaseComplete):
            return {
                "phase": getattr(event, "phase", 0),
                "turn_count": getattr(event, "turn_count", 0),
            }
        if isinstance(event, TurnComplete):
            return {"turn": getattr(event, "turn", 0)}
        if isinstance(event, SessionComplete):
            return {"reason": str(getattr(event, "reason", ""))}
        return {}

    @staticmethod
    async def _broadcast_to_socket(session: "AgentSession", event: Any) -> None:
        """Broadcast an event without allowing socket failures to abort a run."""
        if session.control_socket is None:
            return
        try:
            await session.control_socket.send_event(
                {
                    "type": event.__class__.__name__,
                    "data": AgentControlMixin._event_to_broadcast_dict(event),
                }
            )
        except Exception:
            logger.warning(
                "Control socket broadcast failed issue_id=%s event=%s",
                getattr(getattr(session, "issue", None), "id", "unknown"),
                event.__class__.__name__,
                exc_info=True,
            )

    @staticmethod
    def _apply_pause_session(session: "AgentSession", reason: str = "operator_interrupt") -> None:
        """Pause a session and close both pause gates."""
        session.paused = True
        session.pause_reason = reason
        if session.pause_resume_event is not None:
            session.pause_resume_event.clear()
        if session._pause_gate is not None:
            session._pause_gate.clear()

    @staticmethod
    def _apply_resume_session(session: "AgentSession", prompt_override: str | None = None) -> None:
        """Resume a session and reopen both pause gates."""
        if prompt_override:
            session.prompt_override = prompt_override
        session.paused = False
        if session.pause_resume_event is not None:
            session.pause_resume_event.set()
        if session._pause_gate is not None:
            session._pause_gate.set()

    @staticmethod
    def _drain_control_commands(session: "AgentSession") -> bool:
        """Drain pending socket commands; return whether the run must stop."""
        if session.control_socket is None:
            return False
        stop_requested = False
        try:
            queue = session.control_socket._command_queue
            while True:
                try:
                    cmd = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if cmd.cmd == "pause":
                    AgentControlMixin._apply_pause_session(session)
                    AgentControlMixin._notify_pause_change(session, True, session.pause_reason)
                elif cmd.cmd == "resume":
                    AgentControlMixin._apply_resume_session(session, cmd.payload)
                    AgentControlMixin._notify_pause_change(session, False, "")
                elif cmd.cmd == "stop":
                    session.status = "failed"
                    session.session_end_reason = "operator_stop"
                    session.session_end_summary = "operator sent stop via control socket"
                    if session.pause_resume_event is not None:
                        session.pause_resume_event.set()
                    stop_requested = True
                elif cmd.cmd == "takeover":
                    session.status = "failed"
                    session.session_end_reason = "operator_takeover"
                    session.session_end_summary = "operator requested takeover via control socket"
                    stop_requested = True
                elif cmd.cmd == "inject":
                    AgentControlMixin._deliver_injected_message(session, cmd.payload)
                elif cmd.cmd == "detach":
                    logger.info("control_socket detach received run_id=%s", session.run_id)
                elif cmd.cmd == "flush_transcript":
                    AgentControlMixin._flush_transcript(session)
        except Exception:
            logger.exception("control_socket drain failed")
        return stop_requested

    @staticmethod
    def _notify_pause_change(session: "AgentSession", paused: bool, reason: str) -> None:
        callback = session._on_pause_state_change
        if callback is None:
            return
        try:
            callback(session.issue.id if session.issue else "", paused, reason)
        except Exception:
            logger.exception("_on_pause_state_change failed")

    @staticmethod
    def _deliver_injected_message(session: "AgentSession", payload: str) -> None:
        """Deliver an injected operator message to transcript and runtime."""
        if session._transcript_storage is not None:
            try:
                from extensions.orchestrator_runtime.utils.messages_impl import (
                    TextBlock,
                    create_user_message,
                )

                session._transcript_storage.write_message(
                    create_user_message(
                        content=[TextBlock(text=payload)],
                        origin="inject",
                    )
                )
                session._transcript_storage.flush()
            except Exception:
                logger.exception("Failed to write inject to transcript run_id=%s", session.run_id)

        queued = False
        if session._runtime_tasks is not None and session.run_id:
            try:
                from src.tasks.local_agent import queue_pending_message

                queued = queue_pending_message(
                    session.run_id,
                    payload,
                    session._runtime_tasks,
                )
            except Exception:
                queued = False
        if not queued:
            AgentControlMixin._write_operator_hint(session, payload)
        AgentControlMixin._emit_inject_delivered(session, payload)

    @staticmethod
    def _emit_inject_delivered(session: "AgentSession", payload: str) -> None:
        if session.control_socket is None:
            return
        try:
            coro = session.control_socket.send_event(
                {
                    "type": "InjectDelivered",
                    "data": {"hint_snippet": payload[:80] if payload else ""},
                }
            )
            try:
                asyncio.get_running_loop().create_task(coro)
            except RuntimeError:
                asyncio.run(coro)
        except Exception:
            logger.exception("Failed to emit InjectDelivered")

    @staticmethod
    def _flush_transcript(session: "AgentSession") -> None:
        if session._transcript_storage is None:
            return
        try:
            session._transcript_storage.flush()
            logger.info("Transcript flushed on request run_id=%s", session.run_id)
        except Exception:
            logger.exception("Failed to flush transcript run_id=%s", session.run_id)

    @staticmethod
    def _make_control_drain_fn(
        session: "AgentSession",
    ) -> Callable[[], str | None]:
        """Build the near-real-time control callback used by QueryRunner."""

        def _drain() -> str | None:
            if AgentControlMixin._drain_control_commands(session):
                return "stop"
            return None

        return _drain

    @staticmethod
    def _make_pause_wait_fn(
        session: "AgentSession",
    ) -> Callable[[], Awaitable[None]]:
        """Build a drain-and-wait callback that avoids resume deadlocks."""

        async def _pause_wait() -> None:
            if not session.paused or session.pause_resume_event is None:
                return
            if session.control_socket is not None:
                try:
                    await session.control_socket.send_event(
                        {
                            "type": "Paused",
                            "data": {
                                "turn": session.turn_count,
                                "tool_name": session.last_tool_name,
                                "issue_id": session.issue.id,
                                "run_id": session.run_id,
                            },
                        }
                    )
                except Exception:
                    pass  # nosec B110 - confirmation is best effort

            while session.paused:
                AgentControlMixin._drain_control_commands(session)
                if not session.paused:
                    break
                if session.status == "failed":
                    session.paused = False
                    break
                try:
                    await asyncio.wait_for(session.pause_resume_event.wait(), timeout=0.06)
                except asyncio.TimeoutError:
                    continue

            if session.control_socket is not None:
                try:
                    await session.control_socket.send_event(
                        {
                            "type": "Resumed",
                            "data": {
                                "turn": session.turn_count,
                                "issue_id": session.issue.id,
                                "run_id": session.run_id,
                            },
                        }
                    )
                except Exception:
                    pass  # nosec B110 - confirmation is best effort

        return _pause_wait

    @staticmethod
    def _write_operator_hint(session: "AgentSession", hint: str) -> None:
        """Append a deduplicated operator hint to the workspace."""
        if not hint or not session.run_id:
            return
        try:
            workspace_path = getattr(session.workspace, "path", None)
            if not workspace_path:
                return
            hints_file = Path(str(workspace_path)) / ".operator_hints.md"
            next_num = 1
            if hints_file.exists():
                content = hints_file.read_text(encoding="utf-8")
                if hint.strip() in content:
                    return
                next_num = content.count("--- Operator Hint #") + 1
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            header = f"--- Operator Hint #{next_num} (injected at {timestamp}) ---\n"
            with open(hints_file, "a", encoding="utf-8") as stream:
                stream.write(header)
                stream.write(hint + "\n")
                stream.write("-" * 50 + "\n")
        except Exception:
            logger.exception("Failed to write operator hint run_id=%s", session.run_id)


__all__ = ["AgentControlMixin"]
