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
"""Audit, transcript, sink and rate-limit helpers for AgentRunner."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from extensions.api.query import ToolCallEvent, ToolResultEvent

    from .agent_session import AgentSession

logger = logging.getLogger(__name__)

_TOOL_EVENT_LOG_ROTATE_BYTES = 50 * 1024 * 1024


class AgentEventMixin:
    """Provide event persistence, transcript flushing and 429 handling."""

    def _append_tool_event_log(
        self,
        event: "ToolCallEvent",
        session_context: dict[str, Any],
    ) -> None:
        """Persist one approval decision to the run's NDJSON audit log."""
        audit_log = session_context.get("audit_log", "full")
        if audit_log == "none":
            return
        if audit_log == "minimal" and event._approved is not False:
            return
        try:
            from .tool_event_log import ToolEventLog

            run_id = session_context.get("run_id") or "unknown"
            base_dir = self._event_log_dir(session_context.get("workspace_path"))
            try:
                base_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.exception(
                    "tool-event log mkdir failed run_id=%s path=%s",
                    run_id,
                    base_dir,
                )
                return

            log_path = base_dir / f"{run_id}.events.ndjson"
            self._rotate_event_log(log_path)
            row = ToolEventLog(
                tool=event.tool_name,
                params=event.params,
                approved=event._approved,
                deny_reason=event._deny_reason,
                permission_mode=session_context.get("permission_mode", "unknown"),
                turn=session_context.get("turn", 0),
                session_run_id=run_id,
                tool_use_id=getattr(event, "tool_use_id", None),
            )
            try:
                with open(log_path, "a", encoding="utf-8") as stream:
                    stream.write(row.to_json() + "\n")
            except Exception:
                logger.exception(
                    "tool-event log append failed run_id=%s path=%s",
                    run_id,
                    log_path,
                )
        except Exception:
            logger.exception("tool-event log unexpected failure")

    @staticmethod
    def _event_log_dir(workspace_path: Any) -> Path:
        if workspace_path:
            return Path(workspace_path) / ".reports"
        return Path.home() / ".clawcodex" / "tool-events"

    @staticmethod
    def _rotate_event_log(log_path: Path) -> None:
        try:
            if not log_path.exists():
                return
            if log_path.stat().st_size < _TOOL_EVENT_LOG_ROTATE_BYTES:
                return
            rotated = log_path.with_name(log_path.name + ".1")
            try:
                rotated.unlink(missing_ok=True)
            except Exception:
                logger.debug(
                    "tool-event rotated log cleanup failed path=%s",
                    rotated,
                    exc_info=True,
                )
            log_path.replace(rotated)
        except Exception:
            logger.exception("tool-event log rotate failed path=%s", log_path)

    def _append_agent_spawn_result_log(
        self,
        event: "ToolResultEvent",
        session_context: dict[str, Any],
    ) -> None:
        """Persist the spawned agent id once the Agent tool returns."""
        try:
            result = getattr(event, "result", None) or {}
            output = result.get("output")
            agent_id = output.get("agent_id") if isinstance(output, dict) else None
            if not agent_id:
                return
            from .tool_event_log import ToolEventLog

            run_id = session_context.get("run_id") or "unknown"
            base_dir = self._event_log_dir(session_context.get("workspace_path"))
            base_dir.mkdir(parents=True, exist_ok=True)
            row = ToolEventLog(
                tool="Agent",
                params={"description": output.get("description") or ""},
                approved=not result.get("is_error", False),
                deny_reason=None,
                permission_mode=session_context.get("permission_mode", "unknown"),
                turn=session_context.get("turn", 0),
                session_run_id=run_id,
                tool_use_id=getattr(event, "tool_use_id", None),
                kind="agent_result",
                agent_id=str(agent_id),
            )
            log_path = base_dir / f"{run_id}.events.ndjson"
            with open(log_path, "a", encoding="utf-8") as stream:
                stream.write(row.to_json() + "\n")
        except Exception:
            logger.exception("agent spawn result log failed")

    def _flush_turn_transcript(self, session: "AgentSession") -> None:
        """Write one assistant/tool-result pair and reset per-turn buffers."""
        if session._transcript_storage is None:
            return
        from extensions.orchestrator_runtime.utils.messages_impl import (
            TextBlock,
            ToolResultBlock,
            create_assistant_message,
            create_user_message,
        )

        storage = session._transcript_storage
        blocks: list[Any] = []
        if session._transcript_asst_text:
            blocks.append(TextBlock(text=session._transcript_asst_text))
        blocks.extend(session._transcript_tool_uses)
        if blocks:
            storage.write_message(
                create_assistant_message(
                    content=blocks,
                    model=self.agent_config.model,
                )
            )

        if session._transcript_tool_uses:
            result_blocks: list[Any] = []
            for tool_use in session._transcript_tool_uses:
                pending = session._transcript_pending_results.get(tool_use.id)
                if pending is None:
                    logger.warning(
                        "Transcript tool result missing tool_use_id=%s",
                        tool_use.id,
                    )
                    pending = ToolResultBlock(
                        tool_use_id=tool_use.id,
                        content="[Tool result missing — internal error]",
                        is_error=True,
                    )
                result_blocks.append(pending)
            storage.write_message(create_user_message(content=result_blocks, origin="tool_result"))

        session._transcript_asst_text = ""
        session._transcript_tool_uses = []
        session._transcript_pending_results = {}
        session._transcript_result_order = []

    def _is_429_response(self, turn_output: str) -> bool:
        """Detect retryable provider rate limits while excluding quota errors."""
        if not turn_output:
            return False
        low = turn_output.lower()
        quota_indicators = (
            "exceeded your current quota",
            "limit: 0",
            "token plan",
            "quota",
        )
        if any(item in low for item in quota_indicators):
            return False
        return (
            "error code: 429" in low
            or "rate_limit_error" in low
            or '"type": "rate_limit_error"' in low
            or "rate limit" in low
        )

    def _dispatch_sink(
        self,
        sink: Any,
        method: str,
        event: Any,
        session: "AgentSession",
    ) -> None:
        """Dispatch to a progress sink without letting it break the run."""
        if sink is None:
            return
        try:
            getattr(sink, method)(event, session)
        except Exception as exc:  # noqa: BLE001
            logger.warning("progress_sink.%s dispatch failed: %s", method, exc)

    def _compute_rate_limit_backoff(self, session: "AgentSession") -> float:
        """Compute capped exponential backoff plus up to ten percent jitter."""
        base_ms = self.agent_config.rate_limit_base_delay_ms
        max_ms = self.agent_config.rate_limit_max_backoff_ms
        factor = self.agent_config.rate_limit_exponential_factor
        count = max(1, session.consecutive_429_count)
        delay_ms = min(base_ms * (factor ** (count - 1)), max_ms)
        delay_s = delay_ms / 1000.0
        # Retry desynchronisation is non-security-sensitive.
        jitter = random.uniform(0, 0.1 * delay_s) if delay_s > 0 else 0.0  # nosec B311
        return delay_s + jitter

    async def _handle_rate_limit(
        self,
        session: "AgentSession",
        turn_output: str,
        turn_number: int,
        status_dashboard: Any | None,
    ) -> str:
        """Back off once, or open the circuit after the configured limit."""
        del turn_output
        session.consecutive_429_count += 1
        max_retries = self.agent_config.rate_limit_max_retries
        if session.consecutive_429_count > max_retries:
            session.status = "rate_limit_circuit_open"
            logger.error(
                "Rate limit circuit breaker open issue_id=%s consecutive=%d max=%d",
                session.issue.id,
                session.consecutive_429_count,
                max_retries,
            )
            return session.status

        delay_s = self._compute_rate_limit_backoff(session)
        session.total_429_backoff_seconds += delay_s
        notice = (
            f"\n[rate-limit] 429 detected "
            f"(attempt {session.consecutive_429_count}/{max_retries}); "
            f"sleeping {delay_s:.0f}s before retry\n"
        )
        session.output_text += notice
        from extensions.api.query import TextDelta

        if status_dashboard is not None:
            try:
                status_dashboard.on_event(TextDelta(content=notice), session)
            except Exception:
                pass  # nosec B110 - dashboard delivery is best effort
        session.rate_limit_pending_turn = turn_number
        await self._sleep(delay_s)
        return "running"


__all__ = ["AgentEventMixin"]
