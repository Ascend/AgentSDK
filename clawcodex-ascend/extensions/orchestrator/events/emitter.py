#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""OrchestratorEventEmitter — a ProgressSink + explicit emit().

Captures session-level terminal events via the ProgressSink protocol
(``on_session_complete``) and lets call sites emit explicit events for
blind spots the sink cannot see (clarification, git/PR, control). Every
sink dispatch is exception-isolated so an IM failure never propagates
into the orchestrator main flow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from .types import EventLevel, OrchestratorEvent, TERMINAL_REASON_LEVEL

if TYPE_CHECKING:
    from extensions.api.query import PhaseComplete, SessionComplete, TurnComplete

    from ..progress_sink import ProgressSink  # noqa: F401

logger = logging.getLogger(__name__)

EventSink = Callable[[OrchestratorEvent], None]

# Terminal SessionComplete reasons whose IM event is owned by the
# orchestrator's status dispatch (``agent.stagnation`` /
# ``agent.loop_detected`` / ``agent.max_turns_exceeded``). The sink
# callback must NOT also emit a generic ``issue.failed`` here: the
# operator would be double-notified for one failure.
_STATUS_BRANCH_TERMINAL_REASONS = frozenset(
    {
        "stagnation",
        "loop_detected",
        "budget_exhausted",
        "max_turns_exceeded",
    }
)

_SUCCESS_TERMINAL_REASONS = frozenset({"success", "task_complete", "already_completed"})


class OrchestratorEventEmitter:
    """ProgressSink that fans orchestrator events out to IM/audit sinks.

    ``task_id`` is the bound issue id. Phase/turn events are NOT pushed
    to IM (too noisy — they live in audit/LiveView); only terminal
    session events and explicit ``emit()`` calls reach the sinks.
    """

    def __init__(
        self,
        task_id: str,
        sinks: list[EventSink] | None = None,
    ) -> None:
        self.task_id = task_id
        self._sinks: list[EventSink] = list(sinks or [])

    def add_sink(self, sink: EventSink) -> None:
        self._sinks.append(sink)

    def emit(self, event: OrchestratorEvent) -> None:
        """Fan ``event`` to sinks immediately with exception isolation."""
        self._dispatch(event)

    def flush(self, issue_id: str) -> None:
        """Compatibility no-op; events are no longer buffered."""
        return None

    def _dispatch(self, event: OrchestratorEvent) -> None:
        for sink in list(self._sinks):
            try:
                sink(event)
            except Exception as exc:  # noqa: BLE001
                logger.exception("IM event sink failed for %r: %s", sink, exc)

    # -- ProgressSink ----------------------------------------------------
    def on_phase_complete(self, event: "PhaseComplete", session: Any) -> None:
        # Phase events are audit/LiveView only; not pushed to IM.
        return None

    def on_turn_complete(self, event: "TurnComplete", session: Any) -> None:
        # Turn events are audit/LiveView only; not pushed to IM.
        return None

    def on_session_complete(self, event: "SessionComplete", session: Any) -> None:
        reason = getattr(event, "reason", "") or ""
        issue_id = self.task_id or getattr(getattr(session, "issue", None), "id", "") or ""
        if reason in _SUCCESS_TERMINAL_REASONS or reason == "noop_completed":
            # Agent completion is not issue completion. Git sync, verification,
            # PR creation/update, and the human-review gate still run after
            # this callback. The orchestrator emits the authoritative success
            # or pending-review event once those steps finish.
            return
        if reason in _STATUS_BRANCH_TERMINAL_REASONS:
            # The orchestrator's status dispatch emits the specific
            # ``agent.*`` terminal event; emitting a generic
            # ``issue.failed`` here would double-notify (different
            # event_type).
            return
        level = TERMINAL_REASON_LEVEL.get(reason, EventLevel.WARN)
        if reason == "rate_limit_circuit_open":
            event_type = "agent.rate_limit_circuit_open"
            message = "限流熔断，会话终止"
        else:
            event_type = "issue.failed"
            message = f"会话结束: {reason}"
        # Build a rich payload from the session context.
        payload: dict[str, Any] = {}
        issue = getattr(session, "issue", None)
        if issue is not None:
            pr = getattr(issue, "pr_url", None)
            if pr:
                payload["pr"] = pr
            title = getattr(issue, "title", None)
            if title:
                payload["title"] = title
            branch = getattr(issue, "branch_name", None)
            if branch:
                payload["branch"] = branch
        ver = getattr(session, "verification_status", None)
        if ver:
            payload["verification"] = ver
        turns = getattr(session, "turn_count", None)
        if turns:
            payload["turns"] = turns
        self.emit(
            OrchestratorEvent(
                event_type=event_type,
                issue_id=issue_id,
                level=level,
                message=message,
                payload=payload,
            )
        )


__all__ = ["EventSink", "OrchestratorEventEmitter"]
