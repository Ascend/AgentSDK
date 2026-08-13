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

"""State Journal Sink — bridges ProgressReporter to NDJSON.

Implements the :class:`ProgressSink` protocol so that agent progress events
(phase/turn/session completion) are automatically written to the
``state_journal.ndjson`` file alongside the explicit events written by
``AgentRunner``.

Design: this sink is added to the ``CompositeProgressSink`` fan-out
alongside the existing ``ToolContextProgressSink``.  It holds a reference
to the :class:`StateJournalWriter` and translates the three ``on_*``
callbacks into NDJSON events.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..api.query import PhaseComplete, SessionComplete, TurnComplete  # pylint: disable=import-error,no-name-in-module

logger = logging.getLogger(__name__)


class StateJournalSink:
    """A :class:`ProgressSink` that writes events to the State Journal.

    Parameters
    ----------
    writer:
        The :class:`StateJournalWriter` instance that owns the NDJSON file.
    task_id:
        The issue/task id this sink is bound to (injected into every event).
    """

    def __init__(self, writer: Any, task_id: str) -> None:
        self._writer = writer
        self._task_id = task_id
        self._phase_count = 0

    # ------------------------------------------------------------------
    # ProgressSink protocol
    # ------------------------------------------------------------------

    def on_phase_complete(
        self,
        event: "PhaseComplete",
        session: Any,
    ) -> None:
        """A logical phase completed — emit a ``phase`` event."""
        self._phase_count += 1
        phase_name = getattr(event, "phase", self._phase_count)
        progress = getattr(event, "progress", None)
        message = getattr(event, "message", "")
        self._writer.write_phase(
            phase=str(phase_name),
            progress=progress,
            message=message or f"Phase {self._phase_count} completed",
            issue_id=self._task_id,
        )

    def on_turn_complete(
        self,
        event: "TurnComplete",
        session: Any,
    ) -> None:
        """A single turn completed — emit a ``phase`` progress update."""
        turn = getattr(event, "turn", 0)
        self._writer.write_phase(
            phase="agent_turn",
            progress=None,
            message=f"Turn {turn} completed",
            issue_id=self._task_id,
        )

    def on_session_complete(
        self,
        event: "SessionComplete",
        session: Any,
    ) -> None:
        """The whole session is ending — emit a ``complete`` event."""
        status = getattr(session, "status", "completed")
        reason = getattr(session, "session_end_reason", None) or ""
        summary = getattr(session, "session_end_summary", "")
        self._writer.write_complete(
            issue_id=self._task_id,
            overall_status=status,
            message=f"{reason}: {summary}".strip(": "),
        )
