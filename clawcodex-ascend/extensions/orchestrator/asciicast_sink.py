#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
"""Orchestrator adapter for asciicast recording (F-REC).

Implements the :class:`ProgressSink` protocol so it can be registered
into the per-session :class:`CompositeProgressSink` built by
``Orchestrator._build_session_sink``. Maps agent progress events into
the asciicast capture handle:

* :meth:`on_phase_complete` → ``marker("phase:{n}")`` + short text line
* :meth:`on_turn_complete`   → debug-only (turns are noisy, mirrors
  :class:`ToolContextProgressSink`'s policy)
* :meth:`on_session_complete`→ ``marker("session:{reason}")``

Per the F-REC plan we use *structured-event projection* for the
orchestrator: semantic frames instead of raw stdout mirror. The
orchestrator's :class:`StatusDashboard` already prints the rendered
view to the live console, so the recording doesn't need to duplicate
that — a navigable sequence of phase markers + session-end marker is
enough for ``asciinema play`` to show a meaningful timeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from extensions.capabilities.recorder import AsciicastCapture
from extensions.api.query import (
    PhaseComplete,
    SessionComplete,
    TurnComplete,
)

if TYPE_CHECKING:
    from .agent_runner import AgentSession

logger = logging.getLogger(__name__)


class AsciicastSink:
    """``ProgressSink`` that mirrors agent progress into an asciicast capture.

    The sink is bound to one ``task_id`` (matching the orchestrator's
    per-session composite). It owns no shared state with other sinks;
    concurrent issues each get their own sink instance.
    """

    task_id: str = ""

    def __init__(
        self,
        capture: AsciicastCapture,
        task_id: str,
        *,
        phases_total: int | None = None,
    ) -> None:
        self._capture = capture
        self.task_id = task_id
        self._phases_total = phases_total

    def on_phase_complete(
        self,
        event: PhaseComplete,
        session: "AgentSession",
    ) -> None:
        phase_label = format_phase_label(event.phase, self._phases_total)
        try:
            self._capture.marker(phase_label, text=phase_label)
        except Exception as exc:  # noqa: BLE001
            # Recording failures must never block the orchestrator —
            # log and move on so the live run continues.
            logger.warning(
                "AsciicastSink phase marker failed (task_id=%s): %s",
                self.task_id,
                exc,
            )

    def on_turn_complete(
        self,
        event: TurnComplete,
        session: "AgentSession",
    ) -> None:
        # Mirror ToolContextProgressSink: turns are too noisy for the
        # asciicast timeline. Debug-only so recording tools can still
        # verify the sink saw the event.
        logger.debug(
            "AsciicastSink saw turn %d for task %s",
            event.turn,
            self.task_id,
        )

    def on_session_complete(
        self,
        event: SessionComplete,
        session: "AgentSession",
    ) -> None:
        marker_label = f"session:{event.reason}"
        text = f"Session {self.task_id} ended: {event.reason}" if self.task_id else f"Session ended: {event.reason}"
        try:
            self._capture.marker(marker_label, text=text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "AsciicastSink session marker failed (task_id=%s): %s",
                self.task_id,
                exc,
            )


def format_phase_label(phase: int, total: int | None) -> str:
    """Render a phase marker like ``[phase 3/7]`` or ``[phase 3]``."""
    suffix = f"/{total}" if total else ""
    return f"[phase {phase}{suffix}]"


__all__ = ["AsciicastSink", "format_phase_label"]
