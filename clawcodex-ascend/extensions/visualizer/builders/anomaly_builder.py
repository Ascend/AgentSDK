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

"""Detect session anomalies using duration, cost, turn-count, and no-op heuristics."""

from __future__ import annotations

import logging

from extensions.visualizer.models.viz_models import (
    Anomaly,
    AnomalySeverity,
    AnomalyType,
    BarType,
    SessionVizData,
)

logger = logging.getLogger(__name__)

# Thresholds for duration, cost, turn-count, and no-op heuristics
_LONG_TOOL_THRESHOLD_MS = 30_000  # 30 seconds
_HIGH_COST_THRESHOLD_USD = 5.0  # $5
_MAX_TURNS_WARNING = 50
_MAX_TURNS_CRITICAL = 100
_NO_OP_TURNS = 5
_READ_ONLY_SPIRAL_TURNS = 4


class AnomalyBuilder:
    """Build anomaly list from session viz data."""

    def build(self, session: SessionVizData) -> list[Anomaly]:
        anomalies: list[Anomaly] = []

        # Check end reason anomalies
        if session.end_reason:
            anomalies.extend(self._end_reason_anomalies(session))

        # Check long tool executions
        anomalies.extend(self._long_tool_anomalies(session))

        # Check high cost
        if session.stats.cost_usd > _HIGH_COST_THRESHOLD_USD:
            anomalies.append(
                Anomaly(
                    type=AnomalyType.BUDGET_EXHAUSTED,
                    severity=AnomalySeverity.HIGH,
                    session_id=session.session_id,
                    description=f"Session cost ${session.stats.cost_usd:.2f} exceeds threshold ${_HIGH_COST_THRESHOLD_USD}",
                    timestamp=session.end_time or session.start_time,
                    suggestion="Review tool usage patterns and consider cost optimization.",
                )
            )

        # Check turn count
        if session.turn_count > _MAX_TURNS_CRITICAL:
            anomalies.append(
                Anomaly(
                    type=AnomalyType.MAX_TURNS,
                    severity=AnomalySeverity.CRITICAL,
                    session_id=session.session_id,
                    description=f"Session exceeded {_MAX_TURNS_CRITICAL} turns ({session.turn_count})",
                    timestamp=session.end_time or session.start_time,
                    suggestion="Consider increasing max_turns or investigating loop behavior.",
                )
            )
        elif session.turn_count > _MAX_TURNS_WARNING:
            anomalies.append(
                Anomaly(
                    type=AnomalyType.MAX_TURNS,
                    severity=AnomalySeverity.MEDIUM,
                    session_id=session.session_id,
                    description=f"Session reached {session.turn_count} turns",
                    timestamp=session.end_time or session.start_time,
                    suggestion="Monitor for potential stagnation or loops.",
                )
            )

        return anomalies

    def _end_reason_anomalies(self, session: SessionVizData) -> list[Anomaly]:
        """Generate anomalies from session end reason."""
        anomalies: list[Anomaly] = []
        reason = session.end_reason
        ts = session.end_time or session.start_time

        reason_map: dict[str, tuple[AnomalyType, AnomalySeverity, str]] = {
            "noop_completed": (
                AnomalyType.NO_OP,
                AnomalySeverity.MEDIUM,
                "Agent completed without making changes. Consider if the deliverable already exists.",
            ),
            "stagnation": (
                AnomalyType.STAGNATION,
                AnomalySeverity.HIGH,
                "Agent stagnated — no progress detected for multiple turns.",
            ),
            "loop_detected": (
                AnomalyType.LOOP,
                AnomalySeverity.HIGH,
                "Agent entered a loop — repeated the same tool call pattern.",
            ),
            "read_only_loop": (
                AnomalyType.READ_ONLY_SPIRAL,
                AnomalySeverity.MEDIUM,
                "Agent spent consecutive turns only reading without making changes.",
            ),
            "budget_exhausted": (
                AnomalyType.BUDGET_EXHAUSTED,
                AnomalySeverity.CRITICAL,
                "Session terminated due to budget exhaustion.",
            ),
            "max_turns_exceeded": (
                AnomalyType.MAX_TURNS,
                AnomalySeverity.HIGH,
                "Session exceeded max_turns limit.",
            ),
            "failed": (AnomalyType.CUSTOM, AnomalySeverity.HIGH, "Session failed with an error."),
        }

        if reason in reason_map:
            atype, severity, suggestion = reason_map[reason]
            anomalies.append(
                Anomaly(
                    type=atype,
                    severity=severity,
                    session_id=session.session_id,
                    description=session.end_summary or f"Session ended with reason: {reason}",
                    timestamp=ts,
                    suggestion=suggestion,
                )
            )

        return anomalies

    def _long_tool_anomalies(self, session: SessionVizData) -> list[Anomaly]:
        """Detect tool calls that took unusually long."""
        anomalies: list[Anomaly] = []
        for timeline_bar in session.timeline:
            if timeline_bar.type == BarType.TOOL_CALL and timeline_bar.duration_ms > _LONG_TOOL_THRESHOLD_MS:
                anomalies.append(
                    Anomaly(
                        type=AnomalyType.LONG_TOOL,
                        severity=AnomalySeverity.LOW,
                        session_id=session.session_id,
                        description=f"Tool '{timeline_bar.label}' took {timeline_bar.duration_ms / 1000:.1f}s",
                        timestamp=timeline_bar.start_time,
                        suggestion="Consider if the tool input can be narrowed or if it's a known slow operation.",
                        bar_id=timeline_bar.id,
                    )
                )
        return anomalies
