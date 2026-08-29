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

"""Lightweight goal observability hooks.

The upstream implementation routes these events through analytics and metrics.
This Python port keeps a small observer boundary so tests, local logs, and
future telemetry adapters can subscribe without spreading side effects across
commands, tools, or UI code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from clawcodex_ext.goal.model import ThreadGoal, ThreadGoalStatus


@dataclass(frozen=True)
class GoalObservation:
    kind: str
    thread_id: str
    goal_id: str | None = None
    status: str | None = None
    reason: str | None = None
    token_delta: int | None = None
    time_delta_seconds: int | None = None
    cumulative_tokens: int | None = None
    cumulative_time_seconds: int | None = None


class GoalObserver(Protocol):
    def record(self, observation: GoalObservation) -> None:
        """Record one goal observation."""


class GoalObservationRecorder:
    """In-memory observer used by tests and local diagnostics."""

    def __init__(self) -> None:
        self.observations: list[GoalObservation] = []

    def record(self, observation: GoalObservation) -> None:
        self.observations.append(observation)


def record_goal_created(observer: GoalObserver | None, goal: ThreadGoal) -> None:
    _record_goal(observer, "created", goal)


def record_goal_cleared(observer: GoalObserver | None, goal: ThreadGoal) -> None:
    _record_goal(observer, "cleared", goal)


def record_status_transition(
    observer: GoalObserver | None,
    previous_status: ThreadGoalStatus | None,
    goal: ThreadGoal,
) -> None:
    if previous_status == goal.status:
        return
    if previous_status is None and goal.status is ThreadGoalStatus.ACTIVE:
        return
    kind = {
        ThreadGoalStatus.ACTIVE: "resumed",
        ThreadGoalStatus.PAUSED: "paused",
        ThreadGoalStatus.BLOCKED: "blocked",
        ThreadGoalStatus.USAGE_LIMITED: "usage_limited",
        ThreadGoalStatus.BUDGET_LIMITED: "budget_limited",
        ThreadGoalStatus.COMPLETE: "complete",
    }.get(goal.status)
    if kind is None:
        return
    _record_goal(observer, kind, goal)


def record_usage_accounted(
    observer: GoalObserver | None,
    goal: ThreadGoal,
    *,
    token_delta: int,
    time_delta_seconds: int,
) -> None:
    if observer is None:
        return
    if token_delta <= 0 and time_delta_seconds <= 0:
        return
    observer.record(
        GoalObservation(
            kind="usage_accounted",
            thread_id=goal.thread_id,
            goal_id=goal.goal_id,
            status=goal.status.value,
            token_delta=max(int(token_delta), 0),
            time_delta_seconds=max(int(time_delta_seconds), 0),
            cumulative_tokens=goal.tokens_used,
            cumulative_time_seconds=goal.time_used_seconds,
        )
    )


def record_continuation_skipped(
    observer: GoalObserver | None,
    *,
    thread_id: str,
    reason: str,
    goal: ThreadGoal | None = None,
) -> None:
    if observer is None:
        return
    observer.record(
        GoalObservation(
            kind="continuation_skipped",
            thread_id=thread_id,
            goal_id=goal.goal_id if goal is not None else None,
            status=goal.status.value if goal is not None else None,
            reason=reason,
        )
    )


def _record_goal(observer: GoalObserver | None, kind: str, goal: ThreadGoal) -> None:
    if observer is None:
        return
    observer.record(
        GoalObservation(
            kind=kind,
            thread_id=goal.thread_id,
            goal_id=goal.goal_id,
            status=goal.status.value,
        )
    )


__all__ = [
    "GoalObservation",
    "GoalObservationRecorder",
    "GoalObserver",
    "record_continuation_skipped",
    "record_goal_cleared",
    "record_goal_created",
    "record_status_transition",
    "record_usage_accounted",
]
