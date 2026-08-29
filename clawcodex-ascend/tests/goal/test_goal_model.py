#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

# pylint: disable=no-name-in-module

"""Thread goal model parity tests for Spec 2."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from clawcodex_ext.goal.model import GoalCompletionMode, ThreadGoal, ThreadGoalStatus


def test_thread_goal_status_matches_upstream_six_state_set() -> None:
    assert [status.value for status in ThreadGoalStatus] == [
        "active",
        "paused",
        "blocked",
        "usage_limited",
        "budget_limited",
        "complete",
    ]


@pytest.mark.parametrize(
    ("wire_value", "expected"),
    [
        ("active", ThreadGoalStatus.ACTIVE),
        ("paused", ThreadGoalStatus.PAUSED),
        ("blocked", ThreadGoalStatus.BLOCKED),
        ("usage_limited", ThreadGoalStatus.USAGE_LIMITED),
        ("budget_limited", ThreadGoalStatus.BUDGET_LIMITED),
        ("complete", ThreadGoalStatus.COMPLETE),
    ],
)
def test_thread_goal_status_round_trips_wire_values(wire_value: str, expected: ThreadGoalStatus) -> None:
    assert ThreadGoalStatus.from_wire(wire_value) is expected
    assert expected.to_wire() == wire_value


def test_unknown_thread_goal_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown thread goal status"):
        ThreadGoalStatus.from_wire("max_turns")


def test_thread_goal_serializes_without_completion_timestamp_field() -> None:
    created_at = datetime(2026, 6, 29, 10, 0, tzinfo=timezone.utc)
    updated_at = datetime(2026, 6, 29, 10, 1, tzinfo=timezone.utc)
    goal = ThreadGoal(
        thread_id="thread-1",
        goal_id="goal-1",
        objective="ship state layer",
        status=ThreadGoalStatus.ACTIVE,
        token_budget=1000,
        tokens_used=250,
        time_used_seconds=12,
        created_at=created_at,
        updated_at=updated_at,
    )

    payload = goal.to_dict()
    completed_field = "completed" + "_at"

    assert payload == {
        "thread_id": "thread-1",
        "goal_id": "goal-1",
        "objective": "ship state layer",
        "status": "active",
        "token_budget": 1000,
        "tokens_used": 250,
        "time_used_seconds": 12,
        "completion_mode": "tool",
        "evaluation_count": 0,
        "last_evaluation_reason": None,
        "created_at": "2026-06-29T10:00:00+00:00",
        "updated_at": "2026-06-29T10:01:00+00:00",
    }
    assert completed_field not in payload
    assert ThreadGoal.from_dict(payload) == goal


def test_thread_goal_from_dict_normalizes_utc_datetime_suffix() -> None:
    goal = ThreadGoal.from_dict(
        {
            "thread_id": "thread-1",
            "goal_id": "goal-1",
            "objective": "restore",
            "status": "complete",
            "token_budget": None,
            "tokens_used": 1,
            "time_used_seconds": 2,
            "completion_mode": "evaluator",
            "evaluation_count": 3,
            "last_evaluation_reason": "verified by tests",
            "created_at": "2026-06-29T10:00:00Z",
            "updated_at": "2026-06-29T10:01:00Z",
        }
    )

    assert goal.status is ThreadGoalStatus.COMPLETE
    assert goal.completion_mode is GoalCompletionMode.EVALUATOR
    assert goal.evaluation_count == 3
    assert goal.last_evaluation_reason == "verified by tests"
    assert goal.created_at.tzinfo is timezone.utc
    assert goal.updated_at.tzinfo is timezone.utc


def test_legacy_goal_dict_defaults_to_tool_completion_mode() -> None:
    goal = ThreadGoal.from_dict(
        {
            "thread_id": "thread-1",
            "goal_id": "goal-1",
            "objective": "legacy",
            "status": "active",
            "token_budget": None,
            "tokens_used": 0,
            "time_used_seconds": 0,
            "created_at": "2026-06-29T10:00:00Z",
            "updated_at": "2026-06-29T10:01:00Z",
        }
    )

    assert goal.completion_mode is GoalCompletionMode.TOOL
