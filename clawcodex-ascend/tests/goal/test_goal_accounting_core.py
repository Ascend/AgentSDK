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

"""Core goal runtime accounting tests."""

from __future__ import annotations

from clawcodex_ext.goal.accounting import (
    BudgetLimitedGoalDisposition,
    GoalAccountingState,
    goal_token_delta_for_usage,
)
from clawcodex_ext.goal.model import ThreadGoalStatus
from clawcodex_ext.goal.service import _unfinished_goal_error


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def test_goal_token_delta_excludes_cached_input() -> None:
    usage = {"input_tokens": 100, "cache_read_input_tokens": 40, "output_tokens": 25}
    assert goal_token_delta_for_usage(usage) == 85
    assert goal_token_delta_for_usage({"input_tokens": 12, "output_tokens": 8}) == 20


def test_progress_snapshot_tracks_goal_and_avoids_double_counting() -> None:
    clock = FakeClock()
    accounting = GoalAccountingState(clock=clock)
    accounting.start_turn("turn-1", plan_mode=False)
    accounting.bind_goal_to_turn("turn-1", "goal-1")
    clock.now += 5
    accounting.record_token_usage(
        "turn-1",
        {"input_tokens": 20, "cache_read_input_tokens": 5, "output_tokens": 7},
    )

    snapshot = accounting.progress_snapshot("turn-1")
    assert snapshot is not None
    assert (snapshot.expected_goal_id, snapshot.token_delta) == ("goal-1", 22)
    assert snapshot.time_delta_seconds == 5
    accounting.mark_progress_accounted_for_status(
        "turn-1",
        snapshot,
        ThreadGoalStatus.ACTIVE,
        BudgetLimitedGoalDisposition.KEEP_ACTIVE,
    )
    assert accounting.progress_snapshot("turn-1") is None


def test_starting_a_new_turn_retires_the_abandoned_turn() -> None:
    accounting = GoalAccountingState()
    accounting.start_turn("turn-1", plan_mode=False)
    accounting.bind_goal_to_turn("turn-1", "goal-1")
    accounting.start_turn("turn-2", plan_mode=False)
    assert accounting.record_token_usage("turn-1", {"input_tokens": 1}) is None


def test_started_goal_binding_is_immutable_within_a_turn() -> None:
    accounting = GoalAccountingState()
    accounting.start_turn("turn-1", plan_mode=False)
    accounting.bind_goal_to_turn("turn-1", "goal-1")
    accounting.bind_goal_to_turn("turn-1", "goal-2")
    assert accounting.turn_started_goal_id("turn-1") == "goal-1"


def test_finish_turn_preserves_thread_goal_accounting_state() -> None:
    clock = FakeClock()
    accounting = GoalAccountingState(clock=clock)
    accounting.start_turn("turn-1", plan_mode=False)
    accounting.bind_goal_to_turn("turn-1", "goal-1")
    assert accounting.mark_budget_limit_reported_if_new("goal-1") is True

    accounting.finish_turn("turn-1")
    clock.now += 3

    assert accounting.current_turn_id() is None
    snapshot = accounting.idle_progress_snapshot()
    assert snapshot is not None
    assert snapshot.expected_goal_id == "goal-1"
    assert snapshot.time_delta_seconds == 3
    assert accounting.mark_budget_limit_reported_if_new("goal-1") is False


def test_unfinished_goal_error_is_caller_neutral() -> None:
    message = str(_unfinished_goal_error())
    assert "ask the user" not in message
    assert "complete, replace, or clear" in message
