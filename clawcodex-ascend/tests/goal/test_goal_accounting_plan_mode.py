#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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
# pylint: disable=no-name-in-module
"""Goal plan-mode accounting unit test."""

from clawcodex_ext.goal.accounting import GoalAccountingState


def test_plan_mode_turns_do_not_account_goal_progress() -> None:
    accounting = GoalAccountingState()
    accounting.start_turn("turn-1", plan_mode=True)
    accounting.bind_goal_to_turn("turn-1", "goal-1")
    accounting.record_token_usage(
        "turn-1",
        {"input_tokens": 20, "output_tokens": 5},
    )

    assert accounting.progress_snapshot("turn-1") is None
