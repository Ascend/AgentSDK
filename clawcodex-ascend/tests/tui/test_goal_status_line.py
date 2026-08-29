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

"""Spec-6 goal status indicator tests for the TUI status line."""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from src.tui.state import AppState
from src.tui.widgets.status_line import _goal_status_segment


def test_status_line_matches_claude_active_goal_indicator() -> None:
    state = AppState(model="test-model", provider="test-provider")
    state.set_goal_status(
        {
            "status": "active",
            "activeSince": 100.0,
        }
    )

    rendered = _goal_status_segment(state.goal_status, now=113.9)

    assert rendered == "◎ /goal active (13s)"


@pytest.mark.parametrize(
    "status",
    ["paused", "blocked", "usage_limited", "budget_limited", "complete"],
)
def test_status_line_shows_non_active_goal_states(status: str) -> None:
    goal = {"status": status, "timeUsedSeconds": 120}

    assert _goal_status_segment(goal, now=200.0) == f"◎ /goal {status} (2m)"


def test_status_line_falls_back_to_accounted_elapsed_without_live_baseline() -> None:
    goal = {"status": "active", "timeUsedSeconds": 120}

    assert _goal_status_segment(goal, now=200.0) == "◎ /goal active (2m)"
