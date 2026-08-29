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

from __future__ import annotations

from clawcodex_ext.intent_forecast.context import ForecastContext
from clawcodex_ext.intent_forecast.fallback import fallback_suggestions


def test_fallback_returns_no_suggestion_without_signal(tmp_path) -> None:
    context = ForecastContext(cwd=str(tmp_path), response_language="English")

    assert fallback_suggestions(context, min_confidence=0.45) == []


def test_fallback_prioritizes_failure_over_generic_workspace(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        workspace={"git_status": " M file.py"},
        task_state={"blocked_reason": "FAILED tests/test_x.py"},
        intent_stage="debug",
        response_language="English",
    )

    suggestions = fallback_suggestions(context, min_confidence=0.45)

    assert suggestions[0].title == "Fix the recent failure"


def test_fallback_uses_workspace_focus_debug_field(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        workspace={
            "git_status": " M clawcodex_ext/intent_forecast/service.py",
            "focuses": [{"id": "intent_forecast", "confidence": 1.0}],
        },
        response_language="English",
    )

    suggestions = fallback_suggestions(context, min_confidence=0.45)

    assert suggestions[0].title == "Verify Intent Forecast fixes"


def test_fallback_user_strategy_prefers_recent_user_over_workspace(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        current_messages=[{"role": "user", "content": "write docs"}],
        workspace={"git_status": " M file.py"},
        response_language="English",
        intent_strategy="user",
    )

    suggestions = fallback_suggestions(context, min_confidence=0.45)

    assert suggestions[0].title == "Continue the recent task"


def test_fallback_workspace_strategy_prefers_workspace_over_recent_user(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        current_messages=[{"role": "user", "content": "write docs"}],
        workspace={"git_status": " M file.py"},
        response_language="English",
        intent_strategy="workspace",
    )

    suggestions = fallback_suggestions(context, min_confidence=0.45)

    assert suggestions[0].title == "Review current workspace changes"


def test_fallback_history_strategy_prefers_session_next_action(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        current_messages=[{"role": "user", "content": "write docs"}],
        workspace={"git_status": " M file.py"},
        sessions=[
            {
                "session_id": "s1",
                "summary": {"next_action_candidates": ["Continue historical task"]},
            }
        ],
        response_language="English",
        intent_strategy="history",
    )

    suggestions = fallback_suggestions(context, min_confidence=0.45)

    assert suggestions[0].title == "Continue historical task"


def test_fallback_uses_recent_commits_when_only_signal(tmp_path) -> None:
    context = ForecastContext(
        cwd=str(tmp_path),
        response_language="English",
        workspace={
            "recent_commits": [
                {
                    "hash": "abc1234",
                    "short_hash": "abc1234",
                    "subject": "fix: tune no_suggestion_gate threshold",
                    "author": "alice",
                    "timestamp": "1718000000",
                }
            ]
        },
    )

    suggestions = fallback_suggestions(context, min_confidence=0.45)

    assert suggestions
    assert suggestions[0].title == "Continue from recent commits"
    assert "abc1234" in suggestions[0].prompt
    assert "tune no_suggestion_gate threshold" in suggestions[0].prompt
    assert "git:recent-commits" in suggestions[0].refs()


def test_fallback_recent_commits_silent_when_empty(tmp_path) -> None:
    context = ForecastContext(cwd=str(tmp_path), response_language="English")

    suggestions = fallback_suggestions(context, min_confidence=0.45)

    assert all("git:recent-commits" not in s.refs() for s in suggestions)
