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

from __future__ import annotations

from clawcodex_ext.intent_forecast.focus import compute_workspace_focuses


def test_weak_forecast_text_does_not_force_intent_forecast_focus() -> None:
    focuses = compute_workspace_focuses(
        changed_files=["README.md"],
        recent_messages=[{"role": "user", "content": "forecast the next release risks"}],
    )

    assert all(item["id"] != "intent_forecast" for item in focuses)


def test_intent_forecast_path_is_strong_focus() -> None:
    focuses = compute_workspace_focuses(
        changed_files=["clawcodex_ext/intent_forecast/service.py"],
        recent_messages=[],
    )

    assert focuses[0]["id"] == "intent_forecast"
    assert focuses[0]["confidence"] >= 1.0


def test_cross_module_changes_keep_multiple_focuses() -> None:
    focuses = compute_workspace_focuses(
        changed_files=[
            "clawcodex_ext/intent_forecast/service.py",
            "clawcodex_ext/tui/app.py",
        ],
        recent_messages=[],
    )

    assert {item["id"] for item in focuses} == {"intent_forecast", "tui"}
