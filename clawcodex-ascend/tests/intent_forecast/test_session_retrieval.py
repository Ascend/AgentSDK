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

from clawcodex_ext.intent_forecast.session_retrieval import rank_session_rows


def test_session_retrieval_prefers_changed_file_overlap_over_recency() -> None:
    rows = [
        {
            "session_id": "orchestrator",
            "title": "Orchestrator cleanup",
            "last_updated": 200,
            "summary": {"files_touched": ["extensions/orchestrator/orchestrator.py"]},
        },
        {
            "session_id": "forecast",
            "title": "Intent Forecast fallback",
            "last_updated": 100,
            "summary": {
                "files_touched": ["clawcodex_ext/intent_forecast/service.py"],
                "next_action_candidates": ["Run focused tests"],
            },
        },
    ]

    ranked = rank_session_rows(
        rows,
        cwd="C:/WorkSpace/clawcodex",
        changed_files=["clawcodex_ext/intent_forecast/service.py"],
        recent_text="continue intent forecast",
        limit=2,
    )

    assert [row["session_id"] for row in ranked] == ["forecast", "orchestrator"]
    assert ranked[0]["relevance_score"] > ranked[1]["relevance_score"]


def test_session_retrieval_user_strategy_prefers_recent_text() -> None:
    rows = [
        {
            "session_id": "workspace",
            "summary": {"files_touched": ["clawcodex_ext/intent_forecast/service.py"]},
        },
        {"session_id": "user", "title": "Write docs from user request"},
    ]

    ranked = rank_session_rows(
        rows,
        cwd="repo",
        changed_files=["clawcodex_ext/intent_forecast/service.py"],
        recent_text="write docs from user request",
        limit=2,
        strategy="user",
    )

    assert ranked[0]["session_id"] == "user"


def test_session_retrieval_workspace_strategy_prefers_changed_files() -> None:
    rows = [
        {
            "session_id": "workspace",
            "summary": {"files_touched": ["clawcodex_ext/intent_forecast/service.py"]},
        },
        {"session_id": "user", "title": "Write docs from user request"},
    ]

    ranked = rank_session_rows(
        rows,
        cwd="repo",
        changed_files=["clawcodex_ext/intent_forecast/service.py"],
        recent_text="write docs from user request",
        limit=2,
        strategy="workspace",
    )

    assert ranked[0]["session_id"] == "workspace"
