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

"""Contract tests for the model-facing SkillSearch tool wrapper."""
# pylint: disable=no-name-in-module

from __future__ import annotations

from clawcodex_ext.tool_system.tools.skill_search import _skill_search_call


def test_missing_search_query_returns_structured_tool_error() -> None:
    result = _skill_search_call({}, None)  # type: ignore[arg-type]

    assert result.name == "SkillSearch"
    assert result.is_error is True
    assert result.output == (
        "query is required for search action — provide a natural language description of the skill you need"
    )


def test_unknown_action_returns_structured_tool_error() -> None:
    result = _skill_search_call({"action": "unknown"}, None)  # type: ignore[arg-type]

    assert result.name == "SkillSearch"
    assert result.is_error is True
    assert "Unknown action: unknown" in result.output


def test_stats_when_search_is_disabled_returns_normal_result(monkeypatch) -> None:
    class _DisabledSearcher:
        async def ensure_index(self) -> None:
            raise RuntimeError("disabled")

    monkeypatch.setattr(
        "clawcodex_ext.tool_system.tools.skill_search._get_searcher",
        _DisabledSearcher,
    )

    result = _skill_search_call({"action": "stats"}, None)  # type: ignore[arg-type]

    assert result.name == "SkillSearch"
    assert result.is_error is False
    assert result.output == "Index not loaded (feature flag may be off)."
