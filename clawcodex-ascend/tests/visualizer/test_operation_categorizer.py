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

"""Tests for the OperationCategorizer legend mapper."""

from __future__ import annotations


from extensions.visualizer.builders.operation_categorizer import OperationCategorizer
from extensions.visualizer.models.viz_models import (
    BarType,
    OperationCategory,
    TimelineBar,
)


def _bar(label: str, detail: dict | None = None, bar_type: BarType = BarType.TOOL_CALL) -> TimelineBar:
    return TimelineBar(
        id=f"x-{label}",
        type=bar_type,
        label=label,
        start_time=0.0,
        end_time=1.0,
        duration_ms=1000,
        detail=detail or {"tool_name": label},
    )


class TestOperationCategorizerToolRules:
    def test_read_tools(self):
        cat = OperationCategorizer()
        for name in ("Read", "Glob", "Grep", "WebFetch", "WebSearch", "LS"):
            assert cat.categorize(_bar(name)) == OperationCategory.READ, name

    def test_execute_tools(self):
        cat = OperationCategorizer()
        for name in ("Bash", "Execute", "TaskKill", "BashOutput", "KillShell", "Shell"):
            assert cat.categorize(_bar(name)) == OperationCategory.EXECUTE, name

    def test_write_tools(self):
        cat = OperationCategorizer()
        for name in ("Write", "Edit", "MultiEdit", "NotebookEdit", "TodoWrite", "Patch"):
            assert cat.categorize(_bar(name)) == OperationCategory.WRITE, name

    def test_orchestrate_tools(self):
        cat = OperationCategorizer()
        for name in ("Agent", "Task", "SendMessage", "TeamCreate"):
            assert cat.categorize(_bar(name)) == OperationCategory.ORCHESTRATE, name

    def test_unknown_tool_is_other(self):
        cat = OperationCategorizer()
        assert cat.categorize(_bar("MagicalUnicorn")) == OperationCategory.OTHER


class TestOperationCategorizerExplicitFlags:
    def test_isAgentInvocation_flag(self):
        cat = OperationCategorizer()
        bar = _bar(
            "Bash",
            detail={
                "tool_name": "Bash",
                "isAgentInvocation": True,
            },
        )
        # Explicit flag wins over tool_name match
        assert cat.categorize(bar) == OperationCategory.ORCHESTRATE

    def test_is_agent_invocation_snake_case(self):
        cat = OperationCategorizer()
        bar = _bar(
            "Read",
            detail={
                "tool_name": "Read",
                "is_agent_invocation": True,
            },
        )
        assert cat.categorize(bar) == OperationCategory.ORCHESTRATE

    def test_pre_set_category_returned(self):
        cat = OperationCategorizer()
        bar = _bar("Bash")
        bar.category = OperationCategory.WRITE
        assert cat.categorize(bar) == OperationCategory.WRITE

    def test_pre_set_other_can_be_refined(self):
        cat = OperationCategorizer()
        bar = _bar("LLM text", bar_type=BarType.LLM_CALL)
        bar.category = OperationCategory.OTHER
        assert cat.categorize(bar) == OperationCategory.LLM_TEXT


class TestOperationCategorizerBarTypeFallback:
    def test_phase_type_is_orchestrate(self):
        cat = OperationCategorizer()
        bar = _bar("phase-1", bar_type=BarType.PHASE)
        bar.detail = {}  # no tool_name
        assert cat.categorize(bar) == OperationCategory.ORCHESTRATE

    def test_session_type_is_orchestrate(self):
        cat = OperationCategorizer()
        bar = _bar("session-root", bar_type=BarType.SESSION)
        bar.detail = {}
        assert cat.categorize(bar) == OperationCategory.ORCHESTRATE

    def test_llm_call_type_is_llm_text(self):
        cat = OperationCategorizer()
        bar = _bar("LLM text", bar_type=BarType.LLM_CALL)
        bar.detail = {}
        assert cat.categorize(bar) == OperationCategory.LLM_TEXT

    def test_turn_type_is_turn(self):
        cat = OperationCategorizer()
        bar = _bar("turn-1", bar_type=BarType.TURN)
        bar.detail = {}
        assert cat.categorize(bar) == OperationCategory.TURN

    def test_isBackground_flag_is_background(self):
        cat = OperationCategorizer()
        bar = _bar("polling", detail={"isBackground": True})
        assert cat.categorize(bar) == OperationCategory.BACKGROUND

    def test_is_background_snake_case(self):
        cat = OperationCategorizer()
        bar = _bar("polling", detail={"is_background": True})
        assert cat.categorize(bar) == OperationCategory.BACKGROUND

    def test_background_flag_wins_over_bar_type(self):
        cat = OperationCategorizer()
        bar = _bar("background thinking", detail={"isBackground": True}, bar_type=BarType.LLM_CALL)
        assert cat.categorize(bar) == OperationCategory.BACKGROUND

    def test_tool_result_type_is_other(self):
        cat = OperationCategorizer()
        bar = _bar("tool result echo", bar_type=BarType.TOOL_RESULT)
        bar.detail = {}
        assert cat.categorize(bar) == OperationCategory.OTHER


class TestOperationCategoryColor:
    def test_legend_colors_distinct(self):
        colors = {c.color for c in OperationCategory}
        assert len(colors) == len(OperationCategory), "Categories must have distinct colors"

    def test_legend_labels_present(self):
        labels = {c.label for c in OperationCategory}
        # All eight Chinese labels, including the three categories split from OTHER
        assert {"读取", "执行", "写入", "编排", "推理", "轮次", "后台", "其他"} <= labels
