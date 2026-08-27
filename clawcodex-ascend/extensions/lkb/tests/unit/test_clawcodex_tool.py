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

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from clawcodex_ext.feature_gate import get_registry
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.defaults import build_default_registry
from clawcodex_ext.tool_system.tools.tasks_v2 import TaskCreateTool
from lkb.clawcodex_commands import LkbCommandOutcome
from lkb.clawcodex_tool import _is_enabled, _lkb_tool_call


def test_default_registry_exposes_agent_callable_lkb_board(
    tmp_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TASKS", "1")
    monkeypatch.setenv("CLAWCODEX_FEATURE_LKB_PLAN_GRAPH", "1")
    get_registry()._overrides["LKB_PLAN_GRAPH"] = True
    context = ToolContext(workspace_root=tmp_path)

    created = TaskCreateTool.call(
        {"subject": "Visible from Lkb tool", "description": "panel test"},
        context,
    )
    task_id = created.output["task"]["id"]

    registry = build_default_registry(provider=object(), load_agent_tools=False)
    tool = registry.get("Lkb")
    assert tool is not None
    assert tool.is_enabled() is True
    result = tool.call({"action": "board", "compact": True}, context)

    assert result.name == "Lkb"
    assert result.output["command"] == "/lkb board --compact"
    assert "LKB BOARD:" in result.output["text"]
    assert task_id in result.output["text"]


def test_agent_callable_lkb_marks_board_resolution_failure_as_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_board_resolution(_args: str, _context: object) -> LkbCommandOutcome:
        return LkbCommandOutcome(
            text="No LKB board found for this workspace.",
            success=False,
            error_code="board_not_found",
        )

    monkeypatch.setattr("lkb.clawcodex_commands._lkb_call", fail_board_resolution)
    result = _lkb_tool_call({"action": "status"}, ToolContext(workspace_root=tmp_path))

    assert result.is_error is True
    assert result.output["success"] is False
    assert result.output["error_code"] == "board_not_found"


def test_agent_callable_lkb_uses_outcome_not_user_facing_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def successful_message_with_error_word(_args: str, _context: object) -> LkbCommandOutcome:
        return LkbCommandOutcome(
            text="The audit says an earlier error was resolved.",
            success=True,
        )

    monkeypatch.setattr("lkb.clawcodex_commands._lkb_call", successful_message_with_error_word)
    result = _lkb_tool_call({"action": "status"}, ToolContext(workspace_root=tmp_path))

    assert result.is_error is False
    assert result.output["success"] is True


def test_is_enabled_propagates_a_transitive_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def missing_dependency(name: str, *args: object, **kwargs: object):
        if name == "lkb.flags":
            raise ModuleNotFoundError("No module named 'transitive_dep'", name="transitive_dep")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_dependency)

    with pytest.raises(ModuleNotFoundError, match="transitive_dep"):
        _is_enabled()


def test_is_enabled_returns_false_for_normal_feature_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lkb.flags.is_plan_graph_enabled", lambda: False)

    assert _is_enabled() is False
