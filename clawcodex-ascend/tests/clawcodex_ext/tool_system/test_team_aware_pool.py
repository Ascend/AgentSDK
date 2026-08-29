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

"""Unit tests for clawcodex_ext.tool_system.team_aware_pool.

Covers the high-level wrapper that pulls the full tool list from a
ToolRegistry and applies the team-context filter in one call. This is
the entry point used by ``src/query/agent_loop_compat.py`` and
``clawcodex_ext/repl/core.py`` to materialize the model-facing tool
list.
"""

from __future__ import annotations


# pylint: disable=E0611
from dataclasses import dataclass


from clawcodex_ext.tool_system import get_team_aware_tool_list
from src.tool_system.registry import ToolRegistry


@dataclass
class _FakeTool:
    name: str
    aliases: tuple[str, ...] = ()


def _build_registry_with(*tool_names: str) -> ToolRegistry:
    """Build a minimal ToolRegistry populated with the given tool names.

    Avoids depending on the real ``src.tool_system.tools.ALL_STATIC_TOOLS``
    (which is a moving target across PRs) — we only need the registry
    contract: ``list_tools()`` returns a list of objects with a
    ``name`` attribute.
    """
    registry = ToolRegistry()
    for n in tool_names:
        registry.register(_FakeTool(name=n))
    return registry


def test_get_team_aware_tool_list_drops_send_message_when_no_team() -> None:
    registry = _build_registry_with("Read", "Bash", "SendMessage", "TeamCreate", "Edit")
    result = get_team_aware_tool_list(registry, team=None)
    assert [t.name for t in result] == ["Read", "Bash", "TeamCreate", "Edit"]


def test_get_team_aware_tool_list_keeps_all_when_team_active() -> None:
    registry = _build_registry_with(
        "Read",
        "Bash",
        "SendMessage",
        "TeamCreate",
        "TeamDelete",
        "Edit",
    )
    team = {"team_name": "t", "lead_agent_id": "lead-1"}
    result = get_team_aware_tool_list(registry, team=team)
    assert [t.name for t in result] == [
        "Read",
        "Bash",
        "SendMessage",
        "TeamCreate",
        "TeamDelete",
        "Edit",
    ]


def test_get_team_aware_tool_list_keeps_all_when_team_is_empty_dict() -> None:
    # Empty dict is "no team active" per has_team_context — drops
    # active-team-only tools but keeps TeamCreate as the bootstrap entry.
    registry = _build_registry_with("Read", "SendMessage", "TeamCreate")
    result = get_team_aware_tool_list(registry, team={})
    assert [t.name for t in result] == ["Read", "TeamCreate"]


def test_get_team_aware_tool_list_keeps_team_create_when_no_team() -> None:
    # ``TeamCreate`` is the bootstrap tool for creating the active team
    # context; ``TeamDelete`` still only makes sense once a team exists.
    registry = _build_registry_with(
        "Read",
        "TeamCreate",
        "TeamDelete",
        "Bash",
    )
    result = get_team_aware_tool_list(registry, team=None)
    assert [t.name for t in result] == ["Read", "TeamCreate", "Bash"]


def test_get_team_aware_tool_list_empty_registry() -> None:
    registry = ToolRegistry()
    result = get_team_aware_tool_list(registry, team=None)
    assert result == []


def test_get_team_aware_tool_list_returns_list_type() -> None:
    # The downstream QueryParams expects a list (it indexes + iterates).
    registry = _build_registry_with("Read")
    result = get_team_aware_tool_list(registry, team=None)
    assert isinstance(result, list)
