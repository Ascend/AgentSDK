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
#

"""Unit tests for extensions.tool_system_ext.team_filter.

Covers the pure filter function, the has_team_context predicate, and
the TEAM_ONLY_TOOL_NAMES constant. The filter keeps TeamCreate visible
as the team bootstrap entry while hiding tools that require an active
team context.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from extensions.tool_system_ext.team_filter import (
    TEAM_ONLY_TOOL_NAMES,
    filter_team_only_tools,
    has_team_context,
)


@dataclass
class _FakeTool:
    """Minimal stand-in for src.tool_system.build_tool.Tool.

    The real Tool dataclass has 30+ fields; we only need ``name`` for
    the filter under test. The filter reads via ``getattr(t, name_attr)``
    so any object with a ``name`` attribute works.
    """

    name: str


def test_team_only_tool_names_contains_send_message() -> None:
    assert "SendMessage" in TEAM_ONLY_TOOL_NAMES
    assert "TeamDelete" in TEAM_ONLY_TOOL_NAMES
    assert "TeamCreate" not in TEAM_ONLY_TOOL_NAMES


def test_team_only_tool_names_is_immutable() -> None:
    # frozenset — assignment must raise. Guards against accidental
    # mutation that would silently change the visibility rule.
    with pytest.raises((AttributeError, TypeError)):
        TEAM_ONLY_TOOL_NAMES.add("Read")  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "team,expected",
    [
        (None, False),
        ({}, False),
        ([], False),  # non-dict, even if truthy
        ("team-lead", False),  # non-dict string
        ({"team_name": "t"}, True),
        ({"team_name": "t", "members": []}, True),
    ],
)
def test_has_team_context(team: object, expected: bool) -> None:
    assert has_team_context(team) is expected


def test_filter_keeps_team_create_bootstrap_when_no_team() -> None:
    tools = [
        _FakeTool("Read"),
        _FakeTool("SendMessage"),
        _FakeTool("Bash"),
        _FakeTool("TeamCreate"),
        _FakeTool("Edit"),
        _FakeTool("TeamDelete"),
    ]
    result = filter_team_only_tools(tools, has_team=False)
    assert [t.name for t in result] == ["Read", "Bash", "TeamCreate", "Edit"]


def test_filter_passes_through_when_team_active() -> None:
    tools = [
        _FakeTool("Read"),
        _FakeTool("SendMessage"),
        _FakeTool("TeamCreate"),
        _FakeTool("TeamDelete"),
    ]
    result = filter_team_only_tools(tools, has_team=True)
    assert [t.name for t in result] == [
        "Read",
        "SendMessage",
        "TeamCreate",
        "TeamDelete",
    ]


def test_filter_preserves_order() -> None:
    # The registry returns tools in registration order. The filter
    # must not reorder the survivors — downstream code may depend on
    # stable ordering (e.g. prompt-cache key on tool schemas).
    tools = [
        _FakeTool("Bash"),
        _FakeTool("SendMessage"),
        _FakeTool("Read"),
        _FakeTool("TeamCreate"),
        _FakeTool("Edit"),
    ]
    result = filter_team_only_tools(tools, has_team=False)
    assert [t.name for t in result] == ["Bash", "Read", "TeamCreate", "Edit"]


def test_filter_handles_empty_list() -> None:
    assert filter_team_only_tools([], has_team=False) == []
    assert filter_team_only_tools([], has_team=True) == []


def test_filter_handles_list_with_no_team_only_tools() -> None:
    tools = [_FakeTool("Read"), _FakeTool("Bash"), _FakeTool("Edit")]
    assert filter_team_only_tools(tools, has_team=False) == tools


def test_filter_returns_new_list_not_in_place() -> None:
    # Defensive: callers expect a new list (the function's docstring
    # says so). Mutating the result must not affect the input.
    tools = [_FakeTool("Read"), _FakeTool("SendMessage")]
    result = filter_team_only_tools(tools, has_team=False)
    result.append(_FakeTool("Bash"))
    assert [t.name for t in tools] == ["Read", "SendMessage"]


def test_filter_with_custom_name_attr() -> None:
    # Defensive: a future Tool replacement may rename the attribute.
    # The filter accepts a custom ``name_attr`` to stay future-proof.
    @dataclass
    class _RenamedTool:
        display_name: str

    tools = [
        _RenamedTool("Read"),
        _RenamedTool("SendMessage"),
        _RenamedTool("TeamCreate"),
    ]
    result = filter_team_only_tools(
        tools,
        has_team=False,
        name_attr="display_name",
    )
    assert [t.display_name for t in result] == ["Read", "TeamCreate"]


def test_filter_with_object_missing_name_attr_keeps_it() -> None:
    # A tool-like object without a ``name`` attribute is kept (not
    # dropped) — ``getattr(..., name) not in TEAM_ONLY_TOOL_NAMES``
    # is True when the lookup returns the default. This matches
    # defensive tool-list assembly elsewhere in the codebase.
    class _Nameless:
        pass

    tools = [_Nameless(), _FakeTool("SendMessage")]
    result = filter_team_only_tools(tools, has_team=False)
    assert len(result) == 1
    assert isinstance(result[0], _Nameless)
