#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

        http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import importlib
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture()
def parser_module(monkeypatch):
    project_root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "aura" / "agents").exists()
    )
    aura_src = str(project_root / "aura")
    if aura_src not in sys.path:
        sys.path.insert(0, aura_src)

    base_mod = types.ModuleType("rllm.parser.tool_parser.tool_parser_base")

    class ToolParser:
        pass

    base_mod.ToolParser = ToolParser
    monkeypatch.setitem(sys.modules, "rllm", types.ModuleType("rllm"))
    monkeypatch.setitem(sys.modules, "rllm.parser", types.ModuleType("rllm.parser"))
    monkeypatch.setitem(sys.modules, "rllm.parser.tool_parser", types.ModuleType("rllm.parser.tool_parser"))
    monkeypatch.setitem(sys.modules, "rllm.parser.tool_parser.tool_parser_base", base_mod)

    sys.modules.pop("agents.webwalker_agent.parser.tool_parser.webwalker_tool_parser", None)
    return importlib.import_module("agents.webwalker_agent.parser.tool_parser.webwalker_tool_parser")


def test_parse_direct_json_visit_page(parser_module):
    parser = parser_module.WebWalkerToolParser()

    assert parser.parse('{"action": "visit_page", "button": "About"}') == [
        {"name": "visit_page", "arguments": {"button": "About"}}
    ]


def test_parse_last_react_call_and_ignore_think_blocks(parser_module):
    parser = parser_module.WebWalkerToolParser()
    text = """
<think>
Action: visit_page
Action Input: {"button": "Wrong"}
</think>
Thought: open the useful page
Action: visit_page
Action Input: {"button": "Products"}
Observation: page opened
"""

    assert parser.parse(text) == [{"name": "visit_page", "arguments": {"button": "Products"}}]


def test_parse_finish_tool_is_accepted(parser_module):
    parser = parser_module.WebWalkerToolParser()

    result = parser.parse('{"action": "finish", "parameters": {"response": "done"}}')

    assert result == [
        {"name": "finish", "arguments": {"response": "done"}}
    ]


def test_parse_invalid_tool_returns_error_tool(parser_module):
    parser = parser_module.WebWalkerToolParser()

    result = parser.parse('{"action": "unknown", "parameters": {"response": "done"}}')

    assert result == [
        {"name": "error_tool", "arguments": {"response": parser_module.WebWalkerKeyword.PARSE_TOOL_ERROR.value}}
    ]


@pytest.mark.parametrize(
    "text, expected_name",
    [
        ('```json\n{"action": "finish", "parameters": {"response": "x"}}\n```', "finish"),
        (
            'Action: visit_page\nAction Input: {"button":"A"}\nObservation: ...\n'
            'Action: finish\nAction Input: {"response":"done"}',
            "finish",
        ),
        ('Action: visit_page\nAction Input: ```json\n{"button":"A"}\n```', "visit_page"),
        ("", "error_tool"),
        ("<think>Action: visit_page\nAction Input: {}</think>", "error_tool"),
        ("Action: visit_page\nAction Input: not-json", "error_tool"),
        ('{"action": "finish", "parameters": {"response": "x"}} trailing', "error_tool"),
    ],
)
def test_parse_edge_cases(parser_module, text, expected_name):
    parser = parser_module.WebWalkerToolParser()

    result = parser.parse(text)

    assert result[0]["name"] == expected_name


def test_parse_unexpected_input_type_raises(parser_module):
    parser = parser_module.WebWalkerToolParser()

    with pytest.raises(TypeError):
        parser.parse(None)


def test_explorer_tools_include_finish(parser_module):
    tools_module = importlib.import_module("agents.webwalker_agent.webwalker_tools")

    assert "finish" in tools_module.webwalker_explorer_tools
