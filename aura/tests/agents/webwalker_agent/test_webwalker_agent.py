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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def agent_module(monkeypatch):
    project_root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "aura" / "agents").exists()
    )
    aura_src = str(project_root / "aura")
    if aura_src not in sys.path:
        sys.path.insert(0, aura_src)

    base_agent = types.ModuleType("aura.runner.agent_engine_wrapper.base.agent.base_agent")
    base_agent.BaseAgent = object
    base_agent.Action = dict
    base_agent.Step = dict

    class Trajectory:
        pass

    base_agent.Trajectory = Trajectory
    monkeypatch.setitem(sys.modules, "aura.runner.agent_engine_wrapper.base.agent.base_agent", base_agent)

    multi_tool = types.ModuleType("rllm.tools.multi_tool")
    multi_tool.MultiTool = MagicMock
    monkeypatch.setitem(sys.modules, "rllm.tools.multi_tool", multi_tool)

    parser_base = types.ModuleType("rllm.parser.tool_parser.tool_parser_base")
    parser_base.ToolParser = object
    monkeypatch.setitem(sys.modules, "rllm", types.ModuleType("rllm"))
    monkeypatch.setitem(sys.modules, "rllm.tools", types.ModuleType("rllm.tools"))
    monkeypatch.setitem(sys.modules, "rllm.parser", types.ModuleType("rllm.parser"))
    monkeypatch.setitem(sys.modules, "rllm.parser.tool_parser", types.ModuleType("rllm.parser.tool_parser"))
    monkeypatch.setitem(sys.modules, "rllm.parser.tool_parser.tool_parser_base", parser_base)

    sys.modules.pop("agents.webwalker_agent.webwalker_agent", None)
    return importlib.import_module("agents.webwalker_agent.webwalker_agent")


def test_system_prompt_contains_visit_page_tool(agent_module):
    assert "visit_page" in agent_module.SYSTEM_EXPLORER
    assert "Action Input" in agent_module.SYSTEM_EXPLORER


def test_compact_observation_prefers_critic_information(agent_module):
    agent = object.__new__(agent_module.WebWalkerAgent)
    agent.max_prompt_length = 256
    agent.tokenizer = None
    agent.memory = SimpleNamespace(tokenizer=None)

    compact = agent._build_compact_observation(
        "long page body\n\nclickable button:\n\n<button>Products</button>\n\nEach button is wrapped in a <button> tag",
        {
            "critic_useful_information": ["Fact A", "Fact A", "Fact B"],
            "webwalker_memory_snapshot": ["Old", "New"],
        },
        field_name="observation",
    )

    assert "Fact A" in compact
    assert compact.count("Fact A") == 1
    assert "accumulated useful information" in compact
    assert "<button>Products</button>" in compact


def test_format_initial_observation_as_user_message(agent_module):
    agent = object.__new__(agent_module.WebWalkerAgent)
    agent.max_prompt_length = 256
    agent.tokenizer = None
    agent.memory = SimpleNamespace(tokenizer=None)

    messages = agent._format_observation_as_messages(
        {"question": "Where is pricing?", "root_url": "https://example.com", "initial_observation": "home"},
        {"metadata": {}},
    )

    assert messages == [
        {
            "role": "user",
            "content": "Question: Where is pricing?\nURL: https://example.com\n\nObservation: home",
        }
    ]
