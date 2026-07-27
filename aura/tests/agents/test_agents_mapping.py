#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import sys
from pathlib import Path
from unittest.mock import MagicMock

_PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "aura" / "agents").exists()
)
_AURA_SRC = str(_PROJECT_ROOT / "aura")
if _AURA_SRC not in sys.path:
    sys.path.insert(0, _AURA_SRC)

_MODULES_TO_MOCK = [
    "rllm",
    "rllm.rewards",
    "rllm.rewards.reward_types",
    "torch",
    "agents.math_agent.environment.tool_env",
    "agents.math_agent.reward.reward_fn",
    "agents.math_agent.tool_agent",
    "agents.tools_mapping",
    "agents.search_r1_agent.prompt",
    "agents.search_r1_agent.reward.reward_fn",
    "agents.search_r1_agent.reward.search_r1_reward",
    "agents.search_r1_agent.search_r1_agent",
    "agents.search_r1_agent.environment.search_r1_env",
    "agents.search_r1_agent.parser.chat_template",
    "agents.proxy_agent.extern_agent",
    "agents.proxy_agent.environment.tool_env",
    "aura.runner.agent_engine_wrapper.base.environment.env_utils",
]

_original_modules = {}


def setup_module():
    for mod_name in _MODULES_TO_MOCK:
        _original_modules[mod_name] = sys.modules.get(mod_name)
        sys.modules[mod_name] = MagicMock()


def teardown_module():
    for mod_name in _MODULES_TO_MOCK:
        if mod_name in _original_modules:
            orig = _original_modules[mod_name]
            if orig is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = orig


class TestAgentsMapping:
    """Tests for agents_mapping module."""

    def test_agents_mapping_structure(self):
        """Test AGENTS_MAPPING has expected structure."""
        from agents.agents_mapping import AGENTS_MAPPING

        assert isinstance(AGENTS_MAPPING, list)
        assert len(AGENTS_MAPPING) > 0

    def test_agents_mapping_math_agent_exists(self):
        """Test that math agent exists in mapping."""
        from agents.agents_mapping import AGENTS_MAPPING

        math_agent = None
        for agent in AGENTS_MAPPING:
            if agent.get("name") == "math":
                math_agent = agent
                break

        assert math_agent is not None
        assert "env_class" in math_agent
        assert "agent_class" in math_agent
        assert "env_args" in math_agent
        assert "agent_args" in math_agent

    def test_agents_mapping_math_agent_config(self):
        """Test math agent configuration."""
        from agents.agents_mapping import AGENTS_MAPPING

        math_agent = None
        for agent in AGENTS_MAPPING:
            if agent.get("name") == "math":
                math_agent = agent
                break

        assert math_agent is not None
        env_args = math_agent.get("env_args", {})
        assert "tools" in env_args
        assert "reward_fn" in env_args

    def test_agents_mapping_webwalker_agent_config_is_lazy(self):
        """Test WebWalker agent registration without importing optional deps."""
        from agents.agents_mapping import AGENTS_MAPPING

        webwalker_agent = None
        for agent in AGENTS_MAPPING:
            if agent.get("name") == "webwalker":
                webwalker_agent = agent
                break

        assert webwalker_agent is not None
        assert webwalker_agent["env_class"].module_path == "agents.webwalker_agent.environment.webwalker_env"
        assert webwalker_agent["agent_class"].module_path == "agents.webwalker_agent.webwalker_agent"
        assert webwalker_agent["env_args"]["reward_fn"].object_name == "webwalker_reward_fn"
        assert webwalker_agent["agent_args"]["parser_name"] == "webwalker"

    def test_get_agent_by_name_found(self):
        """Test get_agent_by_name returns agent when found."""
        from agents.agents_mapping import get_agent_by_name

        result = get_agent_by_name("math")

        assert result is not None
        assert result.get("name") == "math"

    def test_get_agent_by_name_not_found(self):
        """Test get_agent_by_name returns None when not found."""
        from agents.agents_mapping import get_agent_by_name

        result = get_agent_by_name("nonexistent_agent")

        assert result is None

    def test_get_agent_by_name_empty_string(self):
        """Test get_agent_by_name with empty string."""
        from agents.agents_mapping import get_agent_by_name

        result = get_agent_by_name("")

        assert result is None

    def test_agents_mapping_has_compute_trajectory_reward_fn(self):
        """Test that math agent has compute_trajectory_reward_fn."""
        from agents.agents_mapping import AGENTS_MAPPING

        math_agent = None
        for agent in AGENTS_MAPPING:
            if agent.get("name") == "math":
                math_agent = agent
                break

        assert math_agent is not None
        assert "compute_trajectory_reward_fn" in math_agent
