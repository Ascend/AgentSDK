#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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

import sys
from abc import ABC
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest


# Create proper base classes that ProxyAgent can inherit from
@dataclass
class MockStep:
    chat_completions: list = field(default_factory=list)
    thought: str = ""
    action: object = None
    observation: object = None
    model_response: str = ""
    info: dict = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    mc_return: float = 0.0
    step_id: int = 0
    prompt_ids: list = field(default_factory=list)
    response_ids: list = field(default_factory=list)
    response_masks: object = None
    logprobs: list = field(default_factory=list)

    def to_dict(self):
        return {
            "chat_completions": self.chat_completions,
            "reward": self.reward,
            "mc_return": float(self.mc_return),
            "done": self.done,
            "step_id": self.step_id,
        }


@dataclass
class MockAction:
    action: object = None


@dataclass
class MockTrajectory:
    task: object = None
    steps: list = field(default_factory=list)
    reward: float = 0.0
    toolcall_reward: float = 0.0
    res_reward: float = 0.0
    prompt_id: int = 0
    data_id: str = None
    training_id: str = None
    epoch_id: int = 0
    iteration_id: int = 0
    sample_id: int = 0
    trajectory_id: int = 0
    application_id: str = ""
    termination_reason: str = "unknown"

    def to_dict(self):
        return {
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
            "reward": float(self.reward),
        }


class MockBaseAgent(ABC):
    def __init__(self, *args, **kwargs):
        self._trajectory = MockTrajectory()
        self.messages = []
        self.current_observation = None
        self.system_prompt = ""

    def reset(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    @property
    def chat_completions(self):
        return self.messages

    @property
    def trajectory(self):
        return self._trajectory

    def update_from_env(self, observation, reward, done, info):
        pass

    def update_from_model(self, response):
        pass

    def update_system_prompt(self, new_prompt):
        self.system_prompt = new_prompt
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = new_prompt


class TestExternAgent:

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        # Mock base_agent module with proper classes
        mock_base_agent = MagicMock()
        mock_base_agent.BaseAgent = MockBaseAgent
        mock_base_agent.Step = MockStep
        mock_base_agent.Trajectory = MockTrajectory
        mock_base_agent.Action = MockAction

        # Mock math_agent modules
        mock_system_prompts = MagicMock()
        mock_system_prompts.TOOL_SYSTEM_PROMPT = "You are a helpful assistant"

        mock_parser = MagicMock()
        mock_parser.get_tool_parser = MagicMock(return_value=MagicMock())

        mock_multi_tool = MagicMock()
        mock_multi_tool_instance = MagicMock()
        mock_multi_tool_instance.json = [{"name": "test_tool", "description": "A test tool"}]
        mock_multi_tool.MultiTool = MagicMock(return_value=mock_multi_tool_instance)

        mock_tool_base = MagicMock()
        mock_tool_base.Tool = object

        with patch.dict(sys.modules, {
            "aura.runner.agent_engine_wrapper.base.agent.base_agent": mock_base_agent,
            "agents.math_agent.prompt.system_prompts": mock_system_prompts,
            "agents.math_agent.parser": mock_parser,
            "agents.math_agent.environment.tools.multi_tool": mock_multi_tool,
            "agents.math_agent.environment.tools.tool_base": mock_tool_base,
        }):
            yield

    def test_proxy_agent_init_with_tools(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        agent = ProxyAgent(tools=["tool1", "tool2"])
        assert agent.system_prompt is not None
        assert agent.tools is not None
        assert agent.messages is not None

    def test_proxy_agent_init_with_tool_map(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        agent = ProxyAgent(tool_map={"tool1": MagicMock()})
        assert agent.tools is not None

    def test_proxy_agent_init_tools_and_tool_map_error(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        with pytest.raises(ValueError, match="Cannot specify both"):
            ProxyAgent(tools=["tool1"], tool_map={"tool1": MagicMock()})

    def test_proxy_agent_reset(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        agent = ProxyAgent(tools=["tool1"])
        agent.reset()
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"

    def test_proxy_agent_update_from_env_with_task(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        agent = ProxyAgent(tools=["tool1"])
        agent.reset()
        initial_count = len(agent.messages)

        observation = {"task": {"problem": "What is 2+2?"}}
        agent.update_from_env(observation, reward=0.0, done=False, info={})

        assert len(agent.messages) > initial_count
        assert agent.messages[-1]["role"] == "user"

    def test_proxy_agent_update_from_env_with_question(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        agent = ProxyAgent(tools=["tool1"])
        agent.reset()
        initial_count = len(agent.messages)

        observation = {"task": {"question": "What is 2+2?"}}
        agent.update_from_env(observation, reward=0.0, done=False, info={})

        assert len(agent.messages) > initial_count

    def test_proxy_agent_update_from_env_with_problem(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        agent = ProxyAgent(tools=["tool1"])
        agent.reset()
        initial_count = len(agent.messages)

        observation = {"problem": "What is 2+2?"}
        agent.update_from_env(observation, reward=0.0, done=False, info={})

        assert len(agent.messages) > initial_count

    def test_proxy_agent_update_from_env_with_tool_outputs(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        agent = ProxyAgent(tools=["tool1"])
        agent.reset()
        initial_count = len(agent.messages)

        observation = {"tool_outputs": {"call_1": "result1", "call_2": "result2"}}
        agent.update_from_env(observation, reward=0.0, done=False, info={})

        assert len(agent.messages) > initial_count
        assert any(m["role"] == "tool" for m in agent.messages)

    def test_proxy_agent_update_from_env_with_string(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        agent = ProxyAgent(tools=["tool1"])
        agent.reset()
        initial_count = len(agent.messages)

        observation = "hello"
        agent.update_from_env(observation, reward=0.0, done=False, info={})

        assert len(agent.messages) > initial_count

    def test_proxy_agent_update_from_model_string(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        agent = ProxyAgent(tools=["tool1"])
        agent.reset()

        action = agent.update_from_model("response text")
        assert action is not None
        assert len(agent.messages) > 0

    def test_proxy_agent_update_from_model_dict(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        agent = ProxyAgent(tools=["tool1"])
        agent.reset()

        action = agent.update_from_model(
            {"content": "hello", "tool_calls": []}
        )
        assert action is not None

    def test_proxy_agent_update_from_model_with_tool_calls(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        agent = ProxyAgent(tools=["tool1"])
        agent.reset()

        tool_call = {
            "id": "1",
            "type": "function",
            "function": {"name": "calc", "arguments": {"expr": "2+2"}},
        }
        action = agent.update_from_model(
            {"content": "let me calculate", "tool_calls": [tool_call]}
        )
        assert action is not None

    def test_proxy_agent_update_system_prompt(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        agent = ProxyAgent(tools=["tool1"])
        agent.reset()

        new_prompt = "New system prompt"
        agent.update_system_prompt(new_prompt)
        assert agent.system_prompt == new_prompt
        assert agent.messages[0]["content"] == new_prompt

    def test_proxy_agent_chat_completions(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        agent = ProxyAgent(tools=["tool1"])
        agent.reset()

        completions = agent.chat_completions
        assert isinstance(completions, list)
        assert len(completions) > 0

    def test_proxy_agent_trajectory(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        agent = ProxyAgent(tools=["tool1"])
        agent.reset()

        traj = agent.trajectory
        assert traj is not None

    def test_proxy_agent_init_no_tools(self):
        from agents.proxy_agent.extern_agent import ProxyAgent

        agent = ProxyAgent()
        assert agent.tools is not None
        assert agent.tools_prompt is not None
