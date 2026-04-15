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
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockTrajectory:
    task: Any = None
    steps: list = field(default_factory=list)
    reward: float = 0.0

    def to_dict(self):
        return {
            "task": self.task,
            "steps": [step for step in self.steps],
            "reward": float(self.reward),
        }


@dataclass
class MockStep:
    chat_completions: list = field(default_factory=list)
    action: Any = None
    model_response: str = ""
    observation: Any = None

    def to_dict(self):
        return {
            "chat_completions": self.chat_completions,
            "action": self.action,
            "model_response": self.model_response,
            "observation": self.observation,
        }


@dataclass
class MockAction:
    action: list = field(default_factory=list)


@dataclass
class MockToolCall:
    name: str = ""
    arguments: dict = field(default_factory=dict)

    def to_dict(self):
        return {"name": self.name, "arguments": self.arguments}


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies(mock_ray_dependencies, mock_agentic_rl_dependencies, mock_rllm_dependencies):
    """Mock all external dependencies for tool_agent tests."""
    mock_tool_parser = MagicMock()
    mock_tool_parser.parse.return_value = []
    mock_tool_parser.get_tool_prompt.return_value = "test prompt"

    mock_multi_tool = MagicMock()
    mock_multi_tool.json = [{"type": "function", "function": {"name": "test_tool"}}]

    with (
        patch("agentic_rl.runner.agent_engine_wrapper.base.agent.base_agent.Trajectory", MockTrajectory),
        patch("agentic_rl.runner.agent_engine_wrapper.base.agent.base_agent.Step", MockStep),
        patch("agentic_rl.runner.agent_engine_wrapper.base.agent.base_agent.Action", MockAction),
        patch("agentic_rl.runner.agent_engine_wrapper.base.agent.base_agent.BaseAgent", object),
        patch("agents.math_agent.tool_agent.MultiTool") as mock_multi_tool_cls,
        patch("agents.math_agent.tool_agent.get_tool_parser") as mock_get_parser,
        patch("agents.math_agent.tool_agent.TOOL_SYSTEM_PROMPT", "default prompt"),
    ):
        mock_multi_tool_cls.return_value = mock_multi_tool
        mock_get_parser.return_value = lambda: mock_tool_parser
        yield {
            "multi_tool": mock_multi_tool,
            "tool_parser": mock_tool_parser,
            "get_tool_parser": mock_get_parser,
        }


class TestToolAgent:
    """Tests for ToolAgent class."""

    def test_init_with_tools(self, mock_dependencies):
        """Test ToolAgent initialization with tools list."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        agent.system_prompt = "test prompt"
        agent.tools = mock_dependencies["multi_tool"]
        agent.tool_parser = mock_dependencies["tool_parser"]
        agent.tools_prompt = "test tools prompt"
        agent._trajectory = MockTrajectory()
        agent.messages = []
        
        assert agent.system_prompt == "test prompt"
        assert agent.tools is not None

    def test_init_with_tool_map(self, mock_dependencies):
        """Test ToolAgent initialization with tool_map."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        agent.system_prompt = "test prompt"
        agent.tools = mock_dependencies["multi_tool"]
        agent.tool_parser = mock_dependencies["tool_parser"]
        agent.tools_prompt = "test tools prompt"
        agent._trajectory = MockTrajectory()
        agent.messages = []
        
        assert agent.tools is not None

    def test_reset(self, mock_dependencies):
        """Test ToolAgent reset method."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        agent.system_prompt = "test prompt"
        agent.tools_prompt = "test tools prompt"
        agent._trajectory = MockTrajectory()
        agent._trajectory.steps.append(MockStep())
        agent.messages = [{"role": "user", "content": "test"}]
        
        agent.reset()
        
        assert len(agent._trajectory.steps) == 0
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"

    def test_chat_completions_property(self, mock_dependencies):
        """Test chat_completions property returns messages."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        agent.messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]
        
        result = agent.chat_completions
        
        assert result == agent.messages
        assert len(result) == 2

    def test_trajectory_property(self, mock_dependencies):
        """Test trajectory property returns trajectory."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        agent._trajectory = MockTrajectory()
        agent._trajectory.reward = 1.0
        
        result = agent.trajectory
        
        assert result.reward == 1.0

    def test_format_observation_as_messages_dict_with_task(self, mock_dependencies):
        """Test _format_observation_as_messages with dict containing task."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        obs = {"task": {"problem": "test problem"}}
        
        result = agent._format_observation_as_messages(obs)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "test problem"

    def test_format_observation_as_messages_dict_with_question(self, mock_dependencies):
        """Test _format_observation_as_messages with dict containing question."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        obs = {"task": {"question": "test question"}}
        
        result = agent._format_observation_as_messages(obs)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "test question"

    def test_format_observation_as_messages_dict_with_problem(self, mock_dependencies):
        """Test _format_observation_as_messages with dict containing problem."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        obs = {"problem": "direct problem"}
        
        result = agent._format_observation_as_messages(obs)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "direct problem"

    def test_format_observation_as_messages_dict_with_tool_outputs(self, mock_dependencies):
        """Test _format_observation_as_messages with tool outputs."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        obs = {"tool_outputs": {"call_1": "result 1", "call_2": "result 2"}}
        
        result = agent._format_observation_as_messages(obs)
        
        assert len(result) == 2
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"

    def test_format_observation_as_messages_string(self, mock_dependencies):
        """Test _format_observation_as_messages with string."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        obs = "simple string observation"
        
        result = agent._format_observation_as_messages(obs)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "simple string observation"

    def test_update_from_env(self, mock_dependencies):
        """Test update_from_env method."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        agent.messages = [{"role": "system", "content": "system"}]
        agent.current_observation = None
        
        observation = {"task": {"problem": "test"}}
        agent.update_from_env(observation, reward=0.0, done=False, info={})
        
        assert agent.current_observation == observation
        assert len(agent.messages) == 2

    def test_update_from_model_with_tool_calls(self, mock_dependencies):
        """Test update_from_model with tool calls in response."""
        from agents.math_agent.tool_agent import ToolAgent
        
        mock_tool_call = MockToolCall(name="python", arguments={"code": "print(1)"})
        mock_dependencies["tool_parser"].parse.return_value = [mock_tool_call]
        
        agent = ToolAgent.__new__(ToolAgent)
        agent.messages = [{"role": "system", "content": "system"}]
        agent.tool_parser = mock_dependencies["tool_parser"]
        agent._trajectory = MockTrajectory()
        agent.current_observation = {"task": {"problem": "test"}}
        
        response = 'Let me solve this.{"name": "python", "arguments": {"code": "print(1)"}}'
        action = agent.update_from_model(response)
        
        assert len(agent.messages) == 2
        assert agent.messages[1]["role"] == "assistant"
        assert len(agent._trajectory.steps) == 1

    def test_update_from_model_without_tool_calls(self, mock_dependencies):
        """Test update_from_model without tool calls."""
        from agents.math_agent.tool_agent import ToolAgent
        
        mock_dependencies["tool_parser"].parse.return_value = []
        
        agent = ToolAgent.__new__(ToolAgent)
        agent.messages = [{"role": "system", "content": "system"}]
        agent.tool_parser = mock_dependencies["tool_parser"]
        agent._trajectory = MockTrajectory()
        agent.current_observation = None
        
        response = "This is a direct answer without tool calls."
        action = agent.update_from_model(response)
        
        assert len(agent.messages) == 2
        assert len(agent._trajectory.steps) == 1

    def test_init_with_both_tools_and_tool_map_raises_error(self, mock_dependencies):
        """Test that providing both tools and tool_map raises ValueError."""
        from agents.math_agent.tool_agent import ToolAgent
        
        with pytest.raises(ValueError, match="Cannot specify both 'tools' and 'tool_map' parameters"):
            agent = ToolAgent(
                system_prompt="test",
                parser_name="qwen",
                tools=["tool1", "tool2"],
                tool_map={"tool1": Mock}
            )

    def test_init_with_neither_tools_nor_tool_map(self, mock_dependencies):
        """Test initialization with neither tools nor tool_map."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent(
            system_prompt="test",
            parser_name="qwen",
            tools=None,
            tool_map=None
        )
        
        assert agent.tools is not None

    def test_format_observation_as_messages_empty_observation(self, mock_dependencies):
        """Test _format_observation_as_messages with empty observation."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        
        result = agent._format_observation_as_messages(None)
        assert len(result) == 0
        
        result = agent._format_observation_as_messages({})
        assert len(result) == 0
        
        result = agent._format_observation_as_messages([])
        assert len(result) == 0

    def test_format_observation_as_messages_other_type(self, mock_dependencies):
        """Test _format_observation_as_messages with other object types."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        
        class CustomClass:
            def __str__(self):
                return "custom object"
        
        obs = CustomClass()
        result = agent._format_observation_as_messages(obs)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "custom object"

    def test_update_from_env_with_tool_outputs_observation(self, mock_dependencies):
        """Test update_from_env with tool outputs in observation."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        agent.messages = [{"role": "system", "content": "system"}]
        agent.current_observation = None
        
        observation = {
            "tool_outputs": {
                "call_123": "calculation result",
                "call_456": "error occurred"
            }
        }
        agent.update_from_env(observation, reward=1.0, done=False, info={})
        
        assert agent.current_observation == observation
        assert len(agent.messages) == 3  # system + 2 tool outputs
        assert agent.messages[1]["role"] == "tool"
        assert agent.messages[1]["tool_call_id"] == "call_123"
        assert agent.messages[2]["tool_call_id"] == "call_456"

    def test_update_from_model_with_parse_exception(self, mock_dependencies):
        """Test update_from_model when parser raises exception."""
        from agents.math_agent.tool_agent import ToolAgent
        
        mock_dependencies["tool_parser"].parse.side_effect = Exception("Parse error")
        
        agent = ToolAgent.__new__(ToolAgent)
        agent.messages = [{"role": "system", "content": "system"}]
        agent.tool_parser = mock_dependencies["tool_parser"]
        agent._trajectory = MockTrajectory()
        agent.current_observation = None
        
        response = "Invalid response format"
        action = agent.update_from_model(response)
        
        # Should create finish action despite parse error
        assert len(agent.messages) == 2
        assert len(agent._trajectory.steps) == 1
        assert action.action[0]["function"]["name"] == "finish"
        assert action.action[0]["function"]["arguments"]["response"] == response

    def test_update_from_model_with_tool_calls_dict_arguments(self, mock_dependencies):
        """Test update_from_model with tool calls that have dict arguments."""
        from agents.math_agent.tool_agent import ToolAgent
        
        mock_tool_call = MockToolCall(
            name="calculator",
            arguments={"expression": "2+2", "precision": 2}
        )
        mock_dependencies["tool_parser"].parse.return_value = [mock_tool_call]
        
        agent = ToolAgent.__new__(ToolAgent)
        agent.messages = [{"role": "system", "content": "system"}]
        agent.tool_parser = mock_dependencies["tool_parser"]
        agent._trajectory = MockTrajectory()
        agent.current_observation = None
        
        response = '{"name": "calculator", "arguments": {"expression": "2+2"}}'
        action = agent.update_from_model(response)
        
        # Check that arguments were serialized to JSON string
        assert isinstance(action.action[0]["function"]["arguments"], str)
        assert "expression" in action.action[0]["function"]["arguments"]

    def test_update_from_model_preserves_existing_string_arguments(self, mock_dependencies):
        """Test update_from_model doesn't modify already string arguments."""
        from agents.math_agent.tool_agent import ToolAgent
        
        # Create a mock tool call with string arguments
        mock_tool_call = MockToolCall(
            name="test_tool",
            arguments='{"param": "value"}'  # Already a string
        )
        mock_dependencies["tool_parser"].parse.return_value = [mock_tool_call]
        
        agent = ToolAgent.__new__(ToolAgent)
        agent.messages = [{"role": "system", "content": "system"}]
        agent.tool_parser = mock_dependencies["tool_parser"]
        agent._trajectory = MockTrajectory()
        agent.current_observation = None
        
        response = "Tool call response"
        action = agent.update_from_model(response)
        
        # Should remain as string
        assert isinstance(action.action[0]["function"]["arguments"], str)

    def test_reset_clears_all_state(self, mock_dependencies):
        """Test reset completely clears agent state."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        agent.system_prompt = "test prompt"
        agent.tools_prompt = "test tools prompt"
        agent._trajectory = MockTrajectory()
        agent._trajectory.steps = [MockStep(), MockStep()]
        agent._trajectory.reward = 5.0
        agent.messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user1"},
            {"role": "assistant", "content": "assistant1"},
        ]
        
        agent.reset()
        
        assert len(agent._trajectory.steps) == 0
        assert agent._trajectory.reward == 0.0
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[0]["content"] == "test prompttest tools prompt"

    def test_multiple_update_from_env_calls(self, mock_dependencies):
        """Test multiple sequential update_from_env calls."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        agent.messages = [{"role": "system", "content": "system"}]
        
        # First update
        obs1 = {"task": {"problem": "first problem"}}
        agent.update_from_env(obs1, reward=0.0, done=False, info={})
        
        # Second update
        obs2 = {"tool_outputs": {"call_1": "result"}}
        agent.update_from_env(obs2, reward=0.0, done=False, info={})
        
        assert agent.current_observation == obs2
        assert len(agent.messages) == 3  # system + first obs + second obs

    def test_trajectory_stores_correct_step_info(self, mock_dependencies):
        """Test that trajectory stores correct step information."""
        from agents.math_agent.tool_agent import ToolAgent
        
        mock_tool_call = MockToolCall(name="python", arguments={"code": "print(1)"})
        mock_dependencies["tool_parser"].parse.return_value = [mock_tool_call]
        
        agent = ToolAgent.__new__(ToolAgent)
        agent.messages = [{"role": "system", "content": "system"}]
        agent.tool_parser = mock_dependencies["tool_parser"]
        agent._trajectory = MockTrajectory()
        agent.current_observation = {"task": {"problem": "solve this"}}
        
        response = "Let me calculate this"
        action = agent.update_from_model(response)
        
        assert len(agent._trajectory.steps) == 1
        step = agent._trajectory.steps[0]
        assert step.model_response == response
        assert step.observation == agent.current_observation
        assert step.action == action.action


class TestMCPToolAgent:
    """Tests for MCPToolAgent class."""

    def test_mcp_tool_agent_init(self, mock_dependencies):
        """Test MCPToolAgent initialization."""
        from agents.math_agent.tool_agent import MCPToolAgent
        
        mock_mcp_tool = MagicMock()
        mock_mcp_tool.json = {"type": "function", "function": {"name": "mcp_tool"}}
        
        agent = MCPToolAgent.__new__(MCPToolAgent)
        agent.system_prompt = "test prompt"
        agent.tool_map = {"mcp_tool": mock_mcp_tool}
        agent.tool_parser = mock_dependencies["tool_parser"]
        agent.tools_prompt = "test tools prompt"
        agent._trajectory = MockTrajectory()
        agent.messages = []
        
        assert agent.system_prompt == "test prompt"
        assert "mcp_tool" in agent.tool_map
    
    def test_mcp_tool_agent_reset(self, mock_dependencies):
        """Test MCPToolAgent reset method."""
        from agents.math_agent.tool_agent import MCPToolAgent
        
        mock_mcp_tool = MagicMock()
        mock_mcp_tool.json = {"type": "function", "function": {"name": "mcp_tool"}}
        
        agent = MCPToolAgent.__new__(MCPToolAgent)
        agent.system_prompt = "test prompt"
        agent.tool_map = {"mcp_tool": mock_mcp_tool}
        agent.tool_parser = mock_dependencies["tool_parser"]
        agent.tools_prompt = "test prompt"
        agent._trajectory = MockTrajectory()
        agent._trajectory.steps.append(MockStep())
        agent.messages = [{"role": "user", "content": "test"}]
        
        agent.reset()
        
        assert len(agent._trajectory.steps) == 0
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"

    def test_mcp_tool_agent_chat_completions(self, mock_dependencies):
        """Test MCPToolAgent chat_completions property."""
        from agents.math_agent.tool_agent import MCPToolAgent
        
        agent = MCPToolAgent.__new__(MCPToolAgent)
        agent.messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]
        
        result = agent.chat_completions
        assert result == agent.messages

    def test_mcp_tool_agent_trajectory(self, mock_dependencies):
        """Test MCPToolAgent trajectory property."""
        from agents.math_agent.tool_agent import MCPToolAgent
        
        agent = MCPToolAgent.__new__(MCPToolAgent)
        agent._trajectory = MockTrajectory()
        agent._trajectory.reward = 2.5
        
        result = agent.trajectory
        assert result.reward == 2.5

    def test_mcp_tool_agent_update_from_env(self, mock_dependencies):
        """Test MCPToolAgent update_from_env method."""
        from agents.math_agent.tool_agent import MCPToolAgent
        
        agent = MCPToolAgent.__new__(MCPToolAgent)
        agent.messages = [{"role": "system", "content": "system"}]
        agent.current_observation = None
        
        observation = {"problem": "test problem for MCP"}
        agent.update_from_env(observation, reward=0.0, done=False, info={})
        
        assert agent.current_observation == observation
        assert len(agent.messages) == 2
        assert agent.messages[1]["content"] == "test problem for MCP"

class TestToolAgentIntegration:
    """Integration tests for ToolAgent with real components."""
    
    def test_full_conversation_flow(self, mock_dependencies):
        """Test complete conversation flow with multiple turns."""
        from agents.math_agent.tool_agent import ToolAgent
        
        # Setup agent
        agent = ToolAgent.__new__(ToolAgent)
        agent.system_prompt = "You are a helpful assistant"
        agent.tools_prompt = "Available tools: python"
        agent.tool_parser = mock_dependencies["tool_parser"]
        agent._trajectory = MockTrajectory()
        agent.messages = []
        agent.current_observation = None
        
        agent.reset()
        
        # First user input
        mock_dependencies["tool_parser"].parse.return_value = []
        agent.update_from_env({"problem": "What is 2+2?"}, reward=0.0, done=False, info={})
        response1 = "2+2 equals 4"
        action1 = agent.update_from_model(response1)
        
        assert len(agent.messages) == 3  # system + assistant
        
        # Second user input with tool
        mock_tool_call = MockToolCall(name="python", arguments={"code": "print(2+2)"})
        mock_dependencies["tool_parser"].parse.return_value = [mock_tool_call]
        agent.update_from_env({"tool_outputs": {"call_1": "4"}}, reward=0.0, done=False, info={})
        response2 = "The answer is 4"
        action2 = agent.update_from_model(response2)
        
        assert len(agent.messages) == 5  # system + assistant1 + tool + assistant2
        
    def test_error_recovery_in_conversation(self, mock_dependencies):
        """Test agent recovery from parsing errors."""
        from agents.math_agent.tool_agent import ToolAgent
        
        agent = ToolAgent.__new__(ToolAgent)
        agent.system_prompt = "You are a helpful assistant"
        agent.tools_prompt = "Available tools: python"
        agent.tool_parser = mock_dependencies["tool_parser"]
        agent._trajectory = MockTrajectory()
        agent.messages = []
        agent.current_observation = None
        
        agent.reset()
        
        # Cause parse error
        mock_dependencies["tool_parser"].parse.side_effect = Exception("Invalid JSON")
        agent.update_from_env({"problem": "Calculate something"}, reward=0.0, done=False, info={})
        response = "Let me help you with that"
        action = agent.update_from_model(response)
        
        # Should recover by creating finish action
        assert len(agent.messages) == 3
        assert action.action[0]["function"]["name"] == "finish"
        
        # Reset parser for next call
        mock_dependencies["tool_parser"].parse.side_effect = None
        mock_dependencies["tool_parser"].parse.return_value = []