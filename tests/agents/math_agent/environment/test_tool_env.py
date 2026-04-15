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


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies(mock_ray_dependencies, mock_agentic_rl_dependencies, mock_rllm_dependencies):
    """Mock all external dependencies for tool_env tests."""
    mock_multi_tool_instance = MagicMock()
    mock_multi_tool_instance.json = [{"type": "function", "function": {"name": "python"}}]
    mock_multi_tool_instance.return_value = MagicMock(to_string=lambda: "tool output")
    mock_multi_tool_instance.__call__ = MagicMock(return_value=MagicMock(to_string=lambda: "result"))
    
    mock_reward_output = MagicMock()
    mock_reward_output.reward = 1.0
    mock_reward_output.metadata = {}
    
    mock_reward_fn = MagicMock(return_value=mock_reward_output)
    
    with (
        patch("rllm.tools.multi_tool.MultiTool", return_value=mock_multi_tool_instance),
        patch("rllm.tools.tool_base.Tool", MagicMock()),
        patch("agents.math_agent.reward.reward_fn", return_value=mock_reward_output),
    ):
        yield {
            "multi_tool": mock_multi_tool_instance,
            "reward_fn": mock_reward_fn,
            "reward_output": mock_reward_output,
        }


class TestToolEnvironment:
    """Tests for ToolEnvironment class."""

    def test_init_with_tools(self, mock_dependencies):
        """Test ToolEnvironment initialization with tools."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        env = ToolEnvironment(tools=["python"])
        
        assert env.step_count == 0
        assert env.max_steps == 10

    def test_init_with_tool_map(self, mock_dependencies):
        """Test ToolEnvironment initialization with tool_map."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        mock_tool_class = MagicMock()
        env = ToolEnvironment(tool_map={"custom_tool": mock_tool_class})
        
        assert env.step_count == 0

    def test_init_with_task(self, mock_dependencies):
        """Test ToolEnvironment initialization with task."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        task = {"question": "What is 2+2?"}
        env = ToolEnvironment(task=task, tools=["python"])
        
        assert env.task == task

    def test_init_with_custom_max_steps(self, mock_dependencies):
        """Test ToolEnvironment initialization with custom max_steps."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        env = ToolEnvironment(tools=["python"], max_steps=5)
        
        assert env.max_steps == 5

    def test_init_raises_error_when_both_tools_and_tool_map(self, mock_dependencies):
        """Test ToolEnvironment raises error when both tools and tool_map provided."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        mock_tool_class = MagicMock()
        
        with pytest.raises(ValueError, match="Cannot specify both"):
            ToolEnvironment(tools=["python"], tool_map={"custom": mock_tool_class})

    def test_reset(self, mock_dependencies):
        """Test ToolEnvironment reset method."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        task = {"question": "test question"}
        env = ToolEnvironment(task=task, tools=["python"])
        env.step_count = 5
        
        result = env.reset()
        
        assert env.step_count == 0
        assert result == (task, {})

    def test_step_with_finish_action(self, mock_dependencies):
        """Test ToolEnvironment step with finish action."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        task = {"question": "test"}
        env = ToolEnvironment(task=task, tools=["python"], reward_fn=mock_dependencies["reward_fn"])
        
        action = [{"id": "1", "function": {"name": "finish", "arguments": {"response": "final answer"}}}]
        obs, reward, done, info = env.step(action)
        
        assert done is True
        assert reward == 1.0

    def test_step_with_tool_call(self, mock_dependencies):
        """Test ToolEnvironment step with tool call."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        task = {"question": "test"}
        env = ToolEnvironment(task=task, tools=["python"])
        
        action = [{"id": "1", "function": {"name": "python", "arguments": {"code": "print(1)"}}}]
        obs, reward, done, info = env.step(action)
        
        assert done is False
        assert reward == 0
        assert "tool_outputs" in obs

    def test_step_with_string_action(self, mock_dependencies):
        """Test ToolEnvironment step with string action."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        task = {"question": "test"}
        env = ToolEnvironment(task=task, tools=["python"], reward_fn=mock_dependencies["reward_fn"])
        
        action = "direct answer"
        obs, reward, done, info = env.step(action)
        
        assert done is True

    def test_step_with_dict_action(self, mock_dependencies):
        """Test ToolEnvironment step with dict action."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        task = {"question": "test"}
        env = ToolEnvironment(task=task, tools=["python"])
        
        action = {"id": "1", "function": {"name": "python", "arguments": {"code": "print(1)"}}}
        obs, reward, done, info = env.step(action)
        
        assert done is False

    def test_step_max_steps_reached(self, mock_dependencies):
        """Test ToolEnvironment step when max_steps reached."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        task = {"question": "test"}
        env = ToolEnvironment(task=task, tools=["python"], max_steps=2)
        
        action = [{"id": "1", "function": {"name": "python", "arguments": {}}}]
        env.step(action)
        obs, reward, done, info = env.step(action)
        
        assert done is True

    def test_from_dict(self, mock_dependencies):
        """Test ToolEnvironment from_dict static method."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        env_args = {
            "tools": ["python"],
            "max_steps": 5,
            "question": "test question",
        }
        
        env = ToolEnvironment.from_dict(env_args)
        
        assert env.max_steps == 5

    def test_from_dict_with_tool_map(self, mock_dependencies):
        """Test ToolEnvironment from_dict with tool_map."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        mock_tool_class = MagicMock()
        env_args = {
            "tool_map": {"custom": mock_tool_class},
            "max_steps": 3,
        }
        
        env = ToolEnvironment.from_dict(env_args)
        
        assert env.max_steps == 3

    def test_execute_tool_calls(self, mock_dependencies):
        """Test _execute_tool_calls method."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        env = ToolEnvironment(tools=["python"])
        
        tool_calls = [
            {"id": "call_1", "function": {"name": "python", "arguments": {"code": "print(1)"}}},
        ]
        
        result = env._execute_tool_calls(tool_calls)
        
        assert "call_1" in result

    def test_execute_tool_calls_with_string_args(self, mock_dependencies):
        """Test _execute_tool_calls with string arguments."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        env = ToolEnvironment(tools=["python"])
        
        tool_calls = [
            {"id": "call_1", "function": {"name": "python", "arguments": '{"code": "print(1)"}'}},
        ]
        
        result = env._execute_tool_calls(tool_calls)
        
        assert "call_1" in result

    def test_execute_tool_calls_with_dict_args(self, mock_dependencies):
        """Test _execute_tool_calls with dict arguments."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        env = ToolEnvironment(tools=["python"])
        
        tool_calls = [
            {"id": "call_1", "function": {"name": "python", "arguments": {"code": "print(1)"}}},
        ]
        
        result = env._execute_tool_calls(tool_calls)
        
        assert "call_1" in result

    def test_step_increments_step_count(self, mock_dependencies):
        """Test that step increments step_count."""
        from agents.math_agent.environment.tool_env import ToolEnvironment
        
        env = ToolEnvironment(tools=["python"])
        initial_count = env.step_count
        
        action = [{"id": "1", "function": {"name": "python", "arguments": {}}}]
        env.step(action)
        
        assert env.step_count == initial_count + 1
