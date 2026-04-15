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
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass


@dataclass
class MockRewardOutput:
    reward: float
    metadata: dict
    is_correct: bool = False


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies(mock_ray_dependencies, mock_agentic_rl_dependencies, mock_rllm_dependencies):
    """Mock all external dependencies for reward_fn tests."""
    mock_reward_output = MockRewardOutput(reward=1.0, metadata={}, is_correct=True)
    
    mock_math_fn = MagicMock(return_value=mock_reward_output)
    mock_search_fn = MagicMock(return_value=mock_reward_output)
    mock_code_fn = MagicMock(return_value=mock_reward_output)
    
    with (
        patch("agents.math_agent.reward.reward_fn.RewardMathFn") as mock_math_class,
        patch("agents.math_agent.reward.reward_fn.RewardSearchFn") as mock_search_class,
        patch("agents.math_agent.reward.reward_fn.RewardCodeFn") as mock_code_class,
        patch("agents.math_agent.reward.reward_fn.RewardConfig") as mock_config,
        patch("agents.math_agent.reward.reward_fn.RewardInput") as mock_input,
        patch("agents.math_agent.reward.reward_fn.RewardOutput", MockRewardOutput),
    ):
        mock_math_class.return_value = mock_math_fn
        mock_search_class.return_value = mock_search_fn
        mock_code_class.return_value = mock_code_fn
        mock_config.return_value = MagicMock()
        yield {
            "math_fn": mock_math_fn,
            "search_fn": mock_search_fn,
            "code_fn": mock_code_fn,
            "reward_output": mock_reward_output,
        }


class TestRewardFunctionProtocol:
    """Tests for RewardFunction protocol."""

    def test_zero_reward(self, mock_dependencies):
        """Test zero_reward function."""
        from agents.math_agent.reward.reward_fn import zero_reward
        
        result = zero_reward({"task": "test"}, "action")
        
        assert result.reward == 0.0
        assert result.metadata == {}

    def test_zero_reward_with_empty_task(self, mock_dependencies):
        """Test zero_reward with empty task."""
        from agents.math_agent.reward.reward_fn import zero_reward
        
        result = zero_reward({}, "any action")
        
        assert result.reward == 0.0


class TestMathRewardFn:
    """Tests for math_reward_fn function."""

    def test_math_reward_fn_basic(self, mock_dependencies):
        """Test math_reward_fn with basic input."""
        from agents.math_agent.reward.reward_fn import math_reward_fn
        
        task_info = {"task": {"question": "What is 2+2?"}, "ground_truth": "4"}
        result = math_reward_fn(task_info, "The answer is 4")
        
        assert result is not None

    def test_math_reward_fn_with_ground_truth(self, mock_dependencies):
        """Test math_reward_fn with ground truth."""
        from agents.math_agent.reward.reward_fn import math_reward_fn
        
        task_info = {
            "task": {"question": "Calculate"},
            "ground_truth": ["4", "four"],
        }
        result = math_reward_fn(task_info, "4")
        
        assert result is not None


class TestSearchRewardFn:
    """Tests for search_reward_fn function."""

    def test_search_reward_fn_basic(self, mock_dependencies):
        """Test search_reward_fn with basic input."""
        from agents.math_agent.reward.reward_fn import search_reward_fn
        
        task_info = {"question": "What is the capital of France?"}
        result = search_reward_fn(task_info, "Paris")
        
        assert result is not None


class TestCodeRewardFn:
    """Tests for code_reward_fn function."""

    def test_code_reward_fn_basic(self, mock_dependencies):
        """Test code_reward_fn with basic input."""
        from agents.math_agent.reward.reward_fn import code_reward_fn
        
        task_info = {"problem": "Write a function"}
        result = code_reward_fn(task_info, "def f(): pass")
        
        assert result is not None


class TestRewardFunctionIntegration:
    """Integration tests for reward functions."""

    def test_all_reward_functions_return_reward_output(self, mock_dependencies):
        """Test that all reward functions return RewardOutput."""
        from agents.math_agent.reward.reward_fn import (
            zero_reward,
            math_reward_fn,
            search_reward_fn,
            code_reward_fn,
        )
        
        task_info = {"task": "test"}
        action = "test action"
        
        zero_result = zero_reward(task_info, action)
        math_result = math_reward_fn(task_info, action)
        search_result = search_reward_fn(task_info, action)
        code_result = code_reward_fn(task_info, action)
        
        assert hasattr(zero_result, "reward")
        assert hasattr(math_result, "reward")
        assert hasattr(search_result, "reward")
        assert hasattr(code_result, "reward")
