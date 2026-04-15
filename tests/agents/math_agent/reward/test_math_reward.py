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
    is_correct: bool = False
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies(mock_ray_dependencies, mock_agentic_rl_dependencies, mock_rllm_dependencies):
    """Mock all external dependencies for math_reward tests."""
    with (
        patch("rllm.globals.OAI_RM_MODEL", "test-model"),
        patch("rllm.globals.THOUGHT_DELIMITER_END", "</think"),
        patch("agents.math_agent.reward.math_reward.call_gemini_llm") as mock_gemini,
        patch("agents.math_agent.reward.math_reward.call_oai_rm_llm") as mock_oai,
        patch("rllm.system_prompts.ORM_PROMPT", "test prompt"),
        patch("agents.math_agent.reward.math_reward.extract_answer") as mock_extract,
        patch("agents.math_agent.reward.math_reward.grade_answer_mathd") as mock_mathd,
        patch("agents.math_agent.reward.math_reward.grade_answer_sympy") as mock_sympy,
        patch("agents.math_agent.reward.reward_types.RewardOutput", MockRewardOutput),
    ):
        mock_extract.return_value = "42"
        mock_mathd.return_value = False
        mock_sympy.return_value = False
        mock_gemini.return_value = "[[NO]]"
        mock_oai.return_value = "[[NO]]"
        yield {
            "extract_answer": mock_extract,
            "grade_mathd": mock_mathd,
            "grade_sympy": mock_sympy,
            "call_gemini": mock_gemini,
            "call_oai": mock_oai,
        }


class TestRewardMathFn:
    """Tests for RewardMathFn class."""

    def test_init(self, mock_dependencies):
        """Test RewardMathFn initialization."""
        from agents.math_agent.reward.math_reward import RewardMathFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        config = RewardConfig()
        fn = RewardMathFn(config)
        
        assert fn.config == config

    def test_call_with_correct_answer(self, mock_dependencies):
        """Test __call__ with correct answer."""
        from agents.math_agent.reward.math_reward import RewardMathFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        mock_dependencies["grade_mathd"].return_value = True
        mock_dependencies["extract_answer"].return_value = "4"
        
        config = RewardConfig()
        fn = RewardMathFn(config)
        
        task_info = {
            "task": {"question": "What is 2+2?", "ground_truth": "4"},
        }
        action = "The answer is \\boxed{4}"
        
        result = fn(task_info, action)
        
        assert result.is_correct is True
        assert result.reward == 1.0

    def test_call_with_incorrect_answer(self, mock_dependencies):
        """Test __call__ with incorrect answer."""
        from agents.math_agent.reward.math_reward import RewardMathFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        mock_dependencies["grade_mathd"].return_value = False
        mock_dependencies["grade_sympy"].return_value = False
        
        config = RewardConfig()
        fn = RewardMathFn(config)
        
        task_info = {
            "task": {"question": "What is 2+2?"},
            "ground_truth": "4",
        }
        action = "The answer is 5"
        
        result = fn(task_info, action)
        
        assert result.is_correct is False
        assert result.reward == 0.0

    def test_call_with_empty_response(self, mock_dependencies):
        """Test __call__ with empty response."""
        from agents.math_agent.reward.math_reward import RewardMathFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        config = RewardConfig()
        fn = RewardMathFn(config)
        
        task_info = {"task": {"question": "test", "ground_truth": "4"}}
        
        result = fn(task_info, "")
        
        assert result.reward == 0.0

    def test_call_with_none_response(self, mock_dependencies):
        """Test __call__ with None response."""
        from agents.math_agent.reward.math_reward import RewardMathFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        config = RewardConfig()
        fn = RewardMathFn(config)
        
        task_info = {"task": {"question": "test"}, "ground_truth": "4"}
        
        result = fn(task_info, None)
        
        assert result.reward == 0.0

    def test_call_with_no_ground_truth(self, mock_dependencies):
        """Test __call__ with no ground truth."""
        from agents.math_agent.reward.math_reward import RewardMathFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        config = RewardConfig()
        fn = RewardMathFn(config)
        
        task_info = {"task": {"question": "test"}}
        action = "The answer is 4"
        
        result = fn(task_info, action)
        
        assert result.reward == 0.0

    def test_call_with_multiple_ground_truths(self, mock_dependencies):
        """Test __call__ with multiple ground truths."""
        from agents.math_agent.reward.math_reward import RewardMathFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        mock_dependencies["grade_mathd"].side_effect = [False, True]
        mock_dependencies["extract_answer"].return_value = "four"
        
        config = RewardConfig()
        fn = RewardMathFn(config)
        
        task_info = {
            "task": {"question": "test", "ground_truth": ["4", "four"]},
        }
        action = "The answer is four"
        
        result = fn(task_info, action)
        
        assert result.is_correct is True

    def test_call_with_toolcall_bonus(self, mock_dependencies):
        """Test __call__ with toolcall bonus."""
        from agents.math_agent.reward.math_reward import RewardMathFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        mock_dependencies["grade_mathd"].return_value = True
        mock_dependencies["extract_answer"].return_value = "4"
        
        config = RewardConfig()
        fn = RewardMathFn(config)
        
        task_info = {
            "task": {"question": "test", "ground_truth": "4"},
            "has_toolcall": True,
        }
        action = "The answer is \\boxed{4}"
        
        result = fn(task_info, action)
        
        assert result.reward == 1.5

    def test_call_with_boxed_ground_truth(self, mock_dependencies):
        """Test __call__ with boxed ground truth."""
        from agents.math_agent.reward.math_reward import RewardMathFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        mock_dependencies["grade_mathd"].return_value = True
        mock_dependencies["extract_answer"].side_effect = ["4", "4"]
        
        config = RewardConfig()
        fn = RewardMathFn(config)
        
        task_info = {
            "task": {"question": "test", "ground_truth": "\\boxed{4}"},
        }
        action = "The answer is 4"
        
        result = fn(task_info, action)
        
        assert result.is_correct is True

    def test_call_with_orm_enabled(self, mock_dependencies):
        """Test __call__ with ORM enabled."""
        from agents.math_agent.reward.math_reward import RewardMathFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        mock_dependencies["grade_mathd"].return_value = False
        mock_dependencies["grade_sympy"].return_value = False
        mock_dependencies["call_gemini"].return_value = "[[YES]]"
        mock_dependencies["extract_answer"].return_value = "four"
        
        config = RewardConfig(use_math_orm=True)
        fn = RewardMathFn(config)
        
        task_info = {
            "task": {"question": "test", "ground_truth": "4"},
        }
        action = "The answer is four"
        
        result = fn(task_info, action)
        
        assert result.is_correct is True

    def test_call_with_format_reward_enabled(self, mock_dependencies):
        """Test __call__ with format reward enabled."""
        from agents.math_agent.reward.math_reward import RewardMathFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        config = RewardConfig(apply_format_reward=True)
        fn = RewardMathFn(config)
        
        task_info = {"task": {"question": "test", "ground_truth": "4"}}
        action = "answer without delimiter"
        
        result = fn(task_info, action)
        
        assert result.reward == 0.0


class TestRllmRewardFnMath:
    """Tests for rllm_reward_fn_math function."""

    def test_rllm_reward_fn_math_basic(self, mock_dependencies):
        """Test rllm_reward_fn_math with basic input."""
        from agents.math_agent.reward.math_reward import rllm_reward_fn_math
        
        mock_dependencies["grade_mathd"].return_value = True
        
        with patch("agents.math_agent.reward.math_reward.RewardMathFn") as mock_fn:
            mock_instance = mock_fn.return_value
            mock_instance.return_value = MockRewardOutput(reward=1.0, is_correct=True)
            
            result = rllm_reward_fn_math(
                data_source="gsm8k",
                llm_solution="The answer is 4",
                ground_truth="4",
            )
            
            assert result is not None
            assert result.is_correct is True

    def test_rllm_reward_fn_math_with_extra_info(self, mock_dependencies):
        """Test rllm_reward_fn_math with extra info."""
        from agents.math_agent.reward.math_reward import rllm_reward_fn_math
        
        mock_dependencies["grade_mathd"].return_value = True
        
        with patch("agents.math_agent.reward.math_reward.RewardMathFn") as mock_fn:
            mock_instance = mock_fn.return_value
            mock_instance.return_value = MockRewardOutput(reward=1.0, is_correct=True)
            
            result = rllm_reward_fn_math(
                data_source="gsm8k",
                llm_solution="4",
                ground_truth="4",
                extra_info={"difficulty": "easy"},
            )
            
            assert result is not None
            assert result.is_correct is True

    def test_rllm_reward_fn_math_with_list_ground_truth(self, mock_dependencies):
        """Test rllm_reward_fn_math with list ground truth."""
        from agents.math_agent.reward.math_reward import rllm_reward_fn_math
        
        mock_dependencies["grade_mathd"].return_value = True
        
        with patch("agents.math_agent.reward.math_reward.RewardMathFn") as mock_fn:
            mock_instance = mock_fn.return_value
            mock_instance.return_value = MockRewardOutput(reward=1.0, is_correct=True)
            
            result = rllm_reward_fn_math(
                data_source="gsm8k",
                llm_solution="4",
                ground_truth=["4", "four"],
            )
            
            assert result is not None
            assert result.is_correct is True
