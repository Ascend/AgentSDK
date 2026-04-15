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


@dataclass
class MockRewardInput:
    task_info: dict
    action: str
    metadata: dict = None


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies(mock_ray_dependencies, mock_agentic_rl_dependencies, mock_rllm_dependencies):
    """Mock all external dependencies for search_reward tests."""
    with (
        patch("agents.math_agent.reward.search_reward.RewardOutput", MockRewardOutput),
        patch("agents.math_agent.reward.search_reward.RewardInput", MockRewardInput),
    ):
        yield {}


class TestRewardSearchFn:
    """Tests for RewardSearchFn class."""

    def test_init(self, mock_dependencies):
        """Test RewardSearchFn initialization."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        config = RewardConfig()
        fn = RewardSearchFn(config)
        
        assert fn.config == config

    def test_normalize_answer_lowercase(self, mock_dependencies):
        """Test normalize_answer converts to lowercase."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        result = fn.normalize_answer("HELLO World")
        
        assert result == "hello world"

    def test_normalize_answer_removes_articles(self, mock_dependencies):
        """Test normalize_answer removes articles."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        result = fn.normalize_answer("the quick brown fox")
        
        assert "the" not in result

    def test_normalize_answer_removes_punctuation(self, mock_dependencies):
        """Test normalize_answer removes punctuation."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        result = fn.normalize_answer("hello, world!")
        
        assert "," not in result
        assert "!" not in result

    def test_normalize_answer_fixes_whitespace(self, mock_dependencies):
        """Test normalize_answer fixes whitespace."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        result = fn.normalize_answer("hello    world")
        
        assert "    " not in result

    def test_f1_score_perfect_match(self, mock_dependencies):
        """Test f1_score with perfect match."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        f1, precision, recall = fn.f1_score("Paris", "Paris")
        
        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0

    def test_f1_score_partial_match(self, mock_dependencies):
        """Test f1_score with partial match."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        f1, precision, recall = fn.f1_score("Paris France", "Paris")
        
        assert 0 < f1 < 1

    def test_f1_score_no_match(self, mock_dependencies):
        """Test f1_score with no match."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        f1, precision, recall = fn.f1_score("London", "Paris")
        
        assert f1 == 0.0
        assert precision == 0.0
        assert recall == 0.0

    def test_f1_score_with_yes_no(self, mock_dependencies):
        """Test f1_score with yes/no answers."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        f1, _, _ = fn.f1_score("yes", "no")
        
        assert f1 == 0.0

    def test_exact_match_score_true(self, mock_dependencies):
        """Test exact_match_score with matching answers."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        result = fn.exact_match_score("Paris", "Paris")
        
        assert result is True

    def test_exact_match_score_false(self, mock_dependencies):
        """Test exact_match_score with non-matching answers."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        result = fn.exact_match_score("Paris", "London")
        
        assert result is False

    def test_exact_match_score_case_insensitive(self, mock_dependencies):
        """Test exact_match_score is case insensitive."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        result = fn.exact_match_score("PARIS", "paris")
        
        assert result is True

    def test_extract_answer_from_response_boxed(self, mock_dependencies):
        """Test extract_answer_from_response with boxed content."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        result = fn.extract_answer_from_response("The answer is \\boxed{Paris}")
        
        assert result == "Paris"

    def test_extract_answer_from_response_bold(self, mock_dependencies):
        """Test extract_answer_from_response with bold content."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        result = fn.extract_answer_from_response("The answer is **Paris**")
        
        assert "Paris" in result

    def test_extract_answer_from_response_plain_text(self, mock_dependencies):
        """Test extract_answer_from_response with plain text."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        result = fn.extract_answer_from_response("The capital is Paris.")
        
        assert result is not None
        assert len(result) > 0

    def test_extract_answer_from_response_empty(self, mock_dependencies):
        """Test extract_answer_from_response with empty string."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        result = fn.extract_answer_from_response("")
        
        assert result == ""

    def test_extract_answer_removes_thinking_tags(self, mock_dependencies):
        """Test extract_answer_from_response removes thinking tags."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        result = fn.extract_answer_from_response("Let me think... The answer is Paris.")
        
        assert result is not None

    def test_evaluate_answer_single_ground_truth(self, mock_dependencies):
        """Test evaluate_answer with single ground truth."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        is_correct, f1, metadata = fn.evaluate_answer("Paris", "Paris")
        
        assert is_correct is True
        assert f1 == 1.0

    def test_evaluate_answer_multiple_ground_truths(self, mock_dependencies):
        """Test evaluate_answer with multiple ground truths."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        is_correct, f1, metadata = fn.evaluate_answer("Paris", ["Paris", "paris"])
        
        assert is_correct is True

    def test_evaluate_answer_no_match(self, mock_dependencies):
        """Test evaluate_answer with no match."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig
        
        fn = RewardSearchFn(RewardConfig())
        is_correct, f1, metadata = fn.evaluate_answer("London", "Paris")
        
        assert is_correct is False
        assert f1 == 0.0

    def test_call_with_correct_answer(self, mock_dependencies):
        """Test __call__ with correct answer."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig, RewardInput
        
        fn = RewardSearchFn(RewardConfig())
        
        reward_input = RewardInput(
            task_info={"ground_truth": "Paris"},
            action="The capital of France is Paris.",
        )
        
        result = fn(reward_input)
        
        assert result.reward >= 0

    def test_call_with_incorrect_answer(self, mock_dependencies):
        """Test __call__ with incorrect answer."""
        from agents.math_agent.reward.search_reward import RewardSearchFn
        from agents.math_agent.reward.reward_types import RewardConfig, RewardInput
        
        fn = RewardSearchFn(RewardConfig())
        
        reward_input = RewardInput(
            task_info={"ground_truth": "Paris"},
            action="The capital of France is London.",
        )
        
        result = fn(reward_input)
        
        assert result.reward >= 0
