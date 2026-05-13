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


class TestRewardConfig:
    """Tests for RewardConfig dataclass."""

    def test_default_values(self):
        """Test RewardConfig default values."""
        from agents.math_agent.reward.reward_types import RewardConfig
        
        config = RewardConfig()
        
        assert config.apply_format_reward is False
        assert config.math_reward_weight == 1.0
        assert config.use_math_orm is False
        assert config.code_reward_weight == 1.0
        assert config.cot_reward_weight == 0.0
        assert config.correct_reward == 1.0
        assert config.incorrect_reward == 0.0
        assert config.format_error_reward == 0.0
        assert config.unk_error_reward == 0.0
        assert config.toolcall_bonus == 0.5
        assert config.use_together_code_interpreter is False

    def test_custom_values(self):
        """Test RewardConfig with custom values."""
        from agents.math_agent.reward.reward_types import RewardConfig
        
        config = RewardConfig(
            apply_format_reward=True,
            math_reward_weight=0.5,
            correct_reward=2.0,
        )
        
        assert config.apply_format_reward is True
        assert config.math_reward_weight == 0.5
        assert config.correct_reward == 2.0

    def test_correct_reward_default(self):
        """Test correct_reward default value."""
        from agents.math_agent.reward.reward_types import RewardConfig
        
        config = RewardConfig()
        
        assert config.correct_reward == 1.0

    def test_incorrect_reward_default(self):
        """Test incorrect_reward default value."""
        from agents.math_agent.reward.reward_types import RewardConfig
        
        config = RewardConfig()
        
        assert config.incorrect_reward == 0.0

    def test_toolcall_bonus_default(self):
        """Test toolcall_bonus default value."""
        from agents.math_agent.reward.reward_types import RewardConfig
        
        config = RewardConfig()
        
        assert config.toolcall_bonus == 0.5


class TestRewardType:
    """Tests for RewardType enum."""

    def test_math_type(self):
        """Test RewardType.MATH value."""
        from agents.math_agent.reward.reward_types import RewardType
        
        assert RewardType.MATH.value == "MATH"

    def test_code_type(self):
        """Test RewardType.CODE value."""
        from agents.math_agent.reward.reward_types import RewardType
        
        assert RewardType.CODE.value == "CODE"

    def test_web_type(self):
        """Test RewardType.WEB value."""
        from agents.math_agent.reward.reward_types import RewardType
        
        assert RewardType.WEB.value == "WEB"

    def test_unk_type(self):
        """Test RewardType.UNK value."""
        from agents.math_agent.reward.reward_types import RewardType
        
        assert RewardType.UNK.value == "UNK"

    def test_all_types_exist(self):
        """Test all expected reward types exist."""
        from agents.math_agent.reward.reward_types import RewardType
        
        expected_types = {"MATH", "CODE", "WEB", "UNK"}
        actual_types = {rt.value for rt in RewardType}
        
        assert expected_types == actual_types

    def test_reward_type_enum_members(self):
        """Test RewardType enum members count."""
        from agents.math_agent.reward.reward_types import RewardType
        
        assert len(RewardType) == 4


class TestRewardOutput:
    """Tests for RewardOutput dataclass."""

    def test_reward_output_creation(self):
        """Test RewardOutput creation."""
        from agents.math_agent.reward.reward_types import RewardOutput
        
        output = RewardOutput(reward=1.0, metadata={"key": "value"})
        
        assert output.reward == 1.0
        assert output.metadata == {"key": "value"}

    def test_reward_output_with_is_correct(self):
        """Test RewardOutput with is_correct field."""
        from agents.math_agent.reward.reward_types import RewardOutput
        
        output = RewardOutput(reward=1.0, metadata={}, is_correct=True)
        
        assert output.is_correct is True

    def test_reward_output_default_is_correct(self):
        """Test RewardOutput default is_correct value."""
        from agents.math_agent.reward.reward_types import RewardOutput
        
        output = RewardOutput(reward=0.5, metadata={})
        
        assert output.is_correct is None


class TestRewardInput:
    """Tests for RewardInput dataclass."""

    def test_reward_input_creation(self):
        """Test RewardInput creation."""
        from agents.math_agent.reward.reward_types import RewardInput
        
        task_info = {"question": "test"}
        inp = RewardInput(task_info=task_info, action="test action")
        
        assert inp.task_info == task_info
        assert inp.action == "test action"

    def test_reward_input_fields(self):
        """Test RewardInput has required fields."""
        from agents.math_agent.reward.reward_types import RewardInput
        
        inp = RewardInput(
            task_info={"q": "test"},
            action="action",
        )
        
        assert inp.task_info == {"q": "test"}
        assert inp.action == "action"
