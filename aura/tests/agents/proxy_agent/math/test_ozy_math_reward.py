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
from unittest.mock import MagicMock, patch

import pytest


class MockRewardOutput:
    """Real class to store RewardOutput fields properly."""
    def __init__(self, reward=0.0, metadata=None, is_correct=False):
        self.reward = reward
        self.metadata = metadata or {}
        self.is_correct = is_correct


class TestOzyMathReward:

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        mock_reward_output = MockRewardOutput(reward=1.0, is_correct=True)

        mock_reward_fn_instance = MagicMock()
        mock_reward_fn_instance.return_value = mock_reward_output

        mock_reward_math_fn = MagicMock()
        mock_reward_math_fn.RewardMathFn = MagicMock(return_value=mock_reward_fn_instance)

        mock_reward_types = MagicMock()
        mock_reward_types.RewardConfig = MagicMock
        mock_reward_types.RewardInput = MagicMock
        mock_reward_types.RewardOutput = MockRewardOutput

        mock_loggers = MagicMock()
        mock_loggers.get_logger = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {
            "agents.math_agent.reward.math_reward": mock_reward_math_fn,
            "agents.math_agent.reward.reward_types": mock_reward_types,
            "aura.base.log.loggers": MagicMock(Loggers=mock_loggers),
        }):
            yield

    def test_math_res_reward_fn_not_done(self):
        from agents.proxy_agent.math.ozy_math_reward import math_res_reward_fn

        action = {
            "step_idx": 0,
            "is_last_step": False,
            "max_steps": 10,
            "tool_outputs": [],
            "assistant_message": "let me think",
        }
        task_info = {"ground_truth": "42"}

        result = math_res_reward_fn(action=action, task_info=task_info)
        assert result.reward == 0.0
        assert result.is_correct is False

    def test_math_res_reward_fn_done_with_string(self):
        from agents.proxy_agent.math.ozy_math_reward import math_res_reward_fn

        action = {
            "step_idx": 9,
            "is_last_step": True,
            "max_steps": 10,
            "tool_outputs": [],
            "assistant_message": "42",
        }
        task_info = {"ground_truth": "42"}

        result = math_res_reward_fn(action=action, task_info=task_info)
        assert result is not None

    def test_math_res_reward_fn_done_with_finish_tool(self):
        from agents.proxy_agent.math.ozy_math_reward import math_res_reward_fn

        action = {
            "step_idx": 9,
            "is_last_step": True,
            "max_steps": 10,
            "tool_outputs": [],
            "assistant_message": {
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "arguments": {"response": "42"},
                        },
                    }
                ]
            },
        }
        task_info = {"ground_truth": "42"}

        result = math_res_reward_fn(action=action, task_info=task_info)
        assert result is not None

    def test_math_res_reward_fn_done_with_content(self):
        from agents.proxy_agent.math.ozy_math_reward import math_res_reward_fn

        action = {
            "step_idx": 9,
            "is_last_step": True,
            "max_steps": 10,
            "tool_outputs": [],
            "assistant_message": {"content": "42"},
        }
        task_info = {"ground_truth": "42"}

        result = math_res_reward_fn(action=action, task_info=task_info)
        assert result is not None

    def test_math_res_reward_fn_done_with_raw_dict(self):
        from agents.proxy_agent.math.ozy_math_reward import math_res_reward_fn

        action = {
            "step_idx": 9,
            "is_last_step": True,
            "max_steps": 10,
            "tool_outputs": [],
            "assistant_message": {"role": "assistant"},
        }
        task_info = {"ground_truth": "42"}

        result = math_res_reward_fn(action=action, task_info=task_info)
        assert result is not None

    def test_math_res_reward_fn_max_steps_done(self):
        from agents.proxy_agent.math.ozy_math_reward import math_res_reward_fn

        action = {
            "step_idx": 9,
            "is_last_step": False,
            "max_steps": 10,
            "tool_outputs": [],
            "assistant_message": "42",
        }
        task_info = {"ground_truth": "42"}

        result = math_res_reward_fn(action=action, task_info=task_info)
        assert result is not None

    def test_math_res_reward_fn_assertions(self):
        from agents.proxy_agent.math.ozy_math_reward import math_res_reward_fn

        action = {
            "step_idx": None,
            "max_steps": None,
        }
        task_info = {}

        with pytest.raises(AssertionError):
            math_res_reward_fn(action=action, task_info=task_info)
