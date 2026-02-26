#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
import math
import unittest
import pytest
import torch
import numpy as np
from agentic_rl.runner.agent_engine_wrapper.base import Trajectory, Step, StepTrajectory


_idx = 3
_prompt_tokens = torch.tensor([101, 200, 300, 400], dtype=torch.long)
_response_tokens = torch.tensor([500, 600, 700, 800], dtype=torch.long)
_response_masks = torch.tensor([1, 1, 1, 1], dtype=torch.long)
_trajectory_reward = 1.5
_chat_completions = [{"role": "assistant", "content": "This is a mock response."}]
_metrics = {
    "steps": 5,
    "reward_time": None,
    "env_time": 0.22,
    "llm_time": 30.21,
    "total_time": 30.43,
    "toolcall_reward": 0.0,
    "res_reward": 0.1
}
_metrics_type_invalid = "mock metrics invalid"
_metrics_keys_type_invalid = {1: 5,
                             "reward_time": None,
                             "env_time": 0.22,
                             "llm_time": 30.21,
                             "total_time": 30.43,
                             "toolcall_reward": 0.0,
                             "res_reward": 0.1}
_metrics_keys_invalid = {"steps": "5"}
_metrics_steps_type_invalid = {"steps": "5",
                             "reward_time": None,
                             "env_time": 0.22,
                             "llm_time": 30.21,
                             "total_time": 30.43,
                             "toolcall_reward": 0.0,
                             "res_reward": 0.1}
_metrics_env_time_type_invalid = {"steps": 5,
                                  "reward_time": None,
                                  "env_time": "0.22",
                                  "llm_time": 30.21,
                                  "total_time": 30.43,
                                  "toolcall_reward": 0.0,
                                  "res_reward": 0.1}
_metrics_env_time_negative_invalid = {"steps": 5,
                                      "reward_time": None,
                                      "env_time": -5.2,
                                      "llm_time": 30.21,
                                      "total_time": 30.43,
                                      "toolcall_reward": 0.0,
                                      "res_reward": 0.1}
_metrics_llm_time_negative_invalid = {"steps": 5,
                                      "reward_time": None,
                                      "env_time": 0.22,
                                      "llm_time": -30.21,
                                      "total_time": 30.43,
                                      "toolcall_reward": 0.0,
                                      "res_reward": 0.1}
_metrics_total_time_negative_invalid = {"steps": 5,
                                        "reward_time": None,
                                        "env_time": 0.22,
                                        "llm_time": 30.21,
                                        "total_time": -30.43,
                                        "toolcall_reward": 0.0,
                                        "res_reward": 0.1}
_metrics_step_negative_invalid = {"steps": -5,
                                  "reward_time": None,
                                  "env_time": 0.22,
                                  "llm_time": 30.21,
                                  "total_time": 30.43,
                                  "toolcall_reward": 0.0,
                                  "res_reward": 0.1}
_metrics_toolcall_reward_negative_invalid = {"steps": -5,
                                             "reward_time": None,
                                             "env_time": 0.22,
                                             "llm_time": 30.21,
                                             "total_time": 30.43,
                                             "toolcall_reward": "0.0",
                                             "res_reward": 0.1}
_metrics_res_reward_negative_invalid = {"steps": -5,
                                        "reward_time": None,
                                        "env_time": 0.22,
                                        "llm_time": 30.21,
                                        "total_time": 30.43,
                                        "toolcall_reward": 0.0,
                                        "res_reward": "0.1"}


@pytest.fixture
def valid_kwargs():
    return {
        "idx": _idx,
        "prompt_tokens": _prompt_tokens,
        "response_tokens": _response_tokens,
        "response_masks": _response_masks,
        "trajectory_reward": _trajectory_reward,
        "chat_completions": _chat_completions,
        "metrics": _metrics,
    }


class TestTrajectopy:
    @staticmethod
    def test_valid_construction(valid_kwargs):
        traj = Trajectory(**valid_kwargs)
        assert traj.idx == _idx
        assert torch.equal(traj.prompt_tokens, _prompt_tokens)
        assert torch.equal(traj.response_tokens, _response_tokens)
        assert torch.equal(traj.response_masks, _response_masks)
        assert math.isclose(traj.trajectory_reward, 1.5)
        assert traj.chat_completions == _chat_completions
        assert traj.metrics == _metrics

    @pytest.mark.parametrize("invalid_idx, expected_exception, expected_msg", [
        (3.0, TypeError, "idx must be an integer, got float"),
        ("3", TypeError, "idx must be an integer, got str"),
        (-3, ValueError, "idx must be non-negative"),
    ])
    def test_invalid_idx(self, valid_kwargs, invalid_idx, expected_exception, expected_msg):
        kwargs = valid_kwargs.copy()
        kwargs["idx"] = invalid_idx
        with pytest.raises(expected_exception) as exc_info:
            Trajectory(**kwargs)
        assert expected_msg in str(exc_info.value)

    @pytest.mark.parametrize("invalid_tokens, expected_exception, expected_msg", [
        (["_prompt_tokens"], TypeError, "prompt_tokens must be torch.Tensor, got list"),
        (123, TypeError, "prompt_tokens must be torch.Tensor, got int"),
        (torch.tensor([1, 2, np.nan]), ValueError, "prompt_tokens contains NaN values"),
        (torch.tensor([1, 2, float('inf')]), ValueError, "prompt_tokens contains Inf values"),
    ])
    def test_invalid_prompt_tokens(self, valid_kwargs, invalid_tokens, expected_exception, expected_msg):
        kwargs = valid_kwargs.copy()
        kwargs["prompt_tokens"] = invalid_tokens
        with pytest.raises(expected_exception) as exc_info:
            Trajectory(**kwargs)
        assert expected_msg in str(exc_info.value)

    @pytest.mark.parametrize("invalid_tokens, expected_exception, expected_msg", [
        (["_response_tokens"], TypeError, "response_tokens must be torch.Tensor, got list"),
        ("_response_tokens", TypeError, "response_tokens must be torch.Tensor, got str"),
        (torch.tensor([1, 2, np.nan]), ValueError, "response_tokens contains NaN values"),
        (torch.tensor([1, 2, float('inf')]), ValueError, "response_tokens contains Inf values"),
    ])
    def test_invalid_response_tokens(self, valid_kwargs, invalid_tokens, expected_exception, expected_msg):
        kwargs = valid_kwargs.copy()
        kwargs["response_tokens"] = invalid_tokens
        with pytest.raises(expected_exception) as exc_info:
            Trajectory(**kwargs)
        assert expected_msg in str(exc_info.value)

    @pytest.mark.parametrize("invalid_masks, expected_exception, expected_msg", [
        (["_response_masks"], TypeError, "response_masks must be torch.Tensor, got list"),
        ("_response_masks", TypeError, "response_masks must be torch.Tensor, got str"),
        (torch.tensor([1, 2, np.nan]), ValueError, "response_masks contains NaN values"),
        (torch.tensor([1, 2, float('inf')]), ValueError, "response_masks contains Inf values"),
    ])
    def test_invalid_response_masks(self, valid_kwargs, invalid_masks, expected_exception, expected_msg):
        kwargs = valid_kwargs.copy()
        kwargs["response_masks"] = invalid_masks
        with pytest.raises(expected_exception) as exc_info:
            Trajectory(**kwargs)
        assert expected_msg in str(exc_info.value)

    @pytest.mark.parametrize("invalid_reward, expected_msg", [
        ("_trajectory_reward", "trajectory_reward must be a number, got str"),
        ([_trajectory_reward], "trajectory_reward must be a number, got list"),
    ])
    def test_invalid_trajectory_reward(self, valid_kwargs, invalid_reward, expected_msg):
        kwargs = valid_kwargs.copy()
        kwargs["trajectory_reward"] = invalid_reward
        with pytest.raises(TypeError) as exc_info:
            Trajectory(**kwargs)
        assert expected_msg in str(exc_info.value)

    @pytest.mark.parametrize("invalid_chat, expected_exception, expected_msg", [
        ("_chat_completions", TypeError, "chat_completions must be a list of dict, got str"),
        ([{"hello": 123}], TypeError, "all keys and values in chat_completions dicts must be strings"),
        ([{123: "world"}], TypeError, "all keys and values in chat_completions dicts must be strings"),
    ])
    def test_invalid_chat_completions(self, valid_kwargs, invalid_chat, expected_exception, expected_msg):
        kwargs = valid_kwargs.copy()
        kwargs["chat_completions"] = invalid_chat
        with pytest.raises(expected_exception) as exc_info:
            Trajectory(**kwargs)
        assert expected_msg in str(exc_info.value)

    @pytest.mark.parametrize("invalid_metrics, expected_exception, expected_msg", [
        (_metrics_type_invalid, TypeError, "metrics must be a dict, got str"),
        (_metrics_keys_type_invalid, TypeError, "all keys in metrics must be strings"),
        (_metrics_keys_invalid, ValueError,
         f"metrics must contain exactly these keys: {sorted(set(_metrics.keys()))}"),
        (_metrics_steps_type_invalid, TypeError, "metric steps must be an integer, got str"),
        (_metrics_total_time_negative_invalid, ValueError, "metric total_time must be non-negative if not None"),
        (_metrics_step_negative_invalid, ValueError, "metric steps must be non-negative"),
        (_metrics_toolcall_reward_negative_invalid, TypeError, "metric toolcall_reward must be a number"),
        (_metrics_res_reward_negative_invalid, TypeError, "metric res_reward must be a number"),
    ])
    def test_invalid_metrics(self, valid_kwargs, invalid_metrics, expected_exception, expected_msg):
        kwargs = valid_kwargs.copy()
        kwargs["metrics"] = invalid_metrics
        with pytest.raises(expected_exception) as exc_info:
            Trajectory(**kwargs)
        assert expected_msg in str(exc_info.value)


class TestStep(unittest.TestCase):
    def test_validate_step_all_valid(self):
        try:
            Step(chat_completions=[{"key1": "value1"}, {"key2": "value2"}],
                 thought="valid thought",
                 action="action",
                 observation={"key": "value"},
                 model_response="valid response",
                 info={"key": "value"},
                 reward=1.0,
                 done=True,
                 mc_return=2.0)
        except Exception as e:
            self.fail(f"step failed with valid inputs: {e}")


class TestStepTrajectory(unittest.TestCase):
    def test_validate_step_trajectory_all_valid(self):
        try:
            step = Step(chat_completions=[{"key1": "value1"}, {"key2": "value2"}],
                        thought="valid thought",
                        action="action",
                        observation={"key": "value"},
                        model_response="valid response",
                        info={"key": "value"},
                        reward=1.0,
                        done=True,
                        mc_return=2.0)
            StepTrajectory(prompt_tokens=_prompt_tokens, response_tokens=_response_tokens,
                           response_masks=_response_masks,
                           idx=_idx, trajectory_reward=_trajectory_reward, chat_completions=_chat_completions,
                           metrics=_metrics,
                           task={"task": "task info"}, steps=[step, step])
        except Exception as e:
            self.fail(f"step trajectory failed with valid inputs: {e}")

    def test_invalid_trajectory(self):
        with self.assertRaises(TypeError) as context:
            StepTrajectory(idx="error_idx", prompt_tokens=_prompt_tokens, response_tokens=_response_tokens,
                           response_masks=_response_masks, steps=[Step(action="move", reward=1.0)])
        self.assertEqual(str(context.exception), "Trajectory's idx must be an integer, got str")

    def test_empty_steps(self):
        # 测试空的steps列表
        with self.assertRaises(ValueError) as context:
            StepTrajectory(prompt_tokens=_prompt_tokens, response_tokens=_response_tokens,
                           response_masks=_response_masks, steps=[])
        self.assertEqual(str(context.exception), "steps must be a non empty list of Step instances")

    def test_non_list_steps(self):
        # 测试非列表的steps
        with self.assertRaises(ValueError) as context:
            StepTrajectory(prompt_tokens=_prompt_tokens, response_tokens=_response_tokens,
                           response_masks=_response_masks, steps="not a list")
        self.assertEqual(str(context.exception), "steps must be a non empty list of Step instances")

    def test_steps_with_non_step_instance(self):
        # 测试steps列表中包含非Step实例
        with self.assertRaises(ValueError) as context:
            StepTrajectory(prompt_tokens=_prompt_tokens, response_tokens=_response_tokens,
                           response_masks=_response_masks,
                           steps=[Step(action="move", reward=1.0), "not a Step instance"])
        self.assertEqual(str(context.exception), "steps must be a non empty list of Step instances")
