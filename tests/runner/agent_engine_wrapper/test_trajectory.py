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
import pytest
import math
import torch
import numpy as np
from agentic_rl.runner.agent_engine_wrapper.base import Trajectory

_idx = 3
_prompt_tokens = torch.tensor([101, 200, 300, 400], dtype=torch.long)
_response_tokens = torch.tensor([500, 600, 700, 800], dtype=torch.long)
_response_masks = torch.tensor([1, 1, 1, 1], dtype=torch.long)
_trajectory_reward = 1.5
_chat_completions = [{"role": "assistant", "content": "This is a mock response."}]
_metrics = {"steps": 5,
            "reward_time": None,
            "env_time": 0.22,
            "llm_time": 30.21,
            "total_time": 30.43}
_metrics_type_invalid = "mock metrics invalid"
_metrics_keys_type_invalid = {1: 5,
                             "reward_time": None,
                             "env_time": 0.22,
                             "llm_time": 30.21,
                             "total_time": 30.43}
_metrics_keys_invalid = {"steps": "5"}
_metrics_steps_type_invalid = {"steps": "5",
                             "reward_time": None,
                             "env_time": 0.22,
                             "llm_time": 30.21,
                             "total_time": 30.43}
_metrics_env_time_type_invalid = {"steps": 5,
                                  "reward_time": None,
                                  "env_time": "0.22",
                                  "llm_time": 30.21,
                                  "total_time": 30.43}
_metrics_env_time_negative_invalid = {"steps": 5,
                                      "reward_time": None,
                                      "env_time": -5.2,
                                      "llm_time": 30.21,
                                      "total_time": 30.43}
_metrics_llm_time_negative_invalid = {"steps": 5,
                                      "reward_time": None,
                                      "env_time": 0.22,
                                      "llm_time": -30.21,
                                      "total_time": 30.43}
_metrics_total_time_negative_invalid = {"steps": 5,
                                        "reward_time": None,
                                        "env_time": 0.22,
                                        "llm_time": 30.21,
                                        "total_time": -30.43}
_metrics_step_negative_invalid = {"steps": -5,
                                  "reward_time": None,
                                  "env_time": 0.22,
                                  "llm_time": 30.21,
                                  "total_time": -30.43}


class TestTrajectopy:
    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, metrics", [
        (_idx, _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics),
    ])
    def test_Trajectory_with_parameters(self, idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, metrics):
        trajectory = Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=metrics)
        
        assert trajectory.idx == _idx
        assert torch.equal(trajectory.prompt_tokens, _prompt_tokens)
        assert torch.equal(trajectory.response_tokens, _response_tokens)
        assert torch.equal(trajectory.response_masks, _response_masks)
        assert math.isclose(trajectory.trajectory_reward, 1.5)
        assert trajectory.chat_completions == _chat_completions
        assert trajectory.metrics == _metrics

    @pytest.mark.parametrize("invalid_idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, metrics", [
        (3.0, _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics),
        ("3.0", _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics)
    ])
    def test_Trajectory_with_invalid_idx(self, invalid_idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, metrics):
        with pytest.raises(TypeError) as exc_info:
            Trajectory(idx=invalid_idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=metrics)
        assert f"idx must be an integer, got {type(invalid_idx).__name__}" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, metrics", [
        (-3, _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics),
    ])
    def test_Trajectory_with_invalid_idx_negative(self, invalid_idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, metrics):
        with pytest.raises(ValueError) as exc_info:
            Trajectory(idx=invalid_idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=metrics)
        assert f"idx must be non-negative" in str(exc_info.value)

    @pytest.mark.parametrize("idx, invalid_prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, metrics", [
        (_idx, ["_prompt_tokens"], _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics),
        (_idx, 123, _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics)
    ])
    def test_Trajectory_with_invalid_prompt_tokens(self, idx, invalid_prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, metrics):
        with pytest.raises(TypeError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=invalid_prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=metrics)
        assert f"prompt_tokens must be torch.Tensor, got {type(invalid_prompt_tokens).__name__}" in str(exc_info.value)

    @pytest.mark.parametrize("idx, invalid_prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, metrics", [
        (_idx, torch.tensor([1, 2, np.nan]), _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics),
    ])
    def test_Trajectory_with_invalid_prompt_tokens_nan(self, idx, invalid_prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, metrics):
        with pytest.raises(ValueError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=invalid_prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=metrics)
        assert f"prompt_tokens contains NaN values" in str(exc_info.value)

    @pytest.mark.parametrize("idx, invalid_prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, metrics", [
        (_idx, torch.tensor([1, 2, float('inf')]), _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics),
    ])
    def test_Trajectory_with_invalid_prompt_tokens_inf(self, idx, invalid_prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, metrics):
        with pytest.raises(ValueError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=invalid_prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=metrics)
        assert f"prompt_tokens contains Inf values" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, invalid_response_tokens, response_masks, trajectory_reward, chat_completions, metrics", [
        (_idx, _prompt_tokens, ["_response_tokens"], _response_masks, _trajectory_reward, _chat_completions, _metrics),
        (_idx, _prompt_tokens, "_response_tokens", _response_masks, _trajectory_reward, _chat_completions, _metrics)
    ])
    def test_Trajectory_with_invalid_response_tokens(self, idx, prompt_tokens, invalid_response_tokens, response_masks, trajectory_reward, chat_completions, metrics):
        with pytest.raises(TypeError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=invalid_response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=metrics)
        assert f"response_tokens must be torch.Tensor, got {type(invalid_response_tokens).__name__}" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, invalid_response_tokens, response_masks, trajectory_reward, chat_completions, metrics", [
        (_idx, _prompt_tokens, torch.tensor([1, 2, np.nan]), _response_masks, _trajectory_reward, _chat_completions, _metrics),
    ])
    def test_Trajectory_with_invalid_response_tokens_nan(self, idx, prompt_tokens, invalid_response_tokens, response_masks, trajectory_reward, chat_completions, metrics):
        with pytest.raises(ValueError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=invalid_response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=metrics)
        assert f"response_tokens contains NaN values" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, invalid_response_tokens, response_masks, trajectory_reward, chat_completions, metrics", [
        (_idx, _prompt_tokens, torch.tensor([1, 2, float('inf')]), _response_masks, _trajectory_reward, _chat_completions, _metrics),
    ])
    def test_Trajectory_with_invalid_response_tokens_inf(self, idx, prompt_tokens, invalid_response_tokens, response_masks, trajectory_reward, chat_completions, metrics):
        with pytest.raises(ValueError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=invalid_response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=metrics)
        assert f"response_tokens contains Inf values" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, invalid_response_masks, trajectory_reward, chat_completions, metrics", [
        (_idx, _prompt_tokens, _response_tokens, ["_response_masks"], _trajectory_reward, _chat_completions, _metrics),
        (_idx, _prompt_tokens, _response_tokens, "_response_masks", _trajectory_reward, _chat_completions, _metrics)
    ])
    def test_Trajectory_with_invalid_response_masks(self, idx, prompt_tokens, response_tokens, invalid_response_masks, trajectory_reward, chat_completions, metrics):
        with pytest.raises(TypeError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=invalid_response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=metrics)
        assert f"response_masks must be torch.Tensor, got {type(invalid_response_masks).__name__}" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, invalid_response_masks, trajectory_reward, chat_completions, metrics", [
        (_idx, _prompt_tokens, _response_tokens, torch.tensor([1, 2, np.nan]), _trajectory_reward, _chat_completions, _metrics),
    ])
    def test_Trajectory_with_invalid_response_masks_nan(self, idx, prompt_tokens, response_tokens, invalid_response_masks, trajectory_reward, chat_completions, metrics):
        with pytest.raises(ValueError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=invalid_response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=metrics)
        assert f"response_masks contains NaN values" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, invalid_response_masks, trajectory_reward, chat_completions, metrics", [
        (_idx, _prompt_tokens, _response_tokens, torch.tensor([1, 2, float('inf')]), _trajectory_reward, _chat_completions, _metrics),
    ])
    def test_Trajectory_with_invalid_response_masks_inf(self, idx, prompt_tokens, response_tokens, invalid_response_masks, trajectory_reward, chat_completions, metrics):
        with pytest.raises(ValueError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=invalid_response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=metrics)
        assert f"response_masks contains Inf values" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, response_masks, invalid_trajectory_reward, chat_completions, metrics", [
        (_idx, _prompt_tokens, _response_tokens, _response_masks, "_trajectory_reward", _chat_completions, _metrics),
        (_idx, _prompt_tokens, _response_tokens, _response_masks, [_trajectory_reward], _chat_completions, _metrics)
    ])
    def test_Trajectory_with_invalid_trajectory_reward(self, idx, prompt_tokens, response_tokens, response_masks, invalid_trajectory_reward, chat_completions, metrics):
        with pytest.raises(TypeError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=invalid_trajectory_reward, chat_completions=chat_completions, metrics=metrics)
        assert f"trajectory_reward must be a number, got {type(invalid_trajectory_reward).__name__}" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, invalid_chat_completions, metrics", [
        (_idx, _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, "_chat_completions", _metrics),
    ])
    def test_Trajectory_with_chat_completions(self, idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, invalid_chat_completions, metrics):
        with pytest.raises(TypeError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=invalid_chat_completions, metrics=metrics)
        assert f"chat_completions must be a list of dict, got {type(invalid_chat_completions).__name__}" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, invalid_chat_completions, metrics", [
        (_idx, _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, [{"hello": 123}], _metrics),
        (_idx, _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, [{123: "world"}], _metrics),
    ])
    def test_Trajectory_with_chat_completions_content(self, idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, invalid_chat_completions, metrics):
        with pytest.raises(TypeError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=invalid_chat_completions, metrics=metrics)
        assert "all keys and values in chat_completions dicts must be strings" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics", [
        (_idx, _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics_type_invalid),
    ])
    def test_Trajectory_with_metrics_type_invalid(self, idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics):
        with pytest.raises(TypeError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=invalid_metrics)
        assert f"metrics must be a dict, got {type(_metrics_type_invalid).__name__}" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics", [
        (_idx, _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics_keys_type_invalid),
    ])
    def test_Trajectory_with_metrics_keys_type_invalid(self, idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics):
        with pytest.raises(TypeError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=invalid_metrics)
        assert "all keys in metrics must be strings" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics", [
        (_idx, _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics_keys_invalid),
    ])
    def test_Trajectory_with_metrics_keys_invalid(self, idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics):
        with pytest.raises(ValueError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=invalid_metrics)
        assert f"metrics must contain exactly these keys: {sorted(set(_metrics.keys()))}" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics", [
        (_idx, _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics_steps_type_invalid),
    ])
    def test_Trajectory_with_metrics_steps_type_invalid(self, idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics):
        with pytest.raises(TypeError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=invalid_metrics)
        assert f"metric steps must be an integer, got {type(_metrics_steps_type_invalid['steps']).__name__}" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics", [
        (_idx, _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics_env_time_type_invalid),
    ])
    def test_Trajectory_with_metrics_env_time_type_invalid(self, idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics):
        with pytest.raises(TypeError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=invalid_metrics)
        assert f"metric env_time must be a number or None, got {type(_metrics_env_time_type_invalid['env_time']).__name__}" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics", [
        (_idx, _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics_env_time_negative_invalid),
    ])
    def test_Trajectory_with_metrics_env_time_negative_invalid(self, idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics):
        with pytest.raises(ValueError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=invalid_metrics)
        assert f"metric env_time must be non-negative if not None" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics", [
        (_idx, _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics_llm_time_negative_invalid),
    ])
    def test_Trajectory_with_metrics_llm_time_negative_invalid(self, idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics):
        with pytest.raises(ValueError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=invalid_metrics)
        assert f"metric llm_time must be non-negative if not None" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics", [
        (_idx, _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics_total_time_negative_invalid),
    ])
    def test_Trajectory_with_metrics_total_time_negative_invalid(self, idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics):
        with pytest.raises(ValueError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=invalid_metrics)
        assert f"metric total_time must be non-negative if not None" in str(exc_info.value)

    @pytest.mark.parametrize("idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics", [
        (_idx, _prompt_tokens, _response_tokens, _response_masks, _trajectory_reward, _chat_completions, _metrics_step_negative_invalid),
    ])
    def test_Trajectory_with_metrics_step_negative_invalid(self, idx, prompt_tokens, response_tokens, response_masks, trajectory_reward, chat_completions, invalid_metrics):
        with pytest.raises(ValueError) as exc_info:
            Trajectory(idx=idx, prompt_tokens=prompt_tokens, response_tokens=response_tokens, response_masks=response_masks,
                                trajectory_reward=trajectory_reward, chat_completions=chat_completions, metrics=invalid_metrics)
        assert f"metric steps must be non-negative" in str(exc_info.value)
