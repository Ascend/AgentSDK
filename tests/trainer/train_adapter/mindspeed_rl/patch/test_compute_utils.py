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
import re

import pytest
import torch

from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.compute_utils import compute_group_norm_advantage_return_patch


class TestComputeUtils:

    def test_compute_group_norm_advantage_return_patch(self):
        token_level_rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        eos_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])
        response_length = torch.tensor([[3], [3]])
        n_sample_per_prompt = 2

        advantages, returns = compute_group_norm_advantage_return_patch(token_level_rewards,
                                                                        eos_mask,
                                                                        response_length,
                                                                        n_sample_per_prompt)

        assert torch.allclose(advantages, returns), "Advantages and returns should be equal"
        assert advantages.shape == eos_mask.shape, "Advantages shape should match eos_mask shape"
        assert returns.shape == eos_mask.shape, "Returns shape should match eos_mask shape"

    def test_compute_group_norm_advantage_return_patch_with_different_response_length(self):
        token_level_rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        eos_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        response_length = torch.tensor([[2], [3]])
        n_sample_per_prompt = 2

        advantages, returns = compute_group_norm_advantage_return_patch(token_level_rewards,
                                                                        eos_mask,
                                                                        response_length,
                                                                        n_sample_per_prompt)

        assert torch.allclose(advantages, returns), "Advantages and returns should be equal"
        assert advantages.shape == eos_mask.shape, "Advantages shape should match eos_mask shape"
        assert returns.shape == eos_mask.shape, "Returns shape should match eos_mask shape"

    def test_compute_group_norm_advantage_failed_with_none_token_level_rewards(self):
        token_level_rewards = None
        eos_mask = None
        response_length = None

        with pytest.raises(TypeError, match="token_level_rewards must not be none and must be a torch.Tensor"):
            compute_group_norm_advantage_return_patch(token_level_rewards, eos_mask, response_length, 2)

    def test_compute_group_norm_advantage_failed_with_none_eos_mask(self):
        token_level_rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        eos_mask = None
        response_length = None

        with pytest.raises(TypeError, match="eos_mask must not be none and must be a torch.Tensor"):
            compute_group_norm_advantage_return_patch(token_level_rewards, eos_mask, response_length, 2)

    def test_compute_group_norm_advantage_failed_with_none_response_length(self):
        token_level_rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        eos_mask = torch.tensor([[2], [3]])
        response_length = None

        with pytest.raises(TypeError, match="response_length must not be none and must be a torch.Tensor"):
            compute_group_norm_advantage_return_patch(token_level_rewards, eos_mask, response_length, 2)

    def test_compute_group_norm_advantage_failed_with_n_sample_per_prompt_is_zero(self):
        token_level_rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        eos_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        response_length = torch.tensor([[2], [3]])
        n_sample_per_prompt = 0
        with pytest.raises(ValueError, match="n_sample_per_prompt must be greater than or equal to 1"):
            compute_group_norm_advantage_return_patch(
                token_level_rewards, eos_mask, response_length, n_sample_per_prompt)

    def test_compute_group_norm_advantage_failed_with_token_level_rewards_shape_dim_not_match(self):
        token_level_rewards = torch.tensor([1.0, 2.0, 3.0])
        eos_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        response_length = torch.tensor([[2], [3]])
        n_sample_per_prompt = 2
        with pytest.raises(ValueError, match="token_level_rewards must have 2 dimensions"):
            compute_group_norm_advantage_return_patch(
                token_level_rewards, eos_mask, response_length, n_sample_per_prompt)

    def test_compute_group_norm_advantage_failed_with_eos_mask_shape_dim_not_match(self):
        token_level_rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        eos_mask = torch.tensor([1, 1, 0])
        response_length = torch.tensor([[2], [3]])
        n_sample_per_prompt = 2
        with pytest.raises(ValueError, match="eos_mask must have 2 dimensions"):
            compute_group_norm_advantage_return_patch(
                token_level_rewards, eos_mask, response_length, n_sample_per_prompt)

    def test_compute_group_norm_advantage_failed_with_response_length_dim_not_match(self):
        token_level_rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        eos_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        response_length = torch.tensor([2, 3])
        n_sample_per_prompt = 2
        with pytest.raises(ValueError, match="response_length must have 2 dimensions"):
            compute_group_norm_advantage_return_patch(
                token_level_rewards, eos_mask, response_length, n_sample_per_prompt)

    def test_compute_group_norm_advantage_failed_with_shape_mismatch_1(self):
        token_level_rewards = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
        eos_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        response_length = torch.tensor([[2], [3]])
        n_sample_per_prompt = 2
        with pytest.raises(ValueError, match="token_level_rewards need a same shape with eos_mask"):
            compute_group_norm_advantage_return_patch(
                token_level_rewards, eos_mask, response_length, n_sample_per_prompt)

    def test_compute_group_norm_advantage_failed_with_shape_mismatch_2(self):
        token_level_rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        eos_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        response_length = torch.tensor([[2]])
        n_sample_per_prompt = 2
        with pytest.raises(ValueError,
                           match="the first dimension of token_level_rewards need to be same with response_length"):
            compute_group_norm_advantage_return_patch(
                token_level_rewards, eos_mask, response_length, n_sample_per_prompt)

    def test_compute_group_norm_advantage_failed_with_response_length_shape_dim_1(self):
        token_level_rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        eos_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        response_length = torch.tensor([[2, 2], [2, 2]])
        n_sample_per_prompt = 2
        with pytest.raises(ValueError, match=re.escape("response_length must have a shape of (N, 1)")):
            compute_group_norm_advantage_return_patch(
                token_level_rewards, eos_mask, response_length, n_sample_per_prompt)

    def test_compute_group_norm_advantage_failed_with_n_sample_per_prompt_error(self):
        token_level_rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        eos_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        response_length = torch.tensor([[2], [2]])
        n_sample_per_prompt = 3
        with pytest.raises(
                ValueError,
                match="the first dimension of token_level_rewards need to be a multiple of n_sample_per_prompt"):
            compute_group_norm_advantage_return_patch(
                token_level_rewards, eos_mask, response_length, n_sample_per_prompt)
