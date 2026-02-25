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

from copy import deepcopy

import torch

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__)


def compute_group_norm_advantage_return_patch(
        token_level_rewards: torch.Tensor,
        eos_mask: torch.Tensor,
        response_length: torch.Tensor,
        n_sample_per_prompt: int,
        use_stepwise_advantage: bool = False
):
    """
    Compute advantage and return values.

    Args:
        token_level_rewards (torch.Tensor): Reward values at the token level.
        eos_mask (torch.Tensor): Mask indicating the end of sequences.
        response_length (torch.Tensor): Length of the responses.
        n_sample_per_prompt (int): Number of samples per prompt.
        use_stepwise_advantage (bool): Whether to use step-wise advantage mode.

    Returns:
        tuple: A tuple containing advantage and its copy.
    """
    _check_compute_group_norm_shape(token_level_rewards,
                                    eos_mask,
                                    response_length,
                                    n_sample_per_prompt,
                                    use_stepwise_advantage)

    scores = torch.tensor(
        token_level_rewards,
        dtype=torch.float64,
        device=response_length.device
    )
    scores = scores.sum(dim=-1)

    if use_stepwise_advantage:
        scores = torch.tensor(
            scores,
            dtype=torch.float32,
            device=response_length.device
        )
        new_token_level_rewards = scores.unsqueeze(1).repeat(1, eos_mask.shape[1])
    else:
        scores = scores.reshape(-1, n_sample_per_prompt)
        scores = (scores - scores.mean(dim=1, keepdim=True)) / (scores.std(dim=1, keepdim=True) + 1e-6)
        scores = scores.reshape(response_length.shape)
        scores = torch.tensor(
            scores,
            dtype=torch.float32,
            device=response_length.device
        )
        new_token_level_rewards = scores.repeat(1, eos_mask.shape[1])

    new_token_level_rewards = new_token_level_rewards * eos_mask
    advantages = deepcopy(new_token_level_rewards)
    returns = deepcopy(advantages)

    return advantages, returns


def _check_compute_group_norm_shape(token_level_rewards: torch.Tensor,
                                    eos_mask: torch.Tensor,
                                    response_length: torch.Tensor,
                                    n_sample_per_prompt: int,
                                    use_stepwise_advantage: bool):
    if token_level_rewards is None or not isinstance(token_level_rewards, torch.Tensor):
        logger.error("token_level_rewards must not be none and must be a torch.Tensor")
        raise TypeError("token_level_rewards must not be none and must be a torch.Tensor")
    if eos_mask is None or not isinstance(eos_mask, torch.Tensor):
        logger.error("eos_mask must not be none and must be a torch.Tensor")
        raise TypeError("eos_mask must not be none and must be a torch.Tensor")
    if response_length is None or not isinstance(response_length, torch.Tensor):
        logger.error("response_length must not be none and must be a torch.Tensor")
        raise TypeError("response_length must not be none and must be a torch.Tensor")
    if n_sample_per_prompt < 1:
        logger.error("n_sample_per_prompt must be greater than or equal to 1")
        raise ValueError("n_sample_per_prompt must be greater than or equal to 1")

    if len(token_level_rewards.shape) != 2:
        logger.error("token_level_rewards must have 2 dimensions")
        raise ValueError("token_level_rewards must have 2 dimensions")
    if len(eos_mask.shape) != 2:
        logger.error("eos_mask must have 2 dimensions")
        raise ValueError("eos_mask must have 2 dimensions")
    if len(response_length.shape) != 2:
        logger.error("response_length must have 2 dimensions")
        raise ValueError("response_length must have 2 dimensions")

    if token_level_rewards.shape != eos_mask.shape:
        logger.error("token_level_rewards need a same shape with eos_mask")
        raise ValueError("token_level_rewards need a same shape with eos_mask")

    if token_level_rewards.shape[0] != response_length.shape[0]:
        logger.error("the first dimension of token_level_rewards need to be same with response_length")
        raise ValueError("the first dimension of token_level_rewards need to be same with response_length")

    if response_length.shape[1] != 1:
        logger.error("response_length must have a shape of (N, 1)")
        raise ValueError("response_length must have a shape of (N, 1)")

    if not use_stepwise_advantage and token_level_rewards.shape[0] % n_sample_per_prompt != 0:
        logger.error("the first dimension of token_level_rewards need to be a multiple of n_sample_per_prompt")
        raise ValueError("the first dimension of token_level_rewards need to be a multiple of n_sample_per_prompt")
