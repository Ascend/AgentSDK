# -*- coding: utf-8 -*-
#
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
# 
from copy import deepcopy
import torch
from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def compute_group_norm_advantage_return_patch(
        token_level_rewards: torch.Tensor,
        eos_mask: torch.Tensor,
        response_length: torch.Tensor,
        n_sample_per_prompt: int,
):
    """
    Compute advantage

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        response_length: response_length
        n_sample_per_prompt: `int`

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = torch.tensor(
        token_level_rewards,
        dtype=torch.float64,
        device=response_length.device
    )
    scores = scores.sum(dim=-1)
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
    logger.debug(f"Computed advantages shape: {advantages.shape}")
    return advantages, returns