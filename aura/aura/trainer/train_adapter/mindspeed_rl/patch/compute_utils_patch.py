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
from collections import defaultdict
from copy import deepcopy
import torch
from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def compute_group_norm_advantage_return_patch(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    response_length: torch.Tensor,
    n_sample_per_prompt: int,
    use_stepwise_advantage: bool,
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
        use_stepwise_advantage: `bool`

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = torch.tensor(token_level_rewards, dtype=torch.float64, device=response_length.device)
    scores = scores.sum(dim=-1)
    if use_stepwise_advantage:
        scores = torch.tensor(scores, dtype=torch.float32, device=response_length.device)
        new_token_level_rewards = scores.unsqueeze(1).repeat(1, eos_mask.shape[1])
    else:
        scores = scores.reshape(-1, n_sample_per_prompt)
        scores = (scores - scores.mean(dim=1, keepdim=True)) / (scores.std(dim=1, keepdim=True) + 1e-6)
        scores = scores.reshape(response_length.shape)
        scores = torch.tensor(scores, dtype=torch.float32, device=response_length.device)
        new_token_level_rewards = scores.repeat(1, eos_mask.shape[1])

    new_token_level_rewards = new_token_level_rewards * eos_mask
    advantages = deepcopy(new_token_level_rewards)
    returns = deepcopy(advantages)
    logger.debug(f"Computed advantages shape: {advantages.shape}")
    return advantages, returns



def compute_group_norm_advantage_return_by_index_patch(
        token_level_rewards: torch.Tensor,
        index_in_batch_list,
        response_mask: torch.Tensor,
        epsilon: float = 1e-6
):
    """
    Compute group-normalized advantage using index_in_batch_list for flexible grouping.
    Supports both chain (n_samples_per_prompt > 1) and beam stepwise modes.
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index_in_batch_list[i]].append(scores[i])

        max_len = 0
        for idx in id2score:
            if len(id2score[idx]) > max_len:
                max_len = len(id2score[idx])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                if max_len == 1:
                    id2mean[idx] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
                    id2std[idx] = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
                else:
                    stacked = torch.stack(id2score[idx])
                    id2mean[idx] = stacked.mean()
                    id2std[idx] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
            elif len(id2score[idx]) > 1:
                stacked = torch.stack(id2score[idx])
                id2mean[idx] = stacked.mean()
                id2std[idx] = stacked.std(unbiased=False)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index_in_batch_list[i]]) / (id2std[index_in_batch_list[i]] + epsilon)
        scores = scores.unsqueeze(-1) * response_mask
    res = deepcopy(scores)
    return res, res


def compute_centered_scaled_reward_advantage_return_patch(
        token_level_rewards: torch.Tensor,
        index_in_batch_list,
        response_mask: torch.Tensor,
        beta: float = 1.6
):
    """
    Reward-aware step advantage:
        adv = reward - group_avg
        adv > 0: adv *= beta
        adv < 0: adv *= (2 - beta)
    """
    scores = token_level_rewards.sum(dim=-1).float()

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index_in_batch_list[i]].append(scores[i])

        for idx, group_scores in id2score.items():
            id2mean[idx] = torch.stack(group_scores).mean()

        for i in range(bsz):
            adv = scores[i] - id2mean[index_in_batch_list[i]]
            if adv > 0:
                adv = adv * beta
            elif adv < 0:
                adv = adv * (2 - beta)
            scores[i] = adv

        scores = scores.unsqueeze(-1) * response_mask

    res = deepcopy(scores)
    return res, res
