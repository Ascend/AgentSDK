# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0 OR MulanPSL-2.0
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
"""Stepwise GRPO advantage for verl hybrid training."""

from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch


def compute_centered_scaled_reward_advantage(
    token_level_rewards: torch.Tensor,
    group_keys,
    response_mask: torch.Tensor,
    beta: float = 1.6,
):
    """Align with msrl compute_centered_scaled_reward_advantage_return_patch."""
    scores = token_level_rewards.sum(dim=-1).float()

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[group_keys[i]].append(scores[i])

        for idx, group_scores in id2score.items():
            id2mean[idx] = torch.stack(group_scores).mean()

        for i in range(bsz):
            adv = scores[i] - id2mean[group_keys[i]]
            if adv > 0:
                adv = adv * beta
            elif adv < 0:
                adv = adv * (2 - beta)
            scores[i] = adv

        scores = scores.unsqueeze(-1) * response_mask

    res = deepcopy(scores)
    return res, res


def compute_group_norm_advantage(
    token_level_rewards: torch.Tensor,
    group_keys,
    response_mask: torch.Tensor,
    epsilon: float = 1e-6,
):
    """Align with msrl compute_group_norm_advantage_return_by_index_patch."""
    scores = token_level_rewards.sum(dim=-1).float()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[group_keys[i]].append(scores[i])

        max_len = max((len(v) for v in id2score.values()), default=0)
        for idx, group_scores in id2score.items():
            if len(group_scores) == 1:
                if max_len == 1:
                    id2mean[idx] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
                    id2std[idx] = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
                else:
                    id2mean[idx] = torch.stack(group_scores).mean()
                    id2std[idx] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
            else:
                stacked = torch.stack(group_scores)
                id2mean[idx] = stacked.mean()
                id2std[idx] = stacked.std(unbiased=False)

        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[group_keys[i]]) / (id2std[group_keys[i]] + epsilon)
        scores = scores.unsqueeze(-1) * response_mask

    res = deepcopy(scores)
    return res, res


def _group_keys_from_batch(batch):
    """Prefer index columns; fall back to uid set by HybridAgentLoopManager."""
    ntb = batch.non_tensor_batch
    if "index_in_group" in ntb and "index_in_batch" in ntb and "index_in_steps" in ntb:
        g = [str(x) for x in np.asarray(ntb["index_in_group"]).reshape(-1)]
        b = [int(x) for x in np.asarray(ntb["index_in_batch"]).reshape(-1)]
        s = [int(x) for x in np.asarray(ntb["index_in_steps"]).reshape(-1)]
        return [f"{gg}_{bb}_{ss}" for gg, bb, ss in zip(g, b, s)]
    if "index_in_batch" in ntb and "index_in_steps" in ntb:
        b = [int(x) for x in np.asarray(ntb["index_in_batch"]).reshape(-1)]
        s = [int(x) for x in np.asarray(ntb["index_in_steps"]).reshape(-1)]
        return [f"{bb}_{ss}" for bb, ss in zip(b, s)]
    uid = ntb.get("uid")
    if uid is None:
        bsz = batch.batch["token_level_rewards"].shape[0]
        return [str(i) for i in range(bsz)]
    return [str(x) for x in np.asarray(uid).reshape(-1)]


def compute_stepwise_advantage(batch, algo_cfg):
    """Stepwise GRPO advantage entry point for verl hybrid."""
    mode = algo_cfg.get("stepwise_advantage_mode", "immediate_reward_centered_scaled")
    beta = float(algo_cfg.get("stepwise_advantage_beta", 1.6))

    token_level_rewards = batch.batch["token_level_rewards"].float()
    response_mask = batch.batch["response_mask"].float()

    min_len = min(token_level_rewards.shape[1], response_mask.shape[1])
    token_level_rewards = token_level_rewards[:, :min_len]
    response_mask_trunc = response_mask[:, :min_len]

    group_keys = _group_keys_from_batch(batch)

    if mode in ("group_norm", "immediate_reward_group_norm", "mc_return"):
        advantages, returns = compute_group_norm_advantage(
            token_level_rewards, group_keys, response_mask_trunc
        )
    elif mode == "immediate_reward_centered_scaled":
        advantages, returns = compute_centered_scaled_reward_advantage(
            token_level_rewards, group_keys, response_mask_trunc, beta=beta
        )
    else:
        raise ValueError(f"Unsupported stepwise_advantage_mode: {mode}")

    full_len = batch.batch["response_mask"].shape[1]
    if advantages.shape[1] < full_len:
        pad = full_len - advantages.shape[1]
        advantages = torch.nn.functional.pad(advantages, (0, pad))
        returns = torch.nn.functional.pad(returns, (0, pad))

    device = batch.batch["response_mask"].device
    batch.batch["advantages"] = advantages.to(device)
    batch.batch["returns"] = returns.to(device)
    return batch
