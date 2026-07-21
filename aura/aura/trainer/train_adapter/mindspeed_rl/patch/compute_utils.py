#!/usr/bin/env python3
# coding=utf-8

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


import time
import torch
import ray

from mindspeed_rl.utils.utils import generate_mask, get_current_dp_range_indexes
from mindspeed_rl.utils.pad_process import (
    truncate_rows,
    remove_padding_tensor_dict_to_dict,
    padding_dict_to_tensor_dict,
)
from mindspeed_rl.trainer.utils.transfer_dock import pad_experience
from mindspeed_rl.trainer.utils.compute_utils import compute_gae_advantage_return
from .compute_utils_patch import (
    compute_centered_scaled_reward_advantage_return_patch,
    compute_group_norm_advantage_return_by_index_patch,
    compute_group_norm_advantage_return_patch,
)


def compute_advantage(self, blocking=False, guarantee_order=False):
    experience_count = self.micro_batch_size
    rl_config = self.actor_worker.rl_config

    start_adv_time = time.time()
    compute_advantage_ref = compute_advantage_utils.options(num_cpus=self.num_cpus_for_local_task).remote(
        self.transfer_dock,
        self.gamma,
        self.lam,
        adv_estimator=self.adv_estimator,
        experience_count=experience_count,
        tokenizer=self.tokenizer,
        global_batch_size=self.global_batch_size * self.n_samples_per_prompt,
        guarantee_order=guarantee_order,
        n_sample_per_prompt=rl_config.n_samples_per_prompt,
        use_stepwise_advantage=rl_config.use_stepwise_advantage,
        stepwise_advantage_mode=getattr(rl_config, "stepwise_advantage_mode", "immediate_reward_centered_scaled"),
        stepwise_advantage_beta=getattr(rl_config, "stepwise_advantage_beta", 1.6),
    )
    if blocking:
        ray.get(compute_advantage_ref)
    end_adv_time = time.time()
    ray.get(
        self.transfer_dock.update_metrics.remote(
            "timing/adv", value=[round(end_adv_time, 4), round(start_adv_time, 4)], cumulate=True
        )
    )
    ray.get(
        self.transfer_dock.update_metrics.remote("end_time/end_adv_time", value=[round(end_adv_time, 4)], cumulate=True)
    )


@ray.remote
def compute_advantage_utils(
    td,
    gamma,
    lam,
    adv_estimator,
    experience_count,
    tokenizer,
    global_batch_size,
    guarantee_order,
    n_sample_per_prompt,
    use_stepwise_advantage,
    stepwise_advantage_mode="immediate_reward_centered_scaled",
    stepwise_advantage_beta=1.6,
    use_kl_in_reward=False,
):
    """
    Compute the advantage function based on different adv_estimator.

    - gae: unchanged upstream GAE path
    - group_norm + use_stepwise_advantage: stepwise beam (index grouping)
    - group_norm without stepwise: original group_norm patch
    """
    experience_count = ray.get(td.get_experience_len.remote())
    experience_consumer_stage = "compute_advantage"

    if adv_estimator == "gae":
        experience_columns = ["values", "responses", "token_level_rewards", "response_length"]
        if not use_kl_in_reward:
            experience_columns = ["values", "responses", "rm_scores", "response_length"]
    elif adv_estimator == "group_norm":
        if use_stepwise_advantage:
            experience_columns = [
                "responses",
                "response_length",
                "mc_returns",
                "token_level_rewards",
                "index_in_batch_list",
                "index_in_steps_list",
                "idxs",
                "is_last_step",
            ]
        else:
            experience_columns = ["responses", "rm_scores", "response_length"]
    else:
        raise NotImplementedError

    pad_token_id = tokenizer.pad if tokenizer.pad is not None else tokenizer.eod
    sorted_indexes = (
        get_current_dp_range_indexes(experience_count=experience_count, assign_batch_size=global_batch_size)
        if guarantee_order
        else None
    )
    while not ray.get(td.all_consumed.remote(experience_consumer_stage)):
        batch_data, index = ray.get(
            td.get_experience.remote(
                experience_consumer_stage,
                experience_columns,
                experience_count,
                indexes=sorted_indexes.pop(0) if guarantee_order else None,
            )
        )
        batch_data = remove_padding_tensor_dict_to_dict(batch_data)
        if batch_data and index:
            batch_data = pad_experience(batch_data, pad_token_id)
            response_mask = generate_mask(batch_data["responses"], batch_data["response_length"])
            response_length = batch_data["response_length"]

            if adv_estimator == "gae":
                if use_kl_in_reward:
                    token_level_rewards = batch_data["token_level_rewards"]
                else:
                    rm_scores = batch_data["rm_scores"]
                    reward_tensor = torch.zeros_like(batch_data["responses"], dtype=torch.float32)
                    for i in range(batch_data["responses"].shape[0]):
                        valid_response_length = batch_data["response_length"][i] - 1
                        reward_tensor[i, int(valid_response_length.item())] = rm_scores[i]
                    token_level_rewards = reward_tensor
                values = batch_data["values"]
                advantages, returns = compute_gae_advantage_return(
                    token_level_rewards=token_level_rewards,
                    values=values,
                    eos_mask=response_mask,
                    gamma=gamma,
                    lam=lam,
                )
            elif adv_estimator == "group_norm":
                if use_stepwise_advantage:
                    advantages, returns = _compute_stepwise_group_norm_advantage(
                        batch_data=batch_data,
                        response_mask=response_mask,
                        stepwise_advantage_mode=stepwise_advantage_mode,
                        stepwise_advantage_beta=stepwise_advantage_beta,
                    )
                    _record_advantage_metrics(td, advantages)
                else:
                    token_level_rewards = batch_data["rm_scores"]
                    advantages, returns = compute_group_norm_advantage_return_patch(
                        token_level_rewards=token_level_rewards,
                        eos_mask=response_mask,
                        response_length=response_length,
                        n_sample_per_prompt=n_sample_per_prompt,
                        use_stepwise_advantage=use_stepwise_advantage,
                    )
            else:
                raise NotImplementedError

            advantages = truncate_rows(advantages, batch_data["response_length"])
            returns = truncate_rows(returns, batch_data["response_length"])
            output = {
                "advantages": advantages,
                "returns": returns,
            }
            output = padding_dict_to_tensor_dict(output)
            td.put_experience.remote(data_dict=output, indexes=index)


def _compute_stepwise_group_norm_advantage(
    batch_data,
    response_mask,
    stepwise_advantage_mode,
    stepwise_advantage_beta,
):
    if stepwise_advantage_mode == "mc_return":
        token_level_rewards = batch_data["mc_returns"]
        index_in_batch_list = [
            f"{b.item()}_{s.item()}"
            for b, s in zip(batch_data["index_in_batch_list"], batch_data["index_in_steps_list"])
        ]
    elif stepwise_advantage_mode in ("immediate_reward_group_norm", "immediate_reward_centered_scaled"):
        token_level_rewards = batch_data["token_level_rewards"]
        index_in_batch_list = [
            f"{b.item()}_{s.item()}"
            for b, s in zip(batch_data["index_in_batch_list"], batch_data["index_in_steps_list"])
        ]
    elif stepwise_advantage_mode == "broadcast":
        all_token_level_rewards = batch_data["token_level_rewards"]
        all_index_in_batch_list = [t.item() for t in batch_data["index_in_batch_list"]]
        all_idxs = [t.item() for t in batch_data["idxs"]]
        is_last_step = [t.item() for t in batch_data["is_last_step"]]

        other_step_indices = [i for i, v in enumerate(is_last_step) if not v]
        last_step_indices = [i for i, v in enumerate(is_last_step) if v]

        other_step_response_mask = response_mask[other_step_indices]
        other_step_idxs = [all_idxs[i] for i in other_step_indices]

        token_level_rewards = all_token_level_rewards[last_step_indices]
        response_mask = response_mask[last_step_indices]
        index_in_batch_list = [all_index_in_batch_list[x] for x in last_step_indices]
        idxs = [all_idxs[x] for x in last_step_indices]
    else:
        raise ValueError(f"Unsupported stepwise_advantage_mode: {stepwise_advantage_mode}")

    if stepwise_advantage_mode == "immediate_reward_centered_scaled":
        advantages, returns = compute_centered_scaled_reward_advantage_return_patch(
            token_level_rewards=token_level_rewards,
            index_in_batch_list=index_in_batch_list,
            response_mask=response_mask,
            beta=stepwise_advantage_beta,
        )
    else:
        advantages, returns = compute_group_norm_advantage_return_by_index_patch(
            token_level_rewards=token_level_rewards,
            index_in_batch_list=index_in_batch_list,
            response_mask=response_mask,
        )

    if stepwise_advantage_mode == "broadcast":
        other_step_advantages = _stepwise_advantage_broadcast(
            other_step_response_mask, response_mask, advantages, idxs, other_step_idxs
        )
        combined = advantages.new_empty((len(is_last_step), advantages.size(1)))
        combined[last_step_indices] = advantages
        combined[other_step_indices] = other_step_advantages
        advantages = combined
        returns = combined.clone()

    return advantages, returns


def _record_advantage_metrics(td, advantages, eps=1e-8):
    if isinstance(advantages, list):
        row_abs_sums = [row.abs().sum() for row in advantages]
        nonzero_count = sum(float(row_sum.item()) > eps for row_sum in row_abs_sums)
        total_count = len(row_abs_sums)
        non_empty_rows = [row.abs().float().reshape(-1) for row in advantages if row.numel()]
        if non_empty_rows:
            mean_abs_advantage = float(torch.cat(non_empty_rows).mean().item())
        else:
            mean_abs_advantage = 0.0
    else:
        row_abs_sum = advantages.abs().sum(dim=-1)
        nonzero_mask = row_abs_sum > eps
        total_count = int(nonzero_mask.numel())
        nonzero_count = int(nonzero_mask.sum().item())
        mean_abs_advantage = float(advantages.abs().mean().item()) if advantages.numel() else 0.0

    nonzero_rate = nonzero_count / total_count if total_count else 0.0

    td.update_metrics.remote(
        value={
            "zh_adv/nonzero_sample_rate": [nonzero_rate],
            "zh_adv/mean_abs": [mean_abs_advantage],
        },
        cumulate=True,
    )


def _stepwise_advantage_broadcast(tgt_mask, src_mask, src_advantages, src_indices, tgt_indices):
    """Broadcast advantage from last_step_batch to all other steps."""
    idx_to_scalar_adv = {}
    for i, idx in enumerate(src_indices):
        mask = src_mask[i].bool()
        scalar = src_advantages[i][mask].mean()
        idx_to_scalar_adv[int(idx)] = scalar

    scalar_rows = []
    for i, idx in enumerate(tgt_indices):
        scalar_adv = idx_to_scalar_adv.get(int(idx))
        if scalar_adv is None:
            scalar_rows.append(torch.zeros_like(tgt_mask[i], dtype=torch.float32))
        else:
            scalar_rows.append(torch.full_like(tgt_mask[i], fill_value=scalar_adv, dtype=torch.float32))
    if not scalar_rows:
        return torch.zeros_like(tgt_mask, dtype=torch.float32)
    scalar_rows = torch.stack(scalar_rows)
    return scalar_rows * tgt_mask
