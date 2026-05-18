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
from .compute_utils_patch import compute_group_norm_advantage_return_patch


def compute_advantage(self, blocking=False, guarantee_order=False):
    experience_count = self.micro_batch_size

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
        n_sample_per_prompt=self.actor_worker.rl_config.n_samples_per_prompt,
        use_stepwise_advantage=self.actor_worker.rl_config.use_stepwise_advantage,
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
    use_kl_in_reward=False,
):
    """
    Compute the advantage function based on different adv_estimator

    Args:
        td: A data queue object
        gamma: The reward discount factor
        lam: The lambda parameter in advantage estimation
        adv_estimator:  The type of advantage estimator, which can be "gae" or "group_norm"
        experience_count: The number of experiences to retrieve from the experience td
        tokenizer: The pre-trained tokenizer
        global_batch_size: The number of global batch size
        guarantee_order: The switch of guarantee order

    Returns:
        None
    """
    experience_count = ray.get(td.get_experience_len.remote())
    experience_consumer_stage = "compute_advantage"
    if adv_estimator == "gae":
        experience_columns = ["values", "responses", "token_level_rewards", "response_length"]
        if not use_kl_in_reward:
            experience_columns = ["values", "responses", "rm_scores", "response_length"]
    else:
        experience_columns = ["responses", "rm_scores", "response_length"]
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
                experience_count,  # pad_id=pad_token_id
                indexes=sorted_indexes.pop(0) if guarantee_order else None,
            )
        )
        batch_data = remove_padding_tensor_dict_to_dict(batch_data)
        if batch_data and index:
            batch_data = pad_experience(batch_data, pad_token_id)  # multiple, tp_size
            response_mask = generate_mask(batch_data["responses"], batch_data["response_length"])
            response_length = batch_data["response_length"]
            if adv_estimator == "gae":
                if use_kl_in_reward:
                    token_level_rewards = batch_data["token_level_rewards"]
                else:
                    rm_scores = batch_data["rm_scores"]
                    reward_tensor = torch.zeros_like(batch_data['responses'], dtype=torch.float32)
                    for i in range(batch_data['responses'].shape[0]):
                        valid_response_length = batch_data['response_length'][i] - 1
                        reward_tensor[i, int(valid_response_length.item())] = rm_scores[i]
                    token_level_rewards = reward_tensor
                values = batch_data["values"]
                advantages, returns = compute_gae_advantage_return(
                    token_level_rewards=token_level_rewards, values=values, eos_mask=response_mask, gamma=gamma, lam=lam
                )
            elif adv_estimator == "group_norm":
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
            advantages = truncate_rows(advantages, batch_data['response_length'])
            returns = truncate_rows(returns, batch_data['response_length'])
            output = {
                "advantages": advantages,
                "returns": returns,
            }
            output = padding_dict_to_tensor_dict(output)
            td.put_experience.remote(data_dict=output, indexes=index)
