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

from typing import Dict, List

from mindspeed_rl.utils.utils import append_to_dict
from mindspeed_rl.utils.compute import get_parallel_state


def update_mini_batch_size(self, original_n_samples_per_prompt: int, new_samples_per_prompt: int,
                           use_stepwise_advantage: bool = False):
    """
    Update mini_batch_size_per_dp, the size of the mini-batch for each data parallel stage.
    While step mode, data batch length will be changed after padding, we need to compute mini_batch_size_per_dp again.
    """
    if use_stepwise_advantage:
        if original_n_samples_per_prompt <= 0:
            raise ValueError(
                "original_n_samples_per_prompt must be a integer greater than 0.")
        self.mini_batch_size_per_dp_new_size = (self.mini_batch_size_per_dp // original_n_samples_per_prompt *
                                                new_samples_per_prompt)
    else:
        self.mini_batch_size_per_dp_new_size = self.mini_batch_size_per_dp


def update(self, data: Dict, kl_ctrl=None) -> Dict:
    """
    Model Reverse Update
    :param data: Reverse update data
    :param kl_ctrl: KL divergence calculation controller
    :return: Model reverse calculation result.
    """
    self.kl_ctrl = kl_ctrl
    metrics = {}
    grad_norm_list = []
    for k, v in data.items():
        if v is not None:
            if isinstance(v, List):
                data[k] = [t.to(next(self.model[0].parameters()).device) for t in v]
            else:
                data[k] = v.to(next(self.model[0].parameters()).device)
    # batch_size should be new mini_batch_size_per_dp while step mode
    mini_batches = self._split_batches(data, batch_size=self.mini_batch_size_per_dp_new_size,
                                       shuffle_mini_batch=self.shuffle_mini_batch, dim=0, keep_list=True)
    for model_module in self.model:
        model_module.train()
    for _ in range(self.epochs):
        for mini_batch in mini_batches:
            for model_chunk in self.model:
                model_chunk.zero_grad_buffer()
            self.optimizer.zero_grad()
            metric_micro_batch = self._forward_backward_batch(mini_batch)
            update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

            if update_successful:
                data_parallel_world_size = get_parallel_state().get_data_parallel_world_size()
                increment = self.mini_batch_size_per_dp_new_size * data_parallel_world_size
                self.opt_param_scheduler.step(increment=increment)
            grad_norm_list.append(grad_norm)

            for metric in metric_micro_batch:
                append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

    metrics[f"{self.role}/grad_norm"] = grad_norm_list

    return metrics
