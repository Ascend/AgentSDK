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


from typing import List
import torch

def optimized_preprocess_input(batch):
    mini_batches = {}
    prompt_ids = {}
    for idx, mini_id in enumerate(batch["mini_batch_id"]):
        if mini_id not in mini_batches:
            mini_batches[mini_id] = [batch["input_ids"][idx]]
            prompt_ids[mini_id] = [batch["prompt_id"][idx]]
        else:
            mini_batches[mini_id].append(batch["input_ids"][idx])
            prompt_ids[mini_id].append(batch["prompt_id"][idx])
    return mini_batches, prompt_ids

def optimized_put_prompt_experience(
        mini_batch: List[torch.Tensor],
        prompts_ids: List[int],
        dict_to_tensor_dict=None
):
    prompts_length = []
    for batch in mini_batch:
        prompts_length.append(torch.tensor([len(batch)]))

    indexes = [i for i in range(len(prompts_length))]

    new_prompts_ids = []
    for ids in prompts_ids:
        new_prompts_ids.append(torch.tensor(ids))

    data_dict = dict(
        {"prompt_length": prompts_length, "prompts": mini_batch},
    )
    if dict_to_tensor_dict is not None:
        return dict_to_tensor_dict(data_dict), indexes
    return data_dict, indexes