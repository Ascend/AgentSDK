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

from typing import Dict, List, Union
import torch
from torch import Tensor
from tensordict import TensorDict


def padding_dict_to_tensor_dict(experience_data: Dict[str, Union[Tensor, List[Tensor]]]):
    if not experience_data:
        raise ValueError("experience_data cannot be empty")
    if not isinstance(experience_data, dict):
        raise TypeError("experience_data must be a dictionary")
    if not all(isinstance(key, str) for key in experience_data.keys()):
        raise TypeError("all keys in experience_data must be strings")
    if not (all(isinstance(value, Tensor) or isinstance(value, list) for value in experience_data.values())):
        raise TypeError("experience_data value must be Tensor or List[Tensor]")

    experience_batch = {}
    experience_data_length = []
    for experience_column, value in experience_data.items():
        if isinstance(value, list) and not (all(isinstance(v, Tensor) for v in value)):
            raise TypeError("All elements in the list must be Tensor.")
        elif isinstance(value, Tensor):
            if value.ndim == 0:
                raise ValueError("Tensor is scalar (0D) and cannot be processed.")
            elif value.ndim == 1:
                value = value.unsqueeze(0)  # (D) → (1, D)
            value = [value[i] for i in range(value.size(0))]  # List[Tensor]

        max_length = max(len(exp) for exp in value)
        padded_tensors = [torch.nn.functional.pad(exp, (0, max_length - len(exp)),
                                                  mode='constant', value=0) for exp in value]
        experience_batch[experience_column] = torch.stack(padded_tensors, dim=0)
        experience_data_length.extend([torch.tensor(len(exp)) for exp in value])
    experience_batch['original_length'] = torch.stack(experience_data_length)
    experience_batch = TensorDict.from_dict(experience_batch)
    return experience_batch