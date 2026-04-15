#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from typing import Dict, List, Union

import torch
from tensordict import TensorDict
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from mindspeed_rl.utils.utils import mstx_timer_decorator


def padding_dict_to_tensor_dict(experience_data: Dict[str, Union[Tensor, List[Tensor]]]):
    experience_batch = {}
    experience_data_length = []
    for experience_column, value in experience_data.items():
        max_length = max(len(exp) for exp in value)
        padded_tensors = [torch.nn.functional.pad(exp, (0, max_length - len(exp)),
                                                  mode='constant', value=0) for exp in value]
        experience_batch[experience_column] = torch.stack(padded_tensors, dim=0)
        experience_data_length.extend([torch.tensor(len(exp)) for exp in value])
    experience_batch['original_length'] = torch.stack(experience_data_length)
    experience_batch = TensorDict.from_dict(experience_batch)
    return experience_batch


@mstx_timer_decorator
def padding_dict_to_tensor_dict_fast(
        experience_data: Dict[str, List[Tensor]]
) -> TensorDict:
    """
    • Uses `pad_sequence` (C-kernel) once per column.
    • Allocates the final tensor once, instead of calling `F.pad` per row.
    • Stores `original_length` as `(batch,)` int32 tensor – cheaper than duplicates.
    """
    out: Dict[str, Tensor] = {}
    original_lens: List[Tensor] = []

    for col, seqs in experience_data.items():
        # Normalise: all items must be torch.Tensor
        if (not isinstance(seqs, (list, tuple))) or (not torch.is_tensor(seqs[0])):
            raise TypeError(
                f"Column '{col}' must be a list of torch.Tensor, "
                f"but got {type(seqs).__name__} containing "
                f"{set(type(s).__name__ for s in seqs)}"
            )

        # Make sure every seq is at least 1-D
        seqs = [s.unsqueeze(0) if s.ndim == 0 else s for s in seqs]
        # Keep per-row lengths
        lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.int32)
        original_lens.append(lengths)

        # Pad – batch_first=True gives (B, T_max)
        out[col] = pad_sequence(seqs, batch_first=True, padding_value=0)

    out["original_length"] = torch.cat(original_lens, dim=0)
    return TensorDict.from_dict(out)
