#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def no_padding_2_padding_patch(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """Slice response from unpad model output.

    Args:
        tensor: a nested tensor or a 1D tensor in shape (total_nnz,),
            total_nnz is the total number of tokens across all sequences in the batch
        data: TensorDict with "prompts", "responses", "attention_mask"

    Returns:
        tensor: sliced response tensor of shape [bsz, max_response_len]
    """
    logger.debug("no_padding_2_padding_patch")
    values = tensor.values() if tensor.is_nested else tensor
    prompt_ids = data["prompts"]
    response_ids = data["responses"]
    attention_mask = data["attention_mask"]

    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)

    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        if max_response_len < 0:
            max_response_len = response_lens.max().item()
    else:
        if attention_mask.is_nested:
            raise ValueError("attention_mask should not be nested")
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_response_len = response_ids.shape[1]

    sequence_lens = prompt_lens + response_lens
    sequence_offsets = sequence_lens.cumsum(dim=0)
    ## [Aura feature sp] patch begin
    import os

    sp_size = int(os.getenv("ULYSSES_SEQUENCE_PARALLEL_SIZE", 1))
    logger.info(f"===offsets: {sequence_offsets[-1].item()}, values: {values.shape[0]}, sp_size: {sp_size}")
    if values.shape[0] > sequence_offsets[-1].item():
        ratio = round(values.shape[0] / sequence_offsets[-1].item())
        logger.info(f"===ratio: {ratio}")
        if ratio == sp_size:
            sequence_offsets = sequence_offsets * ratio
            sequence_offsets[-1] = values.shape[0]
    ## [Aura feature sp] patch end
    if sequence_offsets[-1].item() != values.shape[0]:
        raise ValueError(
            f"sequence_offsets[-1] ({sequence_offsets[-1].item()}) must equal values.shape[0] ({values.shape[0]})"
        )

    response_list = []
    for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
        pad_size = max_response_len - resp_len
        # left-shift model output by one token for log_probs/values
        response_list.append(F.pad(values[seq_offset - resp_len - 1 : seq_offset - 1], (0, pad_size)))

    output = torch.stack(response_list, dim=0)
    return output
