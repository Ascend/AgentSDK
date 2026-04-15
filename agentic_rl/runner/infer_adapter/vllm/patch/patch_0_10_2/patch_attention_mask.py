#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder


def get_splitfuse_attn_mask_patch(
    self,
    seq_lens: torch.Tensor,
    position: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if dtype not in [torch.float16, torch.bfloat16]:
        raise ValueError(
            f"splitfuse_attn_mask now only supports bf16 and fp16"
        )
    
    max_seq_len = max(seq_lens, default=0)
    self._update_attn_cache(max_seq_len, dtype)
    mask_scale_factor = AttentionMaskBuilder.get_mask_scale_factor(dtype)
    attn_mask = torch.index_select(self.attn_mask_cache, dim=0, index=position)[:, :max_seq_len]
    attn_mask = attn_mask * mask_scale_factor
    return attn_mask.contiguous().to(device, non_blocking=True)


AttentionMaskBuilder.get_splitfuse_attn_mask = get_splitfuse_attn_mask_patch
