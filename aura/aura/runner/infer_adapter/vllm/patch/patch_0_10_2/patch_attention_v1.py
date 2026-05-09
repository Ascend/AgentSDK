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

from typing import Optional

import torch

import torch_npu
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, aligned_16, is_310p,
                               nd_to_nz_2d, nd_to_nz_spec)
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendMetadata


def _forward_prefill_no_cache_patch(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: AscendMetadata,
    output: Optional[torch.Tensor] = None,
    num_tokens: int = 0,
) -> torch.Tensor:
    if attn_metadata is None:
        raise RuntimeError("Attention metadata cannot be None for prefill without cache")
    
    if attn_metadata.attn_mask is None:
        raise RuntimeError("Attention mask cannot be None for prefill without cache")

    mask = attn_metadata.attn_mask

    if is_310p():
        query = aligned_16(query)
        key = aligned_16(key)
        value = aligned_16(value)
        output = aligned_16(output)
        mask = mask.repeat(attn_metadata.seq_lens.size(0), 1, 1, 1)
        mask = torch_npu.npu_format_cast(mask.contiguous(), ACL_FORMAT_FRACTAL_NZ)

    if self.sliding_window is not None and \
            attn_metadata.attn_mask.shape[0] > self.sliding_window:
        key = self._repeat_kv(key, self.num_heads // self.num_kv_heads)
        value = self._repeat_kv(value, self.num_heads // self.num_kv_heads)

        output, _ = torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            pre_tokens=self.sliding_window,
            scale=self.scale,
            actual_seq_lengths=attn_metadata.seq_lens,
            actual_seq_lengths_kv=attn_metadata.seq_lens)
        output = output.view(num_tokens, self.num_heads, self.head_size)
    else:
        torch_npu._npu_flash_attention(
            query=query,
            key=key,
            value=value,
            mask=(mask != 0).long().to(query.dtype),
            seq_len=attn_metadata.seq_lens,
            scale_value=self.scale,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            out=output)
    
    if output is None:
        raise RuntimeError("Output tensor is None after attention computation")
    
    return output[:num_tokens, :, :]


AscendAttentionBackendImpl._forward_prefill_no_cache = _forward_prefill_no_cache_patch
