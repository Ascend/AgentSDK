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
        raise ValueError("splitfuse_attn_mask now only supports bf16 and fp16")

    max_seq_len = max(seq_lens, default=0)
    self._update_attn_cache(max_seq_len, dtype)
    mask_scale_factor = AttentionMaskBuilder.get_mask_scale_factor(dtype)
    attn_mask = torch.index_select(self.attn_mask_cache, dim=0, index=position)[:, :max_seq_len]
    attn_mask = attn_mask * mask_scale_factor
    return attn_mask.contiguous().to(device, non_blocking=True)


AttentionMaskBuilder.get_splitfuse_attn_mask = get_splitfuse_attn_mask_patch
