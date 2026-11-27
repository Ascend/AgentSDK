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

import os
from typing import Optional, Tuple

import torch
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

original_forward_native = RotaryEmbedding.forward_native


def forward_native_patch(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if offsets is None and os.getenv('ENABLE_ATB_ROPE', "False").lower() == "true":
        query_shape, key_shape = query.shape, key.shape
        if self.cos_sin_cache.device != query.device:
            self.cos_sin_cache = self.cos_sin_cache.to(query.device)
        if self.cos_sin_cache.dtype != query.dtype:
            self.cos_sin_cache = self.cos_sin_cache.to(query.dtype)
        neox_style = self.is_neox_style

        import torch_npu
        if self.rotary_dim < self.head_size:
            num_tokens = query.shape[0]
            query = query.view(num_tokens, -1, self.head_size)
            key = key.view(num_tokens, -1, self.head_size)
            q_rot = query[..., :self.rotary_dim]
            q_pass = query[..., self.rotary_dim:]
            k_rot = key[..., :self.rotary_dim]
            k_pass = key[..., self.rotary_dim:]
            q_rot = q_rot.contiguous().view(num_tokens, -1)
            k_rot = k_rot.contiguous().view(num_tokens, -1)
            torch_npu._npu_rotary_embedding(
                positions,
                q_rot,
                k_rot,
                self.head_size,
                self.cos_sin_cache,
                neox_style,
            )
            q_rot = q_rot.view(num_tokens, -1, self.rotary_dim)
            k_rot = k_rot.view(num_tokens, -1, self.rotary_dim)
            q = torch.cat((q_rot, q_pass), dim=-1).reshape(query_shape)
            k = torch.cat((k_rot, k_pass), dim=-1).reshape(key_shape)
            return q, k

        query = query.contiguous().view(query.shape[0], -1)
        key = key.contiguous().view(key.shape[0], -1)
        torch_npu._npu_rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            neox_style,
        )
        return query.view(query_shape), key.view(key_shape)

    return original_forward_native(self, positions, query, key, offsets)


RotaryEmbedding.forward_native = forward_native_patch
