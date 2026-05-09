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
