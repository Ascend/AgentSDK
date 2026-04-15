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

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch_npu
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm_ascend.ops.attention import vanilla_chunked_prefill
from vllm_ascend.utils import get_graph_params

from vllm_ascend.attention.attention_v1 import AscendMetadata, AscendAttentionBackendImpl, AscendAttentionState


def forward_patch(
    self,
    layer: AttentionLayer,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AscendMetadata,
    output: Optional[torch.Tensor] = None,
    trace_flag: bool = True,
) -> torch.Tensor:
    """Forward pass with Ascend attention."""
    num_tokens = query.shape[0]
    if output is None:
        output = torch.empty(num_tokens,
                             self.num_heads,
                             self.head_size,
                             dtype=query.dtype,
                             device=query.device)
    
    if trace_flag:
        torch.ops.vllm.unified_ascend_attention_with_output(
            query=query,
            key=key,
            value=value,
            output=output,
            layer_name=layer.layer_name)
    else:
        if attn_metadata is None:
            return output.view(num_tokens, self.hidden_size)
        
        num_actual_tokens = attn_metadata.num_actual_tokens
        
        if not (layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0):
            raise ValueError(
                f"Key scale {layer._k_scale_float} and value scale {layer._v_scale_float} "
                "must both be 1.0 for Ascend attention backend"
            )
        
        attn_type = self.attn_type
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention "
                "are not implemented for PallasAttentionBackendImpl"
            )
        
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        value = value.contiguous()

        if len(kv_cache) > 0:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping
            torch_npu._npu_reshape_and_cache(
                key=key[:num_actual_tokens],
                value=value[:num_actual_tokens],
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                slot_indices=slots)

        if hasattr(layer, 'quant_method'):
            pass
        elif attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            if attn_metadata is None:
                raise RuntimeError("Attention metadata cannot be None for PrefillNoCache state")
            if attn_metadata.attn_mask is None:
                raise RuntimeError("Attention mask cannot be None for PrefillNoCache state")
            
            mask = attn_metadata.attn_mask
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
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            if attn_metadata is None:
                raise RuntimeError("Attention metadata cannot be None for PrefillCacheHit state")
            if attn_metadata.attn_mask is None:
                raise RuntimeError("Attention mask cannot be None for PrefillCacheHit state")
            
            compress_mask = attn_metadata.attn_mask
            batch_size = attn_metadata.query_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            torch_npu._npu_flash_attention_qlens(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                block_table=block_table,
                mask=compress_mask,
                seq_len=attn_metadata.query_lens,
                context_lens=attn_metadata.seq_lens,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                out=output)
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            graph_params = get_graph_params()
            forward_context = get_forward_context()
            
            if not forward_context.capturing:
                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output)
            else:
                stream = torch_npu.npu.current_stream()
                event = torch.npu.ExternalEvent()
                event.wait(stream)
                event.reset(stream)
                graph_params.events[num_tokens].append(event)

                graph_params.attn_params[num_tokens].append((
                    query,
                    self.key_cache,
                    self.value_cache,
                    self.num_kv_heads,
                    self.num_heads,
                    self.scale,
                    attn_metadata.block_tables,
                    attn_metadata.seq_lens,
                    output,
                ))

                torch.npu.graph_task_group_begin(stream)
                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output)
                handle = torch.npu.graph_task_group_end(stream)
                graph_params.handles[num_tokens].append(handle)
        else:
            if self.head_size == 192:
                cu_seqlen_q = [0] + attn_metadata.query_lens.tolist()
                cu_seqlen_k = [0] + attn_metadata.seq_lens.tolist()
                cu_seqlen_q = torch.tensor(cu_seqlen_q, device="npu")
                cu_seqlen_k = torch.tensor(cu_seqlen_k, device="npu")
                cu_seqlen_q = torch.cumsum(cu_seqlen_q, dim=0)
                cu_seqlen_k = torch.cumsum(cu_seqlen_k, dim=0)
                max_seqlen_q = torch.max(attn_metadata.query_lens)
                max_seqlen_k = torch.max(attn_metadata.seq_lens)
                vanilla_chunked_prefill(
                    output, query, self.key_cache,
                    self.value_cache,
                    attn_metadata.block_tables,
                    cu_seqlen_q, cu_seqlen_k,
                    max_seqlen_q, max_seqlen_k,
                    self.scale, None, True)
            else:
                torch_npu._npu_paged_attention_splitfuse(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    mask=attn_metadata.attn_mask,
                    block_table=attn_metadata.block_tables,
                    seq_len=attn_metadata.query_lens,
                    context_lens=attn_metadata.seq_lens,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    out=output)
    
    return output.view(num_tokens, self.hidden_size)


AscendAttentionBackendImpl.forward = forward_patch
