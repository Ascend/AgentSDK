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
from typing import List

import numpy as np
import torch
import torchair._contrib.custom_torch_ops  # type: ignore  # noqa: F401
from vllm.attention.backends.utils import (PAD_SLOT_ID, CommonAttentionState,
                                           CommonMetadataBuilder,
                                           compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.utils import async_tensor_h2d, make_tensor_with_pad
from vllm_ascend.attention.attention import AscendMetadataBuilder, AscendMetadata, AttentionMaskBuilder


def get_splitfuse_attn_mask_patch(
    self,
    seq_lens,
    query_lens,
    position,
    dtype,
    device,
) -> torch.Tensor:
    max_seq_len = max(seq_lens, default=0)
    if max_seq_len <= self._seq_len_cached:
        self.update_attn_cache(max_seq_len, dtype, device)
        if self.attn_mask_cache.numel() > 1 and self.attn_mask_cache[0][1] > 0:
            attn_mask = self.get_attn_mask(max_seq_len, dtype, device)
            attn_mask = attn_mask * -10000
        else:
            attn_mask = self.attn_mask_cache
        return torch.index_select(attn_mask, dim=0, index=position)[:, :max_seq_len]
    
    total_q_len = sum(query_lens)
    attn_mask = torch.zeros((total_q_len, max_seq_len),
                            dtype=dtype,
                            device="cpu")

    current_row = 0
    for i in range(len(query_lens)):
        seq_len = seq_lens[i]
        q_len = query_lens[i]
        context_len = seq_len - q_len

        if context_len < 0:
            raise ValueError(
                f"Context length {context_len} cannot be negative. "
                f"Sequence length: {seq_len}, Query length: {q_len}"
            )
        
        attn_mask[current_row:current_row + q_len, context_len:] = self.splitfuse_mask_value
        right_tensor = attn_mask[current_row:current_row + q_len, context_len:seq_len]
        right_tensor.masked_fill_(
            right_tensor.tril() == self.splitfuse_mask_value, 0)
        current_row += q_len

    return attn_mask.to(device, non_blocking=True)


def build_patch(
    self,
    seq_lens: List[int],
    query_lens: List[int],
    graph_pad_size: int,
) -> AscendMetadata:
    """Build attention metadata with on-device tensors.
    
    This method constructs the metadata required for attention computation,
    handling both prefill and decode phases with support for NPU graph optimization.
    
    Args:
        seq_lens: The maybe padded sequence lengths of the input sequences.
        query_lens: The query lengths of the input sequences.
        graph_pad_size: Padding size for NPU graph optimization. Use -1 to disable.
    
    Returns:
        AscendMetadata object containing all necessary attention metadata.
    """
    for inter_data in self.input_builder.inter_data_list:
        self._add_seq_group(inter_data, self.input_builder.chunked_prefill_enabled)

    device = self.runner.device
    dtype = self.runner.model_config.dtype
    use_npu_graph = graph_pad_size != -1

    max_query_len = max(query_lens)
    max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
    max_decode_seq_len = max(self.curr_seq_lens, default=0)
    max_seq_len = max(max_prefill_seq_len, max_decode_seq_len)
    num_decode_tokens = self.num_decode_tokens

    if self.num_prefills == 0 and use_npu_graph:
        num_seqs = len(seq_lens)
        self.slot_mapping.extend([PAD_SLOT_ID] * graph_pad_size)
        self.block_tables.extend([[]] * graph_pad_size)
        block_tables = self._get_graph_runner_block_tables(num_seqs, self.block_tables)
    else:
        block_tables = make_tensor_with_pad(
            self.block_tables,
            pad=0,
            dtype=torch.int32,
            device=device,
        )

    if self.num_prefills > 0:
        if block_tables is None or block_tables.numel() == 0:
            self.attn_mask = AscendMetadataBuilder._attn_mask_builder.get_attn_mask(  # type: ignore
                max_prefill_seq_len, dtype, device)
        elif self.num_decode_tokens == 0 and not self.input_builder.chunked_prefill_enabled:
            self.compress_mask = AscendMetadataBuilder._attn_mask_builder.get_attn_mask(  # type: ignore
                128, dtype, device)
        else:
            attn_mask = AscendMetadataBuilder._attn_mask_builder.get_attn_mask(  # type: ignore
                max_seq_len, dtype, device)
            if attn_mask.numel() > 1 and attn_mask[0][1] > 0:
                attn_mask = attn_mask * -10000
            chunk_mask_list = []
            for i, seq_len in enumerate(seq_lens):
                context_len = self.context_lens[i]
                chunk_mask_list.append(attn_mask[context_len:seq_len])
            self.chunk_mask = torch.cat(chunk_mask_list, 0)
    else:
        self.attn_mask = None
        self.compress_mask = None
        self.chunk_mask = None

    if max_query_len <= 0:
        raise ValueError(
            f"Maximum query length must be positive, got {max_query_len}. "
            f"Query lengths: {query_lens}"
        )

    if device is None:
        raise RuntimeError("Device is not initialized in runner")

    slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.int32,
                                           device, self.runner.pin_memory)
    seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                       self.runner.pin_memory)
    placeholder_index_maps = {
        modality: placeholder_map.index_map()
        for modality, placeholder_map in
        self.multimodal_placeholder_maps.items()
    }

    return AscendMetadata(
        num_prefills=self.num_prefills,
        slot_mapping=slot_mapping_tensor,
        num_prefill_tokens=self.num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        seq_lens=seq_lens,
        multi_modal_placeholder_index_maps=placeholder_index_maps,
        enable_kv_scales_calculation=True,
        seq_lens_tensor=seq_lens_tensor,
        query_lens=query_lens,
        max_query_len=max_query_len,
        max_prefill_seq_len=max_prefill_seq_len,
        max_decode_seq_len=max_decode_seq_len,
        block_tables=block_tables,
        attn_mask=self.attn_mask,
        compress_mask=self.compress_mask,
        chunk_mask=self.chunk_mask,
        chunked_prefill_enabled=self.input_builder.chunked_prefill_enabled,
    )


AttentionMaskBuilder.get_splitfuse_attn_mask = get_splitfuse_attn_mask_patch
AscendMetadataBuilder.build = build_patch
