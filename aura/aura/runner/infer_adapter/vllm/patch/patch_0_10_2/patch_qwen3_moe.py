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

# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union

from vllm.forward_context import get_forward_context
from vllm_ascend.ops.sequence_parallel import (MetadataForPadding,
                                               init_metadata_for_sp)
from vllm_ascend.models.qwen3_moe import CustomSparseMoeBlock


def custom_sparse_moe_block_forward(
    self,
    hidden_states,
    attn_metadata=None,
    _metadata_for_padding: Optional[MetadataForPadding] = None,
):
    if attn_metadata is None:
        attn_metadata = get_forward_context().attn_metadata
    
    enable_force_load_balance = False
    is_prefill = get_forward_context().with_prefill

    router_logits, _ = self.gate(hidden_states)

    hidden_states = self.experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        is_prefill=is_prefill,
        top_k=self.top_k,
        enable_force_load_balance=enable_force_load_balance,
        shared_experts=None,
        _metadata_for_padding=_metadata_for_padding,
    )

    return hidden_states


CustomSparseMoeBlock.forward = custom_sparse_moe_block_forward
