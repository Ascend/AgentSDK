# -*- coding: utf-8 -*-
#
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
#
from typing import Dict, List, Tuple

from .seqlen_balancing_patch import rearrange_micro_batches_patch


def _split_batches_with_dynamic_bsz(
    self,
    batch: Dict,
    max_packing_token: int,
    dynamic_max_batch_size: int,
) -> Tuple[List[Dict], List[List[int]]]:
    """Split a batch into micro-batches using dynamic sequence-length balancing.

    Args:
        batch: Full training batch keyed by tensor name.
        max_packing_token: Maximum token budget per micro-batch.
        dynamic_max_batch_size: Upper bound on samples per micro-batch.

    Returns:
        A tuple of (micro-batches, partition index lists).
    """
    seq_len_list = []
    for prompt_len, response_len in zip(batch['prompt_length'], batch['response_length']):
        seq_len_list.append(prompt_len.item() + response_len.item())

    partitions = rearrange_micro_batches_patch(
        seq_len_list, max_packing_token,
        dynamic_max_batch_size=dynamic_max_batch_size,
    )
    batches: List[Dict] = []
    for key, tensors in batch.items():
        for batch_idx, partition in enumerate(partitions):
            if batch_idx >= len(batches):
                batches.append({})
            batches[batch_idx][key] = tensors[partition]
    return batches, partitions