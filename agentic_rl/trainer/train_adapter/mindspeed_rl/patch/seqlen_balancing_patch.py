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
import heapq
from typing import List, Optional

import torch
import torch.distributed as dist
from mindspeed_rl.utils.seqlen_balancing import karmarkar_karp

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def rearrange_micro_batches_patch(
    seqlen_list: List[int],
    max_token_len: int,
    dynamic_max_batch_size: Optional[int] = None,
    dp_group=None,
) -> List[List[int]]:
    """Rearrange micro-batches to minimise training device memory.

    Args:
        seqlen_list: Sequence lengths of each item.
        max_token_len: Maximum token length per micro-batch.
        dynamic_max_batch_size: Optional upper bound on batch size.
        dp_group: Data-parallel process group.

    Returns:
        Partitions as lists of item indices.
    """
    new_seq_len_list = [
        max_token_len if slen > max_token_len else slen
        for slen in seqlen_list
    ]
    partitions = rearrange_micro_batches_raw(
        new_seq_len_list, max_token_len,
        dynamic_max_batch_size=dynamic_max_batch_size,
    )
    num_bs = len(partitions)
    real_length = [
        sum(seqlen_list[idx] for idx in part)
        for part in partitions
    ]
    overflow_count = sum(1 for length in real_length if length > max_token_len)

    if overflow_count == 0:
        return partitions

    new_idx = sorted(range(num_bs), key=lambda idx: -real_length[idx])

    final_idx: List[int] = []
    rep = (num_bs - overflow_count) // overflow_count
    if rep > 0:
        tail = num_bs
        for count_idx in range(overflow_count):
            final_idx.append(new_idx[count_idx])
            final_idx.extend(new_idx[tail - rep: tail])
            tail -= rep
        if num_bs % overflow_count:
            final_idx.extend(
                new_idx[overflow_count: overflow_count + num_bs % overflow_count]
            )
    else:
        for batch_idx in range(num_bs - overflow_count):
            final_idx.append(new_idx[batch_idx])
            final_idx.append(new_idx[num_bs - batch_idx])
        final_idx.extend(new_idx[num_bs - overflow_count: overflow_count])

    partitions = [partitions[idx] for idx in final_idx]

    if torch.distributed.get_rank() == 0:
        batch_tokens = [
            sum(seqlen_list[idx] for idx in part)
            for part in partitions
        ]
        logger.info(
            f"rank {torch.distributed.get_rank()} seqlen_list: {seqlen_list}, "
            f"partitions: {partitions}, batch_tokens: {batch_tokens}"
        )

    return partitions


def rearrange_micro_batches_raw(
    seqlen_list: List[int],
    max_token_len: int,
    dynamic_max_batch_size: Optional[int] = None,
    dp_group=None,
) -> List[List[int]]:
    """Get order of seq lengths to make partitions balanced.

    Used to balance the sum of sequence lengths across dp ranks
    and micro batches.

    Args:
        seqlen_list: Sequence lengths of each item.
        max_token_len: Maximum allowed token length per partition.
        dynamic_max_batch_size: Optional upper bound on batch size.
        dp_group: Data-parallel process group.

    Returns:
        Partitions list containing the index of items.

    Raises:
        ValueError: If any sequence exceeds max_token_len.
    """
    if max(seqlen_list) > max_token_len:
        raise ValueError(
            f"seqlen of items:[{max(seqlen_list)}] must <= max_token_len:[{max_token_len}]"
        )

    total_sum_of_seqlen = sum(seqlen_list)
    k_partitions = (total_sum_of_seqlen + max_token_len - 1) // max_token_len

    # AgenticRL: fix bug which is fixed in newer MindspeedRL versions
    partitions = heapq_partition_with_max(
        seqlen_list=seqlen_list,
        k_partitions=k_partitions,
        max_token_len=max_token_len,
    )
    k_partitions = torch.tensor(len(partitions))

    if dynamic_max_batch_size is not None:
        k_partitions = max(
            k_partitions,
            (len(seqlen_list) + dynamic_max_batch_size - 1) // dynamic_max_batch_size,
        )

    if dist.is_initialized():
        k_partitions = torch.tensor([k_partitions], device='npu')
        dist.all_reduce(k_partitions, op=dist.ReduceOp.MAX, group=dp_group)
        k_partitions = k_partitions.cpu().item()

    partitions = karmarkar_karp(
        seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=False,
    )
    return partitions


def heapq_partition_with_max(
    seqlen_list: List[int],
    k_partitions: int,
    max_token_len: int,
) -> List[List[int]]:
    """Partition sequences into groups using a min-heap with a max-token cap.

    Args:
        seqlen_list: Sequence lengths of each item.
        k_partitions: Initial number of partitions.
        max_token_len: Maximum allowed token sum per group.

    Returns:
        Partitions as lists of original item indices.
    """
    sorted_seqlen = sorted(
        [(seqlen, seq_idx) for seq_idx, seqlen in enumerate(seqlen_list)],
        reverse=True,
    )

    # Each group: [current_sum, element_count, group_id, element_indices]
    groups = [[0, 0, group_idx, []] for group_idx in range(k_partitions)]
    group_num = len(groups)
    heapq.heapify(groups)

    partitions: List[List[int]] = []
    for seqlen, seq_idx in sorted_seqlen:
        current_group = heapq.heappop(groups)

        if current_group[0] + seqlen > max_token_len:
            partitions.append(current_group[3])
            new_group = [seqlen, 1, group_num, [seq_idx]]
            group_num += 1
            heapq.heappush(groups, new_group)
        else:
            current_group[0] += seqlen
            current_group[1] += 1
            current_group[3].append(seq_idx)
            heapq.heappush(groups, current_group)

    partitions.extend([group[3] for group in groups])
    return partitions