#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
from mindspeed_rl.utils.compute import compute_log_probs, vocab_parallel_entropy
from mindspeed_rl.utils.context_parallel import (
    get_tensor_allgather_cp_without_pack,
    get_tensor_allgather_cp_with_pack,
)
from mindspeed_rl.utils.compute import get_parallel_state
from mindspeed_rl.utils.remove_padding import postprocess_packed_seqs


def compute(self, output, batch, skip_entropy, **kwargs):
    # Not skip entropy when entropy_coeff is 0
    use_remove_padding = kwargs.get("use_remove_padding", False)
    index = kwargs.get("index", None)
    labels = batch["labels"]

    log_probs = compute_log_probs(output, labels)

    cp_size = get_parallel_state().get_context_parallel_world_size()

    if use_remove_padding:
        log_probs_allgather = get_tensor_allgather_cp_with_pack(
            log_probs, cp_size, index
        )
        seqlens_in_batch = kwargs.get("seqlens_in_batch", None)
        cu_seqlens_padded = kwargs.get("cu_seqlens_padded", None)
        seq_len = batch["responses"].shape[-1]
        log_probs = postprocess_packed_seqs(
            log_probs_allgather,
            seqlens_in_batch,
            cu_seqlens_padded,
            seq_len,
            prompt_length=batch["prompt_length"],
        )
        entropy = vocab_parallel_entropy(output)
        entropy = postprocess_packed_seqs(
            entropy,
            seqlens_in_batch,
            cu_seqlens_padded,
            seq_len,
            prompt_length=batch["prompt_length"],
        )

        return log_probs, entropy

    else:
        log_probs_allgather = get_tensor_allgather_cp_without_pack(
            log_probs, cp_size, index
        )
        log_probs = self._get_log_probs_remove_prompt_pad(log_probs_allgather, batch)
        entropy = vocab_parallel_entropy(output)
        entropy_allgather = get_tensor_allgather_cp_without_pack(
            entropy, cp_size, index
        )
        entropy = self._get_log_probs_remove_prompt_pad(entropy_allgather, batch)


        return log_probs, entropy