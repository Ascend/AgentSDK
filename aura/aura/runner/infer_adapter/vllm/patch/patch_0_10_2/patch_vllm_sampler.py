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
import os
from vllm.v1.sample.sampler import Sampler
from typing import Optional
from vllm.config import LogprobsMode
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from ..comm.vllm_execute_stat import StatTimeUtil, vllm_output_statics, StatPhase

_SAMPLING_EPS = 1e-5


def forward_patch(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> SamplerOutput:
    # NOTE(woosuk): Use the original logits (before any penalties or
    # temperature scaling) for the top-k logprobs.
    # This is different from the V0 sampler, which uses the logits that
    # is used for sampling (after penalties and temperature scaling).
    time_util_sample = StatTimeUtil()
    num_logprobs = sampling_metadata.max_num_logprobs
    if num_logprobs is not None:
        if self.logprobs_mode == LogprobsMode.RAW_LOGPROBS:
            raw_logprobs = self.compute_logprobs(logits)
        elif self.logprobs_mode == LogprobsMode.RAW_LOGITS:
            raw_logprobs = logits.clone()
    vllm_output_statics.add_stat(StatPhase.post_samper_compute_logprobs_time, time_util_sample.get_duration())

    # Use float32 for the logits.
    logits = logits.to(torch.float32)
    # Apply allowed token ids.
    logits = self.apply_allowed_token_ids(logits, sampling_metadata)
    # Apply bad words exclusion.
    logits = self.apply_bad_words(logits, sampling_metadata)
    vllm_output_statics.add_stat(StatPhase.post_samper_logits_preproc_time, time_util_sample.get_duration())

    # print("===hucuihua Sampler forward, sampling_metadata.logitsprocs.non_argmax_invariant:", sampling_metadata.logitsprocs.non_argmax_invariant)
    # Apply logits processors which can impact greedy sampling
    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        logits = processor.apply(logits)
    vllm_output_statics.add_stat(StatPhase.post_samper_processor_apply_time, time_util_sample.get_duration())

    # Apply penalties (e.g., min_tokens, freq_penalties).
    logits = self.apply_penalties(logits, sampling_metadata)
    vllm_output_statics.add_stat(StatPhase.post_samper_apply_penalties_time, time_util_sample.get_duration())

    # Sample the next token.
    sampled, processed_logprobs = self.sample(logits, sampling_metadata)
    vllm_output_statics.add_stat(StatPhase.post_samper_sample_next_token_time, time_util_sample.get_duration())
    # print("===hucuihua Sampler forward after self.sample, sampled shape:", sampled.shape, " processed_logprobs:", processed_logprobs)
    if processed_logprobs is not None:
        raw_logprobs = processed_logprobs
    # Convert sampled token ids to int64 (long) type to ensure compatibility
    # with subsequent operations that may use these values as indices.
    # This conversion is necessary because FlashInfer sampling operations
    # return int32 (while PyTorch argmax and topk return int64).
    sampled = sampled.long()
    vllm_output_statics.add_stat(StatPhase.post_samper_sampled_long_time, time_util_sample.get_duration())

    # Gather the logprobs of the topk and sampled token (if requested).
    # Get logprobs and rank tensors (if requested)
    logprobs_tensors = (
        None if num_logprobs is None else self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)
    )
    vllm_output_statics.add_stat(StatPhase.post_samper_gather_logprobs_time, time_util_sample.get_duration())

    # Use int32 to reduce the tensor size.
    sampled = sampled.to(torch.int32)
    vllm_output_statics.add_stat(StatPhase.post_samper_sampled_int32_time, time_util_sample.get_duration())

    # These are GPU tensors.
    sampler_output = SamplerOutput(
        # The sampled tokens are expanded to 2D tensor with shape
        # [num_requests, 1], where each row represents one generated
        # token per request.
        sampled_token_ids=sampled.unsqueeze(-1),
        logprobs_tensors=logprobs_tensors,
    )
    return sampler_output


def sample_patch(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sample logits based on sampling metadata.

    The various logits processing functions called in this method
    may update the logits tensor in-place.
    """
    time_util_sample = StatTimeUtil()
    # print("===hucuihua Sampler sample, sampling_metadata.all_greedy:", sampling_metadata.all_greedy,
    #       " sampling_metadata.all_random:", sampling_metadata.all_random,
    #       " sampling_metadata.logitsprocs.argmax_invariant:", sampling_metadata.logitsprocs.argmax_invariant)
    if sampling_metadata.all_greedy and sampling_metadata.all_random:
        raise ValueError("all_greedy and all_random cannot both be True")
    if sampling_metadata.all_random:
        greedy_sampled = None
    else:
        greedy_sampled = self.greedy_sample(logits)
        if sampling_metadata.all_greedy:
            processed_logprobs = None
            if sampling_metadata.max_num_logprobs is not None:
                if self.logprobs_mode == LogprobsMode.PROCESSED_LOGITS:
                    processed_logprobs = logits
                elif self.logprobs_mode == LogprobsMode.PROCESSED_LOGPROBS:
                    processed_logprobs = self.compute_logprobs(logits)
            vllm_output_statics.add_stat(StatPhase.post_samper_sample_greedy_time, time_util_sample.get_duration())
            return greedy_sampled, processed_logprobs

    if sampling_metadata.temperature is None:
        raise ValueError("temperature cannot be None for random sampling")

    vllm_output_statics.add_stat(StatPhase.post_samper_sample_greedy_time, time_util_sample.get_duration())

    # Apply temperature.
    logits = self.apply_temperature(logits, sampling_metadata.temperature)
    vllm_output_statics.add_stat(StatPhase.post_samper_sample_apply_temperature_time, time_util_sample.get_duration())

    # Apply logits processors that only apply to random sampling
    # (argmax invariant)
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)
    vllm_output_statics.add_stat(
        StatPhase.post_samper_sample_processor_apply_again_time, time_util_sample.get_duration()
    )

    # Apply top_k and/or top_p.
    random_sampled, processed_logprobs = self.topk_topp_sampler(
        logits,
        sampling_metadata.generators,
        sampling_metadata.top_k,
        sampling_metadata.top_p,
    )

    vllm_output_statics.add_stat(StatPhase.post_samper_sample_topk_topp_time, time_util_sample.get_duration())

    if greedy_sampled is None:
        return random_sampled, processed_logprobs

    sampled = torch.where(
        sampling_metadata.temperature < _SAMPLING_EPS,
        greedy_sampled,
        random_sampled,
        out=greedy_sampled,  # Reuse tensor
    )

    vllm_output_statics.add_stat(StatPhase.post_samper_sample_greedy_where_time, time_util_sample.get_duration())
    return sampled, processed_logprobs


is_vllm_statistic = os.getenv('ENABLE_VLLM_STAT', "False").lower() == "true"
if is_vllm_statistic:
    Sampler.forward = forward_patch
    Sampler.sample = sample_patch
