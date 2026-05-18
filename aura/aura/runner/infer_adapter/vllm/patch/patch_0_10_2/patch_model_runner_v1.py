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

import torch
import torch._dynamo.cache_size
import numpy as np
import torch.distributed as dist
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_dp_group, get_pp_group, get_tp_group
from vllm.utils import cdiv
from vllm_ascend.utils import ProfileExecuteDuration, lmhead_tp_enable, vllm_version_is
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

import os
from typing import TYPE_CHECKING, Optional, Union
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.forward_context import BatchDescriptor
from vllm.sampling_params import SamplingType
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
from vllm_ascend.worker.mtp_proposer_v1 import MtpProposer
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata
from vllm_ascend.torchair.torchair_attention import AscendTorchairMetadata
from vllm_ascend.torchair.torchair_mla import AscendMLATorchairMetadata
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.worker.npu_input_batch import CachedRequestState

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
from vllm_ascend.worker.eagle_proposer_v1 import EagleProposer
from ..comm.vllm_execute_stat import StatTimeUtil, vllm_output_statics, StatPhase
from ..comm.npu_model_profiling import run_model_with_profiling

from vllm.logger import logger

original_model_runner_init = NPUModelRunner.__init__
stats_prepare = {}


def model_runner_init(self, vllm_config: VllmConfig, device: torch.device):
    original_model_runner_init(self, vllm_config, device)
    self.mc2_tokens_capacity = 256
    dist.barrier(group=get_dp_group().cpu_group)
    self.stat_step = 0


def sync_metadata_across_dp(
    self, num_tokens: int, with_prefill: bool, enable_dbo: bool
) -> tuple[int, Optional[torch.Tensor], bool, bool]:
    # if self.dp_size == 1 or self.vllm_config.model_config.enforce_eager:
    if self.dp_size == 1:
        return num_tokens, None, with_prefill, enable_dbo

    num_tokens_tensor = torch.tensor(
        [num_tokens if i == self.dp_rank else 0 for i in range(self.dp_size)], dtype=torch.int32, device="cpu"
    )

    flags_tensor = torch.tensor([int(with_prefill), int(not enable_dbo)], dtype=torch.int32, device="cpu")

    packed_tensor = torch.cat([num_tokens_tensor, flags_tensor])

    dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)
    dist.barrier(group=get_dp_group().cpu_group)

    # Unpack the results
    num_tokens_across_dp = packed_tensor[:-2]
    synced_flags = packed_tensor[-2:]

    max_tokens_across_dp = torch.max(num_tokens_across_dp).item()
    global_with_prefill = bool(synced_flags[0])
    global_enable_dbo = not bool(synced_flags[1])

    # Create a tensor for num_tokens_after_padding
    num_tokens_after_padding = torch.tensor([max_tokens_across_dp] * self.dp_size, device="npu", dtype=torch.int32)
    return max_tokens_across_dp, num_tokens_after_padding, global_with_prefill, global_enable_dbo


@torch.inference_mode()
def execute_model_patch(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> Union[ModelRunnerOutput, torch.Tensor]:
    time_util = StatTimeUtil()
    with ProfileExecuteDuration().capture_async("prepare input"):
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                logger.debug("skip this step for we receive the data from remote disaggregate prefill node")
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output)
        (
            attn_metadata,
            positions,
            num_scheduled_tokens_np,
            num_input_tokens,
            num_tokens_across_dp,
            maybe_padded_num_tokens,
            logits_indices,
            spec_decode_metadata,
            input_ids,
            inputs_embeds,
            intermediate_tensors,
        ) = self._prepare_inputs(scheduler_output, intermediate_tensors)

    self.stat_step += 1
    requestid_stepid = str(self.device) + "/" + "|".join(self.input_batch.req_ids) + "/" + str(self.stat_step)
    vllm_output_statics.set_cur_requestid_stepid(requestid_stepid, time_util.last_time)

    for key, value in stats_prepare.items():
        vllm_output_statics.add_stat(key, value)
    stats_prepare.clear()

    vllm_output_statics.add_stat(StatPhase.prepare_input_time, time_util.get_duration())
    vllm_output_statics.set_stat(StatPhase.with_prefill, self.with_prefill)
    vllm_output_statics.set_stat(StatPhase.attn_state, attn_metadata.attn_state)
    vllm_output_statics.set_stat(StatPhase.num_actual_tokens, attn_metadata.num_actual_tokens)
    vllm_output_statics.set_stat(StatPhase.batch_num, attn_metadata.seq_lens.shape[0])
    vllm_output_statics.set_stat(StatPhase.seq_lens, attn_metadata.seq_lens.tolist())

    moe_comm_method = self._select_moe_comm_method(num_input_tokens)

    batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens, uniform_decode=False)
    aclgraph_runtime_mode, batch_descriptor = self.aclgraph_dispatcher.dispatch(batch_descriptor)
    vllm_output_statics.add_stat(StatPhase.aclgraph_dispatcher_time, time_util.get_duration())
    # Run forward pass
    with ProfileExecuteDuration().capture_async("forward"):
        with set_ascend_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            with_prefill=self.with_prefill,
            reserved_mc2_mask=self.reserved_mc2_mask,
            moe_comm_method=moe_comm_method,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            batch_descriptor=batch_descriptor,
            num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
        ):
            self.maybe_setup_kv_connector(scheduler_output)
            hidden_states = self._generate_process_reqs_hidden_states(
                attn_metadata,
                self.with_prefill,
                maybe_padded_num_tokens,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )

        self.maybe_wait_for_kv_save()
        finished_sending, finished_recving = self.get_finished_kv_transfer(scheduler_output)

        aux_hidden_states = None
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = hidden_states

    vllm_output_statics.add_stat(StatPhase.forward_time, time_util.get_duration())

    kv_connector_output = None
    if finished_sending is not None or finished_recving is not None:
        kv_connector_output = KVConnectorOutput(finished_sending=finished_sending, finished_recving=finished_recving)
        vllm_output_statics.add_stat(StatPhase.kvconnectoroutput_time, time_util.get_duration())
    else:
        kv_connector_output = None

    finished_sending = None
    finished_recving = None

    time_util_post = StatTimeUtil()
    with ProfileExecuteDuration().capture_async("post process"):
        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend == "external_launcher" and len(get_pp_group().ranks) > 0
        )
        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            if not broadcast_pp_output:
                hidden_states.kv_connector_output = kv_connector_output
                return hidden_states
            if not isinstance(hidden_states, IntermediateTensors):
                raise RuntimeError(
                    f"hidden_states must be IntermediateTensors for mid-pipeline stages, "
                    f"but got {type(hidden_states).__name__}"
                )
            get_pp_group().send_tensor_dict(hidden_states.tensors, all_gather_group=get_tp_group())
            logits = None
        else:
            if self.input_batch.pooling_params:
                if vllm_version_is("0.10.1.1") or vllm_version_is("0.10.1"):
                    return self._pool_v010(
                        hidden_states,
                        scheduler_output.total_num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        finished_sending,
                        finished_recving,
                        kv_connector_output,
                    )
                else:
                    return self._pool(
                        hidden_states,
                        scheduler_output.total_num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        finished_sending,
                        finished_recving,
                        kv_connector_output,
                    )
            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)
            vllm_output_statics.add_stat(StatPhase.post_process_compute_logits_time, time_util_post.get_duration())

        if broadcast_pp_output:
            model_output_broadcast_data = (
                {
                    "logits": logits.contiguous(),
                }
                if logits is not None
                else {}
            )
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
            )
            if model_output_broadcast_data is None:
                raise RuntimeError("model_output_broadcast_data cannot be None after broadcast")
            logits = model_output_broadcast_data["logits"]

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            logits = self.apply_grammar_bitmask(scheduler_output, logits)

        time_util_sample = StatTimeUtil()
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            if lmhead_tp_enable() and logits is not None:
                logits = logits[: self.input_batch.num_reqs]
            vllm_output_statics.add_stat(StatPhase.post_samper_logits_slice_time, time_util_sample.get_duration())
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
            vllm_output_statics.add_stat(StatPhase.post_process_sampler_time, time_util_post.get_duration())
        else:
            if lmhead_tp_enable() and logits is not None:
                logits = logits[: len(spec_decode_metadata.logits_indices)]
            # When indexing with a tensor (bonus_logits_indices), PyTorch
            # creates a new tensor with separate storage from the original
            # logits tensor. This means any in-place operations on bonus_logits
            # won't affect the original logits tensor.
            if logits is None:
                raise RuntimeError("logits cannot be None during speculative decoding")
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids

            # Just like `bonus_logits`, `target_logits` is a new tensor with
            # separate storage from the original `logits` tensor. Therefore,
            # it is safe to update `target_logits` in place.
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                None,
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids

        discard_sampled_tokens_req_indices: List[int] = []
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = req_state.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                discard_sampled_tokens_req_indices.append(i)

        # NOTE: NPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[: scheduler_output.total_num_scheduled_tokens],
            scheduler_output,
        )

        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )

        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()
        # Cache the sampled tokens in the model runner, so that the schedulerAdd commentMore actions
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            if end_idx > self.model_config.max_model_len:
                raise RuntimeError(
                    f"Sampled token IDs exceed the max model length. "
                    f"Total number of tokens: {end_idx} > max_model_len: {self.model_config.max_model_len}"
                )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        if self.speculative_config:
            self._draft_token_ids = self.propose_draft_token_ids(
                valid_sampled_token_ids,
                sampling_metadata,
                scheduler_output,
                spec_decode_metadata,
                positions,
                scheduler_output.total_num_scheduled_tokens,
                hidden_states,
                attn_metadata,
                aux_hidden_states,
            )

        if has_kv_transfer_group():
            get_kv_transfer_group().clear_connector_metadata()
        vllm_output_statics.add_stat(StatPhase.post_process_other_time, time_util_post.get_duration())

    vllm_output_statics.add_stat(StatPhase.post_process_time, time_util.get_duration())

    extra_args = {"kv_connector_output": kv_connector_output}

    if vllm_version_is("0.10.1.1") or vllm_version_is("0.10.1"):
        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            spec_token_ids=self._draft_token_ids,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            **extra_args,
        )
    else:
        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            **extra_args,
        )

    durations = ProfileExecuteDuration().pop_captured_sync()
    if durations:
        dr_str = [f"[{tag}]:{duration:.2f}ms" for tag, duration in durations.items()]
        captured_name = "Decode" if self.attn_state == AscendAttentionState.DecodeOnly else "Prefill"
        logger.info("Profile execute duration [%s]:%s", captured_name, " ".join(dr_str))

    vllm_output_statics.add_stat(StatPhase.pop_captured_sync_time, time_util.get_duration())
    vllm_output_statics.set_step_finish_time(time_util.last_time)

    return model_runner_output


@torch.inference_mode()
def dummy_run(
    self,
    num_tokens: int,
    with_prefill: bool = False,
    is_torchair_compile: bool = False,
    aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
    force_attention: bool = False,
    uniform_decode: bool = False,
) -> torch.Tensor:
    if aclgraph_runtime_mode is not None and aclgraph_runtime_mode not in {
        CUDAGraphMode.NONE,
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL,
    }:
        raise RuntimeError(
            f"aclgraph_runtime_mode must be None, CUDAGraphMode.NONE, "
            f"CUDAGraphMode.PIECEWISE, or CUDAGraphMode.FULL, but got {aclgraph_runtime_mode}"
        )

    if force_attention:
        raise RuntimeError("Capturing attention in aclgraph is unexpected, because full graph is not supported now")

    # Padding for DP
    (num_tokens, num_tokens_across_dp, with_prefill, _) = self._sync_metadata_across_dp(num_tokens, with_prefill, False)

    moe_comm_method = self._select_moe_comm_method(num_tokens)

    max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

    max_num_reqs = self.scheduler_config.max_num_seqs
    if num_tokens > self.scheduler_config.max_num_batched_tokens:
        raise RuntimeError(
            f"num_tokens ({num_tokens}) cannot exceed max_num_batched_tokens "
            f"({self.scheduler_config.max_num_batched_tokens})"
        )

    max_num_reqs = self.scheduler_config.max_num_seqs
    if uniform_decode:
        num_reqs = cdiv(num_tokens, max_query_len)
        num_scheduled_tokens_list = [max_query_len] * num_reqs
        if num_tokens % max_query_len != 0:
            num_scheduled_tokens_list[-1] = num_tokens % max_query_len
    else:
        if with_prefill:
            num_reqs = num_tokens
        else:
            num_reqs = (num_tokens + self.decode_token_per_req - 1) // self.decode_token_per_req
        num_reqs = min(num_reqs, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs

    if sum(num_scheduled_tokens_list) != num_tokens:
        raise RuntimeError(
            f"Sum of scheduled tokens list ({sum(num_scheduled_tokens_list)}) must equal num_tokens ({num_tokens})"
        )

    if len(num_scheduled_tokens_list) != num_reqs:
        raise RuntimeError(
            f"Length of scheduled tokens list ({len(num_scheduled_tokens_list)}) must equal num_reqs ({num_reqs})"
        )

    num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)

    if self.is_kv_producer:
        with_prefill = True

    attn_metadata = self._build_attention_metadata(with_prefill, num_reqs, skip_attn=True)

    with self.maybe_dummy_run_with_lora(self.lora_config, num_scheduled_tokens):
        if self.is_multimodal_model:
            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_tokens]
        else:
            input_ids = self.input_ids[:num_tokens]
            inputs_embeds = None

        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_tokens]
        else:
            positions = self.positions[:num_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            if self.intermediate_tensors is None:
                self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                    batch_size=num_tokens, dtype=self.dtype, device=self.device
                )
            intermediate_tensors = IntermediateTensors(
                {k: v[:num_tokens] for k, v in self.intermediate_tensors.items()}
            )

        _ag_mode, batch_descriptor = self.aclgraph_dispatcher.dispatch(
            BatchDescriptor(num_tokens=num_tokens, uniform_decode=uniform_decode)
        )

        if aclgraph_runtime_mode is not None:
            if not (aclgraph_runtime_mode == CUDAGraphMode.NONE or aclgraph_runtime_mode == _ag_mode):
                raise RuntimeError(
                    f"Aclgraph runtime mode mismatch at dummy_run. "
                    f"Expected {_ag_mode}, but got {aclgraph_runtime_mode}."
                )
        else:
            aclgraph_runtime_mode = _ag_mode

        need_dummy_logits = not self.in_profile_run and lmhead_tp_enable()

        if need_dummy_logits:
            max_num_reqs_across_dp = num_tokens if not with_prefill else max_num_reqs
            dummy_indices = torch.zeros(max_num_reqs_across_dp, dtype=torch.int32)

            def dummy_compute_logits(hidden_states):
                return self.model.compute_logits(hidden_states[dummy_indices], None)

        with set_ascend_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            with_prefill=with_prefill,
            in_profile_run=self.in_profile_run,
            reserved_mc2_mask=self.reserved_mc2_mask,
            moe_comm_method=moe_comm_method,
            num_actual_tokens=0,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        ):
            hidden_states = self._generate_dummy_run_hidden_states(
                with_prefill,
                is_torchair_compile,
                input_ids,
                positions,
                attn_metadata,
                num_tokens,
                intermediate_tensors,
                inputs_embeds,
            )
            if need_dummy_logits:
                dummy_compute_logits(hidden_states)

        if self.speculative_config and self.speculative_config.method == "deepseek_mtp":
            if not isinstance(self.drafter, MtpProposer):
                raise RuntimeError(
                    f"drafter must be MtpProposer for deepseek_mtp, but got {type(self.drafter).__name__}"
                )
            self.drafter.dummy_run(
                num_tokens=num_tokens,
                with_prefill=with_prefill,
                skip_attn=True,
                num_reqs=num_reqs,
                num_tokens_across_dp=num_tokens_across_dp,
            )
            if need_dummy_logits:
                dummy_compute_logits(hidden_states)
        return hidden_states


@torch.inference_mode()
def dummy_run_with_stat(
    self,
    num_tokens: int,
    with_prefill: bool = False,
    is_torchair_compile: bool = False,
    aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
    force_attention: bool = False,
    uniform_decode: bool = False,
) -> torch.Tensor:
    time_util = StatTimeUtil()

    if aclgraph_runtime_mode is not None and aclgraph_runtime_mode not in {
        CUDAGraphMode.NONE,
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL,
    }:
        raise RuntimeError(
            f"aclgraph_runtime_mode must be None, CUDAGraphMode.NONE, "
            f"CUDAGraphMode.PIECEWISE, or CUDAGraphMode.FULL, but got {aclgraph_runtime_mode}"
        )

    if force_attention:
        raise RuntimeError("Capturing attention in aclgraph is unexpected, because full graph is not supported now")

    (num_tokens, num_tokens_across_dp, with_prefill, _) = self._sync_metadata_across_dp(num_tokens, with_prefill, False)

    moe_comm_method = self._select_moe_comm_method(num_tokens)

    max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

    max_num_reqs = self.scheduler_config.max_num_seqs
    if num_tokens > self.scheduler_config.max_num_batched_tokens:
        raise RuntimeError(
            f"num_tokens ({num_tokens}) cannot exceed max_num_batched_tokens "
            f"({self.scheduler_config.max_num_batched_tokens})"
        )

    max_num_reqs = self.scheduler_config.max_num_seqs
    if uniform_decode:
        num_reqs = cdiv(num_tokens, max_query_len)
        num_scheduled_tokens_list = [max_query_len] * num_reqs
        if num_tokens % max_query_len != 0:
            num_scheduled_tokens_list[-1] = num_tokens % max_query_len
    else:
        if with_prefill:
            num_reqs = num_tokens
        else:
            num_reqs = (num_tokens + self.decode_token_per_req - 1) // self.decode_token_per_req
        num_reqs = min(num_reqs, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs

    if sum(num_scheduled_tokens_list) != num_tokens:
        raise RuntimeError(
            f"Sum of scheduled tokens list ({sum(num_scheduled_tokens_list)}) must equal num_tokens ({num_tokens})"
        )

    if len(num_scheduled_tokens_list) != num_reqs:
        raise RuntimeError(
            f"Length of scheduled tokens list ({len(num_scheduled_tokens_list)}) must equal num_reqs ({num_reqs})"
        )

    num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)

    if self.is_kv_producer:
        with_prefill = True

    attn_metadata = self._build_attention_metadata(with_prefill, num_reqs, skip_attn=True)

    with self.maybe_dummy_run_with_lora(self.lora_config, num_scheduled_tokens):
        if self.is_multimodal_model:
            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_tokens]
        else:
            input_ids = self.input_ids[:num_tokens]
            inputs_embeds = None

        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_tokens]
        else:
            positions = self.positions[:num_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            if self.intermediate_tensors is None:
                self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                    batch_size=num_tokens, dtype=self.dtype, device=self.device
                )
            intermediate_tensors = IntermediateTensors(
                {k: v[:num_tokens] for k, v in self.intermediate_tensors.items()}
            )

        self.stat_step += 1
        requestid_stepid = str(self.device) + "/" + f"dummy_run_{self.stat_step}" + "/" + str(self.stat_step)
        vllm_output_statics.set_cur_requestid_stepid(requestid_stepid, time_util.last_time)
        vllm_output_statics.add_stat(StatPhase.prepare_input_time, time_util.get_duration())
        vllm_output_statics.set_stat(StatPhase.with_prefill, with_prefill)
        vllm_output_statics.set_stat(StatPhase.is_dummy_run, True)

        _ag_mode, batch_descriptor = self.aclgraph_dispatcher.dispatch(
            BatchDescriptor(num_tokens=num_tokens, uniform_decode=uniform_decode)
        )

        if aclgraph_runtime_mode is not None:
            if not (aclgraph_runtime_mode == CUDAGraphMode.NONE or aclgraph_runtime_mode == _ag_mode):
                raise RuntimeError(
                    f"Aclgraph runtime mode mismatch at dummy_run. "
                    f"Expected {_ag_mode}, but got {aclgraph_runtime_mode}."
                )
        else:
            aclgraph_runtime_mode = _ag_mode

        need_dummy_logits = not self.in_profile_run and lmhead_tp_enable()

        if need_dummy_logits:
            max_num_reqs_across_dp = num_tokens if not with_prefill else max_num_reqs
            dummy_indices = torch.zeros(max_num_reqs_across_dp, dtype=torch.int32)

            def dummy_compute_logits(hidden_states):
                return self.model.compute_logits(hidden_states[dummy_indices], None)

        vllm_output_statics.add_stat(StatPhase.aclgraph_dispatcher_time, time_util.get_duration())

        with set_ascend_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            with_prefill=with_prefill,
            in_profile_run=self.in_profile_run,
            reserved_mc2_mask=self.reserved_mc2_mask,
            moe_comm_method=moe_comm_method,
            num_actual_tokens=0,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        ):
            hidden_states = self._generate_dummy_run_hidden_states(
                with_prefill,
                is_torchair_compile,
                input_ids,
                positions,
                attn_metadata,
                num_tokens,
                intermediate_tensors,
                inputs_embeds,
            )
            if need_dummy_logits:
                dummy_compute_logits(hidden_states)

        vllm_output_statics.add_stat(StatPhase.forward_time, time_util.get_duration())

        if self.speculative_config and self.speculative_config.method == "deepseek_mtp":
            if not isinstance(self.drafter, MtpProposer):
                raise RuntimeError(
                    f"drafter must be MtpProposer for deepseek_mtp, but got {type(self.drafter).__name__}"
                )
            self.drafter.dummy_run(
                num_tokens=num_tokens,
                with_prefill=with_prefill,
                skip_attn=True,
                num_reqs=num_reqs,
                num_tokens_across_dp=num_tokens_across_dp,
            )
            if need_dummy_logits:
                dummy_compute_logits(hidden_states)

        vllm_output_statics.add_stat(StatPhase.post_process_time, time_util.get_duration())
        vllm_output_statics.set_step_finish_time(time_util.last_time)
        return hidden_states


def _generate_process_reqs_hidden_states_patch(
    self,
    attn_metadata,
    with_prefill,
    maybe_padded_num_tokens,
    input_ids,
    positions,
    intermediate_tensors,
    inputs_embeds,
):
    if self.model is None:
        raise RuntimeError("Model must be initialized before generating hidden states")

    if not with_prefill and (self.stat_step % profiling_sample_prob == 0):
        vllm_output_statics.set_stat(StatPhase.is_profiling, True)
        hidden_states = run_model_with_profiling(
            self.model,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            self.stat_step,
            vllm_output_statics.process_name,
        )
    else:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
    return hidden_states


def _generate_dummy_run_hidden_states_patch(
    self,
    with_prefill,
    is_torchair_compile,
    input_ids,
    positions,
    attn_metadata,
    num_tokens,
    intermediate_tensors,
    inputs_embeds,
):
    if not with_prefill and (self.stat_step % profiling_sample_prob == 0):
        vllm_output_statics.set_stat(StatPhase.is_profiling, True)
        hidden_states = run_model_with_profiling(
            self.model,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            self.stat_step,
            vllm_output_statics.process_name,
        )
    else:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    if self.use_aux_hidden_state_outputs:
        hidden_states, _ = hidden_states
    else:
        hidden_states = hidden_states
    if self.use_spec_decode and isinstance(self.drafter, EagleProposer):
        self.drafter.dummy_run(num_tokens)
    return hidden_states


def _prepare_inputs_patch(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> tuple[
    Union[AscendMetadata, AscendMLAMetadata, AscendTorchairMetadata, AscendMLATorchairMetadata],
    torch.Tensor,
    np.ndarray,
    int,
    torch.Tensor,
    int,
    torch.Tensor,
    SpecDecodeMetadata,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    time_util = StatTimeUtil()
    global stats_prepare
    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    if total_num_scheduled_tokens <= 0:
        raise ValueError(f"total_num_scheduled_tokens must be positive, got {total_num_scheduled_tokens}")
    num_reqs = self.input_batch.num_reqs
    if num_reqs <= 0:
        raise ValueError(f"num_reqs must be positive, got {num_reqs}")

    self.attn_metadata_builder.reorder_batch(self.input_batch, scheduler_output)

    self.input_batch.block_table.commit_block_table(num_reqs)
    stats_prepare[StatPhase.prepare_copy_bt_time] = time_util.get_duration()

    num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
    num_valid_tokens = np.empty(num_reqs, dtype=np.int32)
    max_num_scheduled_tokens = 0
    for i, req_id in enumerate(self.input_batch.req_ids):
        num_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_scheduled_tokens[i] = num_tokens
        num_valid_tokens[i] = num_tokens - len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
        max_num_scheduled_tokens = max(max_num_scheduled_tokens, num_tokens)
    stats_prepare[StatPhase.prepare_get_tokens_time] = time_util.get_duration()

    if self.use_aclgraph and total_num_scheduled_tokens <= self.aclgraph_batch_sizes[-1]:
        # Add padding to the batch size.
        num_input_tokens = self.vllm_config.pad_for_cudagraph(total_num_scheduled_tokens)
        stats_prepare[StatPhase.prepare_pad_tokens_time] = time_util.get_duration()
    else:
        # Eager mode.
        num_input_tokens = total_num_scheduled_tokens
    attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)
    self.attn_state = attn_state

    with_prefill = attn_state not in [AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding]

    self.query_lens = torch.from_numpy(num_scheduled_tokens)
    enable_dbo = self._check_dbo_is_valid(self.query_lens.tolist(), attn_state, total_num_scheduled_tokens)

    (maybe_padded_num_tokens, num_tokens_across_dp, with_prefill, enable_dbo) = self._sync_metadata_across_dp(
        num_input_tokens, with_prefill, enable_dbo
    )
    stats_prepare[StatPhase.prepare_sync_meta_time] = time_util.get_duration()

    # We should consider removing maybe_padded_num_tokens later
    num_input_tokens = maybe_padded_num_tokens

    # Hot-Swap lora model: False
    if self.lora_config:
        self.set_active_loras(self.input_batch, num_scheduled_tokens)
        stats_prepare[StatPhase.prepare_set_lora_time] = time_util.get_duration()

    req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

    cu_num_tokens = np.cumsum(num_scheduled_tokens)
    cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens, num_scheduled_tokens)
    arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets

    positions_np = self.positions_np[:total_num_scheduled_tokens]

    np.add(self.input_batch.num_computed_tokens_cpu[req_indices], arange, out=positions_np)
    stats_prepare[StatPhase.prepare_pos_cpu_time] = time_util.get_duration()

    # Calculate M-RoPE positions.
    # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
    if self.uses_mrope:  # False
        self._calc_mrope_positions(scheduler_output)

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
            self.mrope_positions_cpu[:, :total_num_scheduled_tokens], non_blocking=True
        )
        stats_prepare[StatPhase.prepare_mrope_time] = time_util.get_duration()

    self.positions_cpu[total_num_scheduled_tokens:num_input_tokens].zero_()
    self.positions[:num_input_tokens].copy_(self.positions_cpu[:num_input_tokens], non_blocking=True)
    positions_cpu = self.positions_cpu[:num_input_tokens]
    stats_prepare[StatPhase.prepare_pos_npu_time] = time_util.get_duration()

    positions = self.positions[:num_input_tokens]
    self.query_lens = torch.from_numpy(num_scheduled_tokens)

    self.seq_lens_np[:num_reqs] = self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
    seq_lens_cpu = self.seq_lens_cpu[:num_reqs]

    block_table_indices = req_indices * self.max_num_blocks_per_req + positions_np // self.block_size
    block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
    block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
    block_offsets = positions_np % self.block_size
    np.add(block_numbers * self.block_size, block_offsets, out=self.slot_mapping_np[:total_num_scheduled_tokens])
    stats_prepare[StatPhase.prepare_slot_map_time] = time_util.get_duration()

    attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)

    self.attn_mask = self._make_attention_mask(seq_lens=seq_lens_cpu, position=positions_cpu, attn_state=attn_state)
    self.attn_state = attn_state
    stats_prepare[StatPhase.prepare_atten_mask_time] = time_util.get_duration()

    self.query_start_loc_np[0] = 0
    self.query_start_loc_np[1 : num_reqs + 1] = cu_num_tokens
    self.query_start_loc[: num_reqs + 1].copy_(self.query_start_loc_cpu[: num_reqs + 1], non_blocking=True)
    self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs], non_blocking=True)
    # Fill unused with -1. Needed for reshape_and_cache
    self.seq_lens[num_reqs:].fill_(0)
    self.query_start_loc[num_reqs + 1 :].fill_(-1)

    stats_prepare[StatPhase.prepare_seq_len_time] = time_util.get_duration()

    self.with_prefill = with_prefill
    self.num_tokens_across_dp = num_tokens_across_dp
    self._update_graph_pad_size(with_prefill, maybe_padded_num_tokens)  # self.graph_pad_size = -1
    common_attn_metadata = AscendCommonAttentionMetadata(
        query_start_loc=self.query_start_loc[: num_reqs + 1],
        query_start_loc_cpu=self.query_start_loc_cpu[: num_reqs + 1],
        seq_lens_cpu=self.seq_lens_cpu,
        num_reqs=num_reqs,
        num_actual_tokens=total_num_scheduled_tokens,
        actual_seq_lengths_q=self.actual_seq_lengths_q,
        block_table_tensor=self.input_batch.block_table[0].get_device_tensor(),
        slot_mapping_cpu=self.slot_mapping_cpu,
        positions=self.positions,
        attn_mask=self.attn_mask,
        spec_attn_mask=self.spec_attn_mask,
        attn_state=self.attn_state,
        enable_dbo_across_dp=enable_dbo,
        is_only_prefill=bool(np.all(num_valid_tokens != 1)),
        max_query_len=max_num_scheduled_tokens,
        graph_pad_size=self.graph_pad_size,
        decode_token_per_req=self.decode_token_per_req,
    )
    attn_metadata = self.attn_metadata_builder.build(common_attn_metadata, self.model)
    if self.vllm_config.model_config.use_mla:
        attn_metadata.num_input_tokens = num_input_tokens
    stats_prepare[StatPhase.prepare_attn_meta_time] = time_util.get_duration()

    token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
    torch.index_select(
        self.input_batch.token_ids_cpu_tensor.flatten(),
        0,
        torch.from_numpy(token_indices),
        out=self.input_ids_cpu[:total_num_scheduled_tokens],
    )
    stats_prepare[StatPhase.prepare_inputids_cpu_time] = time_util.get_duration()
    # Copy the tensors to the NPU.
    self.input_ids[:total_num_scheduled_tokens].copy_(
        self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True
    )
    stats_prepare[StatPhase.prepare_copy_inputids_time] = time_util.get_duration()

    if self.is_multimodal_model:
        # Run the multimodal encoder if any.
        self._execute_mm_encoder(scheduler_output)
        mm_embeds = self._gather_mm_embeddings(scheduler_output)

        # NOTE(woosuk): To unify token ids and soft tokens (vision
        # embeddings), we always use embeddings (rather than token ids)
        # as input to the multimodal model, even when the input is text.
        input_ids = self.input_ids[:total_num_scheduled_tokens]
        if mm_embeds:
            inputs_embeds = self.model.get_input_embeddings(input_ids, mm_embeds)
        else:
            inputs_embeds = self.model.get_input_embeddings(input_ids)
        self.inputs_embeds[:total_num_scheduled_tokens].copy_(inputs_embeds)
        inputs_embeds = self.inputs_embeds[:num_input_tokens]
        input_ids = None
        stats_prepare[StatPhase.prepare_inputsembeds_time] = time_util.get_duration()
    else:
        # For text-only models, we use token ids as input.
        # While it is possible to use embeddings as input just like the
        # multimodal models, it is not desirable for performance since
        # then the embedding layer is not included in the ACL graph.
        input_ids = self.input_ids[:num_input_tokens]
        inputs_embeds = None
        stats_prepare[StatPhase.prepare_slice_inputids_time] = time_util.get_duration()

    positions = self.positions[:num_input_tokens]  # TODO 注意耗时，在npu上
    input_ids, positions = self._update_input_ids_and_positions(
        input_ids, positions, num_input_tokens, with_prefill, maybe_padded_num_tokens
    )
    stats_prepare[StatPhase.prepare_update_ids_and_pos_time] = time_util.get_duration()

    if get_pp_group().is_first_rank:
        intermediate_tensors = None
    else:
        if intermediate_tensors is None:
            raise RuntimeError("intermediate_tensors cannot be None on non-first rank.")
        if self.intermediate_tensors is None:
            raise RuntimeError("self.intermediate_tensors cannot be None on non-first rank.")
        for k, v in intermediate_tensors.items():
            self.intermediate_tensors[k][:num_input_tokens].copy_(v[:num_input_tokens], non_blocking=True)
        intermediate_tensors = IntermediateTensors(
            {k: v[:num_input_tokens] for k, v in self.intermediate_tensors.items()}
        )
        stats_prepare[StatPhase.prepare_inter_tensors_time] = time_util.get_duration()

    use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
    if not use_spec_decode:
        spec_decode_metadata = None
        logits_indices = torch.from_numpy(cu_num_tokens - 1).to(self.device, non_blocking=True)
        stats_prepare[StatPhase.prepare_logits_indice_time] = time_util.get_duration()
    else:
        # Get the number of draft tokens for each request.
        # Iterate over the dictionary rather than all requests since not all
        # requests have draft tokens.
        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        for req_id, draft_token_ids in scheduler_output.scheduled_spec_decode_tokens.items():
            req_idx = self.input_batch.req_id_to_index[req_id]
            num_draft_tokens[req_idx] = len(draft_token_ids)

        spec_decode_metadata = self._calc_spec_decode_metadata(num_draft_tokens, cu_num_tokens)
        logits_indices = spec_decode_metadata.logits_indices
        stats_prepare[StatPhase.prepare_specdeco_meta_time] = time_util.get_duration()

    if lmhead_tp_enable():
        max_num_reqs_across_dp = maybe_padded_num_tokens if not with_prefill else self.max_num_reqs
        logits_indices = nn.functional.pad(logits_indices, (0, max_num_reqs_across_dp - logits_indices.shape[0]))
        stats_prepare[StatPhase.prepare_lmhead_logits_indices_time] = time_util.get_duration()

    return (
        attn_metadata,
        positions,
        num_scheduled_tokens,
        num_input_tokens,
        num_tokens_across_dp,
        maybe_padded_num_tokens,
        logits_indices,
        spec_decode_metadata,
        input_ids,
        inputs_embeds,
        intermediate_tensors,
    )


def _update_states_patch(self, scheduler_output: "SchedulerOutput") -> None:
    time_util = StatTimeUtil()
    global stats_prepare
    # Remove finished requests from the cached states.
    for req_id in scheduler_output.finished_req_ids:
        self.requests.pop(req_id, None)
        if vllm_version_is("0.10.1.1") or vllm_version_is("0.10.1"):
            self.encoder_cache.pop(req_id, None)
    # Remove the finished requests from the persistent batch.
    # NOTE(woosuk): There could be an edge case where finished_req_ids and
    # scheduled_req_ids overlap. This happens when a request is aborted and
    # then resubmitted with the same ID. In this case, we treat them as two
    # distinct requests - clearing the cached states for the first request
    # and handling the second as a new request.
    for req_id in scheduler_output.finished_req_ids:
        self.input_batch.remove_request(req_id)
    if vllm_version_is("0.10.1.1") or vllm_version_is("0.10.1"):
        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)
    else:
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)
    # Remove the unscheduled requests from the persistent batch.
    # NOTE(woosuk): The unscheduled requests are either preempted requests
    # or running requests that are not scheduled in this step. We remove
    # them from the persistent batch but keep their cached states since
    # they will be scheduled again sometime in the future.
    scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
    cached_req_ids = self.input_batch.req_id_to_index.keys()
    unscheduled_req_ids = cached_req_ids - scheduled_req_ids
    # NOTE(woosuk): The persistent batch optimization assumes that
    # consecutive batches contain mostly the same requests. If batches
    # have low request overlap (e.g., alternating between two distinct
    # sets of requests), this optimization becomes very inefficient.
    for req_id in unscheduled_req_ids:
        self.input_batch.remove_request(req_id)
    # ==========================remove request======================
    stats_prepare[StatPhase.prepare_remove_reqs_time] = time_util.get_duration()

    req_ids_to_add: list[str] = []
    # Add new requests to the cached states.
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id
        sampling_params = new_req_data.sampling_params
        pooling_params = new_req_data.pooling_params

        if sampling_params and sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        if pooling_params:
            task = pooling_params.task
            if task is None:
                raise ValueError("You did not set `task` in the API")
            model = cast(VllmModelForPooling, self.get_model())
            to_update = model.pooler.get_pooling_updates(task)
            to_update.apply(pooling_params)

        self.requests[req_id] = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            mm_kwargs=new_req_data.mm_kwargs,
            mm_positions=new_req_data.mm_positions,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            generator=generator,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
            output_token_ids=[],
            lora_request=new_req_data.lora_request,
            **(
                {"mm_hashes": new_req_data.mm_hashes}
                if not (vllm_version_is("0.10.1.1") or vllm_version_is("0.10.1"))
                else {"mm_hashes": None}
            ),
        )

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            image_grid_thw = []
            video_grid_thw = []
            second_per_grid_ts = []
            audio_feature_lengths = []
            use_audio_in_video = False
            for mm_item in self.requests[req_id].mm_kwargs:
                mm_input = mm_item.get_data()
                if mm_input.get("image_grid_thw") is not None:
                    image_grid_thw.append(mm_input["image_grid_thw"].tolist())
                if mm_input.get("video_grid_thw") is not None:
                    video_grid_thw.append(mm_input["video_grid_thw"].tolist())
                if mm_input.get("second_per_grid_ts") is not None:
                    second_per_grid_ts.append(mm_input["second_per_grid_ts"])
                if mm_input.get("audio_feature_lengths") is not None:
                    audio_feature_lengths.append(mm_input["audio_feature_lengths"])
                if mm_input.get("use_audio_in_video") is True:
                    use_audio_in_video = True

            hf_config = self.model_config.hf_config

            self.requests[req_id].mrope_positions, self.requests[req_id].mrope_position_delta = (
                MRotaryEmbedding.get_input_positions_tensor(
                    self.requests[req_id].prompt_token_ids,
                    hf_config=hf_config,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    audio_feature_lengths=audio_feature_lengths,
                    use_audio_in_video=use_audio_in_video,
                )
            )

        req_ids_to_add.append(req_id)
    # ==========================add request======================
    stats_prepare[StatPhase.prepare_add_reqs_time] = time_util.get_duration()

    # Update the states of the running/resumed requests.
    is_last_rank = get_pp_group().is_last_rank
    req_data = scheduler_output.scheduled_cached_reqs
    for i, req_id in enumerate(req_data.req_ids):
        req_state = self.requests[req_id]
        num_computed_tokens = req_data.num_computed_tokens[i]
        new_block_ids = req_data.new_block_ids[i]
        resumed_from_preemption = req_data.resumed_from_preemption[i]

        # Update the cached states.
        req_state.num_computed_tokens = num_computed_tokens

        if not is_last_rank:
            # When using PP, the scheduler sends the sampled tokens back,
            # because there's no direct communication between the first-
            # stage worker and the last-stage worker.
            new_token_ids = req_data.new_token_ids[i]
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec tokens.
            num_new_tokens = num_computed_tokens + len(new_token_ids) - req_state.num_tokens
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(new_token_ids[-num_new_tokens:])

        # Update the block IDs.
        if not resumed_from_preemption:
            if new_block_ids is not None:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                    block_ids.extend(new_ids)
        else:
            if new_block_ids is None:
                raise RuntimeError("new_block_ids must not be None when resuming from preemption.")
            # The request is resumed from preemption.
            # Replace the existing block IDs with the new ones.
            req_state.block_ids = new_block_ids

        req_index = self.input_batch.req_id_to_index.get(req_id)
        if req_index is None:
            # The request is not in the persistent batch.
            # The request was either preempted and resumed later, or was not
            # scheduled in the previous step and needs to be added again.
            req_ids_to_add.append(req_id)
            continue

        # Update the persistent batch.
        self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
        if new_block_ids is not None:
            self.input_batch.block_table.append_row(new_block_ids, req_index)

        # For the last rank, we don't need to update the token_ids_cpu
        # because the sampled tokens are already cached.
        if not is_last_rank:
            # Add new_token_ids to token_ids_cpu.
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(new_token_ids)
            self.input_batch.token_ids_cpu[req_index, start_token_index:end_token_index] = new_token_ids
            self.input_batch.num_tokens_no_spec[req_index] = end_token_index
            self.input_batch.num_tokens[req_index] = end_token_index

        # Add spec_token_ids to token_ids_cpu.
        spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())
        if spec_token_ids:
            num_spec_tokens = len(spec_token_ids)
            start_index = self.input_batch.num_tokens_no_spec[req_index]
            end_token_index = start_index + num_spec_tokens
            self.input_batch.token_ids_cpu[req_index, start_index:end_token_index] = spec_token_ids
            # NOTE(woosuk): `num_tokens` here may include spec tokens.
            self.input_batch.num_tokens[req_index] += num_spec_tokens
    # ==========================update states======================
    stats_prepare[StatPhase.prepare_update_states_time] = time_util.get_duration()

    # Add the new or resumed requests to the persistent batch.
    # The smaller empty indices are filled first.
    for req_id in req_ids_to_add:
        req_state = self.requests[req_id]
        self.input_batch.add_request(req_state)

    # Condense the batched states if there are gaps left by removed requests
    self.input_batch.condense()

    # Refresh batch metadata with any pending updates.
    self.input_batch.refresh_metadata()
    # ==========================others======================
    stats_prepare[StatPhase.prepare_other_states_time] = time_util.get_duration()


NPUModelRunner.__init__ = model_runner_init
NPUModelRunner._sync_metadata_across_dp = sync_metadata_across_dp
NPUModelRunner._dummy_run = dummy_run

is_vllm_statistic = os.getenv('ENABLE_VLLM_STAT', "False").lower() == "true"
if is_vllm_statistic:
    NPUModelRunner.execute_model = execute_model_patch
    NPUModelRunner._dummy_run = dummy_run_with_stat
    NPUModelRunner._update_states = _update_states_patch
    NPUModelRunner._prepare_inputs = _prepare_inputs_patch

is_profiling_forward = os.environ.get('PROFILING_FORWARD', "0") == '1'
profiling_sample_prob = int(os.environ.get('PROFILING_SAMPLE_PROB', "100"))
if is_profiling_forward:
    NPUModelRunner._generate_process_reqs_hidden_states = _generate_process_reqs_hidden_states_patch
