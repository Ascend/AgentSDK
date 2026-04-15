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
import datetime
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import torch
import torch._dynamo.cache_size
import torch.distributed as dist
from vllm.config import CompilationLevel, CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import (get_dp_group, get_pp_group,
                                             get_tp_group,
                                             is_global_first_rank)
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        LazyLoader, cdiv, is_pin_memory_available)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT,
                             ModelRunnerOutput)
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.forward_context import BatchDescriptor
from vllm.logger import logger

from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ,
                               AscendSocVersion, ProfileExecuteDuration,
                               get_ascend_soc_version, is_310p,
                               lmhead_tp_enable, vllm_version_is)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.mtp_proposer_v1 import MtpProposer
from vllm_ascend.worker.eagle_proposer_v1 import EagleProposer

from agentic_rl.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat import (
    StatTimeUtil, vllm_output_statics, StatPhase)
from agentic_rl.runner.infer_adapter.vllm.patch.comm.npu_model_profiling import (
    run_model_with_profiling)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

original_model_runner_init = NPUModelRunner.__init__


def model_runner_init(self, vllm_config: VllmConfig, device: torch.device):
    original_model_runner_init(self, vllm_config, device)
    self.mc2_tokens_capacity = 256
    dist.barrier(group=get_dp_group().cpu_group)
    self.stat_step = 0


def sync_metadata_across_dp(
    self,
    num_tokens: int,
    with_prefill: bool,
    enable_dbo: bool
) -> Tuple[int, Optional[torch.Tensor], bool, bool]:
    if self.dp_size == 1:
        return num_tokens, None, with_prefill, enable_dbo

    num_tokens_tensor = torch.tensor([
        num_tokens if i == self.dp_rank else 0 for i in range(self.dp_size)
    ],
                                     dtype=torch.int32,
                                     device="cpu")

    flags_tensor = torch.tensor(
        [int(with_prefill), int(not enable_dbo)],
        dtype=torch.int32,
        device="cpu")

    packed_tensor = torch.cat([num_tokens_tensor, flags_tensor])

    dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)
    dist.barrier(group=get_dp_group().device_group)

    # Unpack the results
    num_tokens_across_dp = packed_tensor[:-2]
    synced_flags = packed_tensor[-2:]

    max_tokens_across_dp = torch.max(num_tokens_across_dp).item()
    global_with_prefill = bool(synced_flags[0])
    global_enable_dbo = not bool(synced_flags[1])

    # Create a tensor for num_tokens_after_padding
    num_tokens_after_padding = torch.tensor([max_tokens_across_dp] *
                                            self.dp_size,
                                            device="npu",
                                            dtype=torch.int32)
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
                logger.debug(
                    "skip this step for we receive the data from remote disaggregate prefill node"
                )
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output)
        (attn_metadata, positions, num_scheduled_tokens_np,
         num_input_tokens, num_tokens_across_dp, maybe_padded_num_tokens,
         logits_indices, spec_decode_metadata, input_ids, inputs_embeds,
         intermediate_tensors) = (self._prepare_inputs(
            scheduler_output, intermediate_tensors))

    self.stat_step += 1
    requestid_stepid = str(self.device) + "/" + "|".join(self.input_batch.req_ids) + "/" + str(self.stat_step)
    vllm_output_statics.set_cur_requestid_stepid(requestid_stepid, time_util.last_time)
    vllm_output_statics.add_stat(StatPhase.prepare_input_time, time_util.get_duration())
    vllm_output_statics.set_stat(StatPhase.with_prefill, self.with_prefill)
    vllm_output_statics.set_stat(StatPhase.attn_state, attn_metadata.attn_state)
    vllm_output_statics.set_stat(StatPhase.num_actual_tokens, attn_metadata.num_actual_tokens)
    vllm_output_statics.set_stat(StatPhase.batch_num, attn_metadata.seq_lens.shape[0])
    vllm_output_statics.set_stat(StatPhase.seq_lens, attn_metadata.seq_lens.tolist())

    moe_comm_method = self._select_moe_comm_method(num_input_tokens)

    batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                       uniform_decode=False)
    aclgraph_runtime_mode, batch_descriptor = \
        self.aclgraph_dispatcher.dispatch(batch_descriptor)
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
                num_actual_tokens=scheduler_output.total_num_scheduled_tokens):
            self.maybe_setup_kv_connector(scheduler_output)
            hidden_states = self._generate_process_reqs_hidden_states(
                attn_metadata, self.with_prefill, maybe_padded_num_tokens,
                input_ids, positions, intermediate_tensors, inputs_embeds)

        self.maybe_wait_for_kv_save()
        finished_sending, finished_recving = self.get_finished_kv_transfer(
            scheduler_output)

        aux_hidden_states = None
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = hidden_states

    vllm_output_statics.add_stat(StatPhase.forward_time, time_util.get_duration())

    kv_connector_output = None
    if finished_sending is not None or finished_recving is not None:
        kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving)
        vllm_output_statics.add_stat(StatPhase.kvconnectoroutput_time, time_util.get_duration())
    
    finished_sending = None
    finished_recving = None

    time_util_post = StatTimeUtil()
    with ProfileExecuteDuration().capture_async("post process"):
        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        broadcast_pp_output = \
            self.parallel_config.distributed_executor_backend \
            == "external_launcher" and len(get_pp_group().ranks) > 0
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
            get_pp_group().send_tensor_dict(
                hidden_states.tensors, all_gather_group=get_tp_group())
            logits = None
        else:
            if self.input_batch.pooling_params:
                if vllm_version_is("0.10.1.1") or vllm_version_is("0.10.1"):
                    return self._pool_v010(
                        hidden_states,
                        scheduler_output.total_num_scheduled_tokens,
                        num_scheduled_tokens_np, finished_sending,
                        finished_recving, kv_connector_output)
                else:
                    return self._pool(
                        hidden_states,
                        scheduler_output.total_num_scheduled_tokens,
                        num_scheduled_tokens_np, finished_sending,
                        finished_recving, kv_connector_output)
            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)
            vllm_output_statics.add_stat(StatPhase.post_process_compute_logits_time, time_util_post.get_duration())
        
        if broadcast_pp_output:
            model_output_broadcast_data = {
                "logits": logits.contiguous(),
            } if logits is not None else {}
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
            if model_output_broadcast_data is None:
                raise RuntimeError("model_output_broadcast_data cannot be None after broadcast")
            logits = model_output_broadcast_data["logits"]

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            logits = self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            if lmhead_tp_enable() and logits is not None:
                logits = logits[:self.input_batch.num_reqs]
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
            vllm_output_statics.add_stat(StatPhase.post_process_sampler_time, time_util_post.get_duration())
        else:
            if lmhead_tp_enable() and logits is not None:
                logits = logits[:len(spec_decode_metadata.logits_indices)]
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
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
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
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:scheduler_output.total_num_scheduled_tokens],
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
        dr_str = [
            f"[{tag}]:{duration:.2f}ms"
            for tag, duration in durations.items()
        ]
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
        CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
    }:
        raise RuntimeError(
            f"aclgraph_runtime_mode must be None, CUDAGraphMode.NONE, "
            f"CUDAGraphMode.PIECEWISE, or CUDAGraphMode.FULL, but got {aclgraph_runtime_mode}"
        )
    
    if force_attention:
        raise RuntimeError(
            "Capturing attention in aclgraph is unexpected, because full graph is not supported now"
        )

    # Padding for DP
    (num_tokens, num_tokens_across_dp, with_prefill,
     _) = self._sync_metadata_across_dp(num_tokens, with_prefill, False)

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
            f"Sum of scheduled tokens list ({sum(num_scheduled_tokens_list)}) "
            f"must equal num_tokens ({num_tokens})"
        )
    
    if len(num_scheduled_tokens_list) != num_reqs:
        raise RuntimeError(
            f"Length of scheduled tokens list ({len(num_scheduled_tokens_list)}) "
            f"must equal num_reqs ({num_reqs})"
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
                self.intermediate_tensors = (
                    self.model.make_empty_intermediate_tensors(
                        batch_size=num_tokens,
                        dtype=self.dtype,
                        device=self.device))
            intermediate_tensors = IntermediateTensors({
                k: v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            })

        _ag_mode, batch_descriptor = \
            self.aclgraph_dispatcher.dispatch(
                BatchDescriptor(num_tokens=num_tokens, uniform_decode=uniform_decode))
        
        if aclgraph_runtime_mode is not None:
            if not (aclgraph_runtime_mode == CUDAGraphMode.NONE or
                    aclgraph_runtime_mode == _ag_mode):
                raise RuntimeError(
                    f"Aclgraph runtime mode mismatch at dummy_run. "
                    f"Expected {_ag_mode}, but got {aclgraph_runtime_mode}."
                )
        else:
            aclgraph_runtime_mode = _ag_mode

        need_dummy_logits = (not self.in_profile_run and lmhead_tp_enable())

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
                batch_descriptor=batch_descriptor):
            hidden_states = self._generate_dummy_run_hidden_states(
                with_prefill, is_torchair_compile, input_ids, positions,
                attn_metadata, num_tokens, intermediate_tensors, inputs_embeds)
            if need_dummy_logits:
                dummy_compute_logits(hidden_states)

        if self.speculative_config and self.speculative_config.method == "deepseek_mtp":
            if not isinstance(self.drafter, MtpProposer):
                raise RuntimeError(
                    f"drafter must be MtpProposer for deepseek_mtp, "
                    f"but got {type(self.drafter).__name__}"
                )
            self.drafter.dummy_run(
                num_tokens=num_tokens,
                with_prefill=with_prefill,
                skip_attn=True,
                num_reqs=num_reqs,
                num_tokens_across_dp=num_tokens_across_dp)
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
        CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
    }:
        raise RuntimeError(
            f"aclgraph_runtime_mode must be None, CUDAGraphMode.NONE, "
            f"CUDAGraphMode.PIECEWISE, or CUDAGraphMode.FULL, but got {aclgraph_runtime_mode}"
        )
    
    if force_attention:
        raise RuntimeError(
            "Capturing attention in aclgraph is unexpected, because full graph is not supported now"
        )

    (num_tokens, num_tokens_across_dp, with_prefill,
     _) = self._sync_metadata_across_dp(num_tokens, with_prefill, False)

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
            f"Sum of scheduled tokens list ({sum(num_scheduled_tokens_list)}) "
            f"must equal num_tokens ({num_tokens})"
        )
    
    if len(num_scheduled_tokens_list) != num_reqs:
        raise RuntimeError(
            f"Length of scheduled tokens list ({len(num_scheduled_tokens_list)}) "
            f"must equal num_reqs ({num_reqs})"
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
                self.intermediate_tensors = (
                    self.model.make_empty_intermediate_tensors(
                        batch_size=num_tokens,
                        dtype=self.dtype,
                        device=self.device))
            intermediate_tensors = IntermediateTensors({
                k: v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            })

        self.stat_step += 1
        requestid_stepid = str(self.device) + "/" + f"dummy_run_{self.stat_step}" + "/" + str(self.stat_step)
        vllm_output_statics.set_cur_requestid_stepid(requestid_stepid, time_util.last_time)
        vllm_output_statics.add_stat(StatPhase.prepare_input_time, time_util.get_duration())
        vllm_output_statics.set_stat(StatPhase.with_prefill, with_prefill)
        vllm_output_statics.set_stat(StatPhase.is_dummy_run, True)

        _ag_mode, batch_descriptor = \
            self.aclgraph_dispatcher.dispatch(
                BatchDescriptor(num_tokens=num_tokens, uniform_decode=uniform_decode))
        
        if aclgraph_runtime_mode is not None:
            if not (aclgraph_runtime_mode == CUDAGraphMode.NONE or
                    aclgraph_runtime_mode == _ag_mode):
                raise RuntimeError(
                    f"Aclgraph runtime mode mismatch at dummy_run. "
                    f"Expected {_ag_mode}, but got {aclgraph_runtime_mode}."
                )
        else:
            aclgraph_runtime_mode = _ag_mode

        need_dummy_logits = (not self.in_profile_run and lmhead_tp_enable())

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
                batch_descriptor=batch_descriptor):
            hidden_states = self._generate_dummy_run_hidden_states(
                with_prefill, is_torchair_compile, input_ids, positions,
                attn_metadata, num_tokens, intermediate_tensors, inputs_embeds)
            if need_dummy_logits:
                dummy_compute_logits(hidden_states)

        vllm_output_statics.add_stat(StatPhase.forward_time, time_util.get_duration())

        if self.speculative_config and self.speculative_config.method == "deepseek_mtp":
            if not isinstance(self.drafter, MtpProposer):
                raise RuntimeError(
                    f"drafter must be MtpProposer for deepseek_mtp, "
                    f"but got {type(self.drafter).__name__}"
                )
            self.drafter.dummy_run(
                num_tokens=num_tokens,
                with_prefill=with_prefill,
                skip_attn=True,
                num_reqs=num_reqs,
                num_tokens_across_dp=num_tokens_across_dp)
            if need_dummy_logits:
                dummy_compute_logits(hidden_states)

        vllm_output_statics.add_stat(StatPhase.post_process_time, time_util.get_duration())
        vllm_output_statics.set_step_finish_time(time_util.last_time)
        return hidden_states


def _generate_process_reqs_hidden_states_patch(self, attn_metadata, with_prefill,
                                               maybe_padded_num_tokens,
                                               input_ids, positions,
                                               intermediate_tensors,
                                               inputs_embeds):
    if self.model is None:
        raise RuntimeError("Model must be initialized before generating hidden states")
    
    if not with_prefill and (self.stat_step % profiling_sample_prob == 0):
        vllm_output_statics.set_stat(StatPhase.is_profiling, True)
        hidden_states = run_model_with_profiling(
            self.model, input_ids, positions, intermediate_tensors,
            inputs_embeds, self.stat_step, vllm_output_statics.process_name)
    else:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
    return hidden_states


def _generate_dummy_run_hidden_states_patch(self, with_prefill,
                                            is_torchair_compile, input_ids,
                                            positions, attn_metadata, num_tokens,
                                            intermediate_tensors, inputs_embeds):
    if not with_prefill and (self.stat_step % profiling_sample_prob == 0):
        vllm_output_statics.set_stat(StatPhase.is_profiling, True)
        hidden_states = run_model_with_profiling(
            self.model, input_ids, positions, intermediate_tensors,
            inputs_embeds, self.stat_step, vllm_output_statics.process_name)
    else:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds)
    
    if self.use_aux_hidden_state_outputs:
        hidden_states, _ = hidden_states
    
    if self.use_spec_decode and isinstance(self.drafter, EagleProposer):
        self.drafter.dummy_run(num_tokens)
    return hidden_states


NPUModelRunner.__init__ = model_runner_init
NPUModelRunner._sync_metadata_across_dp = sync_metadata_across_dp
NPUModelRunner._dummy_run = dummy_run

is_vllm_statistic = os.getenv('ENABLE_VLLM_STAT', "False").lower() == "true"
if is_vllm_statistic:
    NPUModelRunner.execute_model = execute_model_patch
    NPUModelRunner._dummy_run = dummy_run_with_stat

is_profiling_forward = os.environ.get('PROFILING_FORWARD', "0") == '1'
profiling_sample_prob = int(os.environ.get('PROFILING_SAMPLE_PROB', "100"))
if is_profiling_forward:
    NPUModelRunner._generate_process_reqs_hidden_states = _generate_process_reqs_hidden_states_patch
