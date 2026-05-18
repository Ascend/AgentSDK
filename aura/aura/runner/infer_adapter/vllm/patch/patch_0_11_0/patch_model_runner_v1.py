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

from typing import Optional, Union
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
import torch
import torch._dynamo.cache_size
import torch.distributed as dist
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.distributed.parallel_state import get_dp_group
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm.logger import logger
from vllm.v1.outputs import ModelRunnerOutput

from aura.base.accuracy.haco_tool import enable_haco, vllm_model_runner_update_haco

HAS_HACO = enable_haco(logger)

original_model_runner_init = NPUModelRunner.__init__
original_model_execute = NPUModelRunner.execute_model
original_generate_process_reqs_hidden_states = NPUModelRunner._generate_process_reqs_hidden_states
stats_prepare = {}


def model_runner_init(self, vllm_config: VllmConfig, device: torch.device):
    original_model_runner_init(self, vllm_config, device)
    self.sentinel = None


def model_runner_model_execute(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
    if self.sentinel is None and HAS_HACO:
        self.sentinel = vllm_model_runner_update_haco(self.model)

    # original process
    model_runner_output = original_model_execute(self, scheduler_output, intermediate_tensors)

    return model_runner_output


def model_runner_generate_process_reqs_hidden_states(
    self,
    attn_metadata,
    with_prefill,
    maybe_padded_num_tokens,
    input_ids,
    positions,
    intermediate_tensors,
    inputs_embeds,
):
    if HAS_HACO:
        self.sentinel.record_input_id(self.input_batch.req_ids)

    # original process
    hidden_states = original_generate_process_reqs_hidden_states(
        self,
        attn_metadata,
        with_prefill,
        maybe_padded_num_tokens,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
    )

    if HAS_HACO:
        self.sentinel.inference_step()
    return hidden_states


def sync_metadata_across_dp(
    self, num_tokens: int, with_prefill: bool, enable_dbo: bool
) -> tuple[int, Optional[torch.Tensor], bool, bool]:
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
    num_tokens_after_padding = torch.tensor([max_tokens_across_dp] * self.dp_size, device="cpu", dtype=torch.int32)
    return max_tokens_across_dp, num_tokens_after_padding, global_with_prefill, global_enable_dbo


NPUModelRunner._sync_metadata_across_dp = sync_metadata_across_dp
NPUModelRunner.__init__ = model_runner_init
NPUModelRunner.execute_model = model_runner_model_execute
NPUModelRunner._generate_process_reqs_hidden_states = model_runner_generate_process_reqs_hidden_states
