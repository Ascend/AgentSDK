#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright contributors to the vLLM project
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

from typing import List, Tuple

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.core.sched.scheduler import Scheduler

from agentic_rl.runner.infer_adapter.vllm.patch.comm.scheduler_stat import RequestStats


original_scheduler_init = Scheduler.__init__


def scheduler_init(
    self,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    structured_output_manager: StructuredOutputManager,
    mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    include_finished_set: bool = False,
    log_stats: bool = False,
) -> None:
    original_scheduler_init(
        self, vllm_config, kv_cache_config, structured_output_manager,
        mm_registry, include_finished_set, log_stats)
    self.req_stats: RequestStats = RequestStats()


def update_after_schedule_patch(
    self,
    scheduler_output: SchedulerOutput,
) -> None:
    """Update request state after scheduling.

    This method advances the number of computed tokens for each scheduled
    request and tracks scheduling events for statistics. It handles encoder
    input cleanup for requests with encoder inputs.

    Args:
        scheduler_output: Output from the scheduler containing scheduled tokens.

    Returns:
        None
    """
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens
    for req_id, num_scheduled_token in num_scheduled_tokens.items():
        request = self.requests[req_id]
        if request.num_computed_tokens == 0 or request.num_computed_tokens == request.num_cached_tokens:
            self.req_stats.stat_schedule(request.request_id)
        request.num_computed_tokens += num_scheduled_token

        if request.has_encoder_inputs:
            self._free_encoder_inputs(request)


def update_request_with_output_patch(
    self,
    request: Request,
    new_token_ids: List[int],
) -> Tuple[List[int], bool]:
    """Update request state with generated output tokens.
    
    This method appends generated tokens to the request, checks for stop
    conditions, and tracks prefill completion and finish events for
    statistics collection.
    
    Args:
        request: The request to update.
        new_token_ids: List of newly generated token IDs.
    
    Returns:
        Tuple containing:
            - Trimmed list of new token IDs (if stopped early)
            - Boolean indicating if the request has stopped
    """
    stopped = False
    for num_new, output_token_id in enumerate(new_token_ids, 1):
        request.append_output_token_ids(output_token_id)

        stopped = check_stop(request, self.max_model_len)
        if stopped:
            self.req_stats.stat_finish(request.request_id, request.num_prompt_tokens, request.num_output_tokens)
            del new_token_ids[num_new:]
            break
    
    if request.num_output_tokens == 1:
        self.req_stats.stat_prefill_done(request.request_id)
    
    return new_token_ids, stopped


def add_request_patch(self, request: Request) -> None:
    self.waiting.add_request(request)
    self.requests[request.request_id] = request
    if self.log_stats:
        request.record_event(EngineCoreEventType.QUEUED)
    self.req_stats.stat_add(request.request_id)


def reset_prefix_cache_patch(self) -> bool:
    self.req_stats.print()
    return self.kv_cache_manager.reset_prefix_cache()


Scheduler.__init__ = scheduler_init
Scheduler._update_after_schedule = update_after_schedule_patch
Scheduler._update_request_with_output = update_request_with_output_patch
Scheduler.add_request = add_request_patch
Scheduler.reset_prefix_cache = reset_prefix_cache_patch
