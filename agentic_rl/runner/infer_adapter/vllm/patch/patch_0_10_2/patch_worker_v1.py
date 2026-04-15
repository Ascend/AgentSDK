#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Adapted from vllm-project/vllm/vllm/worker/worker.py
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

# SPDX-License-Identifier: Apache-2.0
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.logger import logger
from vllm.utils import GiB_bytes
from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker.worker_v1 import NPUWorker


def multi_level_sleep(self, level: int = 1) -> None:
    NPUPlatform.set_device(self.device)
    free_bytes_before_sleep = NPUPlatform.mem_get_info()[0]
    allocator = CaMemAllocator.get_instance()
    
    if level == 1:
        allocator.sleep(offload_tags=("weights",))
    elif level == 2:
        allocator.sleep(offload_tags=("kv_cache",))
    else:
        allocator.sleep(offload_tags=tuple())
    
    free_bytes_after_sleep, total = NPUPlatform.mem_get_info()
    freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
    used_bytes = total - free_bytes_after_sleep
    
    if freed_bytes < 0:
        raise RuntimeError(f"Memory usage increased after sleeping.")
    
    logger.info(
        "Sleep mode freed %.2f GiB memory, "
        "%.2f GiB memory is still in use.",
        freed_bytes / GiB_bytes,
        used_bytes / GiB_bytes
    )


def worker_execute_dummy_batch(self) -> None:
    force_attention = self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    self.model_runner._dummy_run(
        num_tokens=1,
        uniform_decode=True,
        force_attention=force_attention
    )


NPUWorker.sleep = multi_level_sleep
NPUWorker.execute_dummy_batch = worker_execute_dummy_batch
