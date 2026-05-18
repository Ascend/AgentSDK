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

from vllm.logger import logger
from vllm.utils import GiB_bytes
from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker.worker_v1 import NPUWorker
from vllm_ascend.utils import sleep_mode_enabled


def multi_level_sleep(self, level: int = 1) -> None:
    if not sleep_mode_enabled():
        raise ValueError("Sleep mode is not enabled. Please compile vllm-ascend with COMPILE_CUSTOM_KERNELS=1.")
    free_bytes_before_sleep = NPUPlatform.mem_get_info()[0]
    # Save the buffers before level 2 sleep
    if level == 2:
        model = self.model_runner.model
        self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}
    allocator = CaMemAllocator.get_instance()
    if level == 1:
        allocator.sleep(offload_tags=("weights",))
    elif level == 0:
        allocator.sleep(offload_tags=("kv_cache",))
    else:
        allocator.sleep(offload_tags=tuple())
    free_bytes_after_sleep, total = NPUPlatform.mem_get_info()
    freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
    used_bytes = total - free_bytes_after_sleep
    if freed_bytes < 0:
        raise ValueError("Memory usage increased after sleeping.")
    logger.info(
        "Sleep mode freed %.2f GiB memory, %.2f GiB memory is still in use.",
        freed_bytes / GiB_bytes,
        used_bytes / GiB_bytes,
    )


NPUWorker.sleep = multi_level_sleep
