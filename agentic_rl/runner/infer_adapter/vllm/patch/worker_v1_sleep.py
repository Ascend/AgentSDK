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
from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.platform import NPUPlatform

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__)


def _mem_get_info():
    try:
        free_bytes, _ = NPUPlatform.mem_get_info()
    except RuntimeError as e:
        logger.error(f"get memory info failed with runtime error: {e}")
        raise RuntimeError("get memory info failed with runtime error") from e
    except Exception as e:
        logger.error(f"Unexpected error occurred when get memory info, error: {e}")
        raise RuntimeError("Unexpected error occurred when get memory info") from e

    if free_bytes < 0:
        logger.error("free_bytes should be positive.")
        raise ValueError("free_bytes should be positive.")

    return free_bytes


def worker_v1_sleep(self, level: int = 1) -> None:
    """
    Patch vllm_ascend npu v1 worker's sleep method, support different level sleep
    """
    if level != 0 and level != 1 and level != 2:
        logger.error("sleep level should be 0 or 1 or 2")
        raise ValueError("sleep level should be 0 or 1 or 2")

    try:
        NPUPlatform.set_device(self.device)
    except AttributeError as e:
        logger.error(f"set device failed with attribute error: {e}")
        raise AttributeError("set device failed with attribute") from e
    except Exception as e:
        logger.error(f"Unexpected error occurred when set device, error: {e}")
        raise RuntimeError("Unexpected error occurred when set device") from e

    free_bytes_before_sleep = _mem_get_info()

    try:
        allocator = CaMemAllocator.get_instance()
        if level == 1:
            allocator.sleep(offload_tags=("weights",))
        elif level == 2:
            allocator.sleep(offload_tags=("kv_cache",))
        else:
            allocator.sleep(offload_tags=tuple())
    except RuntimeError as e:
        logger.error(f"sleep failed with runtime error: {e}")
        raise RuntimeError("sleep failed with runtime error") from e
    except Exception as e:
        logger.error(f"Unexpected error occurred when sleeping, error: {e}")
        raise RuntimeError("Unexpected error occurred when sleeping") from e

    free_bytes_after_sleep = _mem_get_info()

    freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
    if freed_bytes < 0:
        logger.error("Memory usage increased after sleeping.")
        raise RuntimeError("Memory usage increased after sleeping.")
