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
from typing import Optional, Tuple, Union

import torch
from acl.rt import memcpy
from vllm.utils import is_pin_memory_available
from vllm_ascend.device_allocator.camem import CaMemAllocator, unmap_and_release

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__)


def _call_sleep(ptr, data, offload_tags) -> None:
    handle = data.handle
    if data.tag in offload_tags:
        if len(handle) < 2:
            raise RuntimeError("handle length less than 2 is not valid.")
        size_in_bytes = handle[1]
        cpu_backup_tensor = torch.empty(size_in_bytes,
                                        dtype=torch.uint8,
                                        device='cpu',
                                        pin_memory=is_pin_memory_available())
        cpu_ptr = cpu_backup_tensor.data_ptr()
        acl_memcpy_device_to_host = 2
        dest_max = cpu_ptr + size_in_bytes * 2
        memcpy(cpu_ptr, dest_max, ptr, size_in_bytes, acl_memcpy_device_to_host)
        data.cpu_backup_tensor = cpu_backup_tensor
        unmap_and_release(handle)


def ca_mem_allocator_sleep(self, offload_tags: Optional[Union[Tuple[str, ...], str]] = None) -> None:
    """
    Patch vllm-ascend CaMemAllocator.sleep
    """
    if offload_tags is None:
        offload_tags = (CaMemAllocator.default_tag,)
    elif isinstance(offload_tags, str):
        offload_tags = (offload_tags,)

    if not isinstance(offload_tags, tuple):
        logger.error("offload_tags must be a tuple")
        raise TypeError("offload_tags must be a tuple")
    for tag in offload_tags:
        if not isinstance(tag, str):
            logger.error("offload_tags must be a tuple of strings")
            raise TypeError("offload_tags must be a tuple of strings")

    if not hasattr(self, "pointer_to_data") or not isinstance(self.pointer_to_data, dict):
        logger.error("CaMemAllocator not initialized, pointer_to_data is not valid.")
        raise AttributeError("CaMemAllocator not initialized, pointer_to_data is not valid.")

    for ptr, data in self.pointer_to_data.items():
        try:
            _call_sleep(ptr, data, offload_tags)
        except AttributeError as e:
            logger.error(f"CaMemAllocator sleep failed by missing attribute, error:{e}")
            raise AttributeError(f"CaMemAllocator sleep failed by missing attribute") from e
        except RuntimeError as e:
            logger.error(f"CaMemAllocator sleep failed by RuntimeError, error:{e}")
            raise RuntimeError(f"CaMemAllocator sleep failed by RuntimeError") from e
        except Exception as e:
            logger.error(f"Unexpected error occurred when CaMemAllocator sleep, error:{e}")
            raise RuntimeError(f"Unexpected error occurred when CaMemAllocator sleep") from e
