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

from typing import Optional, Tuple, Union

import torch
from acl.rt import memcpy
from vllm.utils import is_pin_memory_available
from vllm_ascend.device_allocator.camem import CaMemAllocator, unmap_and_release


def camem_sleep(self, offload_tags: Optional[Union[Tuple[str, ...], str]] = None) -> None:
    """
    Put the allocator in sleep mode.
    All data in the memory allocation with the specified tag will be
    offloaded to CPU memory, and others will be discarded.
    :param offload_tags: The tags of the memory allocation that will be
        offloaded. The rest of the memory allocation will be discarded.
    """
    if offload_tags is None:
        # by default, allocated tensors are offloaded
        # when the allocator sleeps
        offload_tags = (CaMemAllocator.default_tag,)
    elif isinstance(offload_tags, str):
        offload_tags = (offload_tags,)

    if not isinstance(offload_tags, tuple):
        raise TypeError(f"offload_tags must be a tuple, got {type(offload_tags).__name__}")

    for ptr, data in self.pointer_to_data.items():
        handle = data.handle
        if data.tag in offload_tags:
            size_in_bytes = handle[1]
            cpu_backup_tensor = torch.empty(
                size_in_bytes, dtype=torch.uint8, device='cpu', pin_memory=is_pin_memory_available()
            )
            cpu_ptr = cpu_backup_tensor.data_ptr()
            ACL_MEMCPY_DEVICE_TO_HOST = 2
            dest_max = cpu_ptr + size_in_bytes * 2
            memcpy(cpu_ptr, dest_max, ptr, size_in_bytes, ACL_MEMCPY_DEVICE_TO_HOST)
            data.cpu_backup_tensor = cpu_backup_tensor
            # fix offload problems in RL
            unmap_and_release(handle)
    torch.cuda.empty_cache()


CaMemAllocator.sleep = camem_sleep
