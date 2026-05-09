#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co.; Ltd.
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

import gc
from typing import Optional, Tuple, Union

import torch
from acl.rt import memcpy  # type: ignore # noqa: F401
from vllm.logger import logger
from vllm.utils import is_pin_memory_available
from vllm_ascend.device_allocator.camem import CaMemAllocator, unmap_and_release


def camem_sleep(
    self,
    offload_tags: Optional[Union[Tuple[str, ...], str]] = None
) -> None:
    """
    Put the allocator in sleep mode.
    All data in the memory allocation with the specified tag will be
    offloaded to CPU memory, and others will be discarded.
    :param offload_tags: The tags of the memory allocation that will be
        offloaded. The rest of the memory allocation will be discarded.
    """
    if offload_tags is None:
        offload_tags = (CaMemAllocator.default_tag,)
    elif isinstance(offload_tags, str):
        offload_tags = (offload_tags,)

    if not isinstance(offload_tags, tuple):
        raise TypeError(
            f"offload_tags must be a tuple, str, or None, "
            f"but got {type(offload_tags).__name__}"
        )

    for ptr, data in self.pointer_to_data.items():
        handle = data.handle
        if data.tag in offload_tags:
            size_in_bytes = handle[1]
            cpu_backup_tensor = torch.empty(
                size_in_bytes,
                dtype=torch.uint8,
                device='cpu',
                pin_memory=is_pin_memory_available())
            cpu_ptr = cpu_backup_tensor.data_ptr()
            ACL_MEMCPY_DEVICE_TO_HOST = 2
            dest_max = cpu_ptr + size_in_bytes * 2
            memcpy(cpu_ptr, dest_max, ptr, size_in_bytes, ACL_MEMCPY_DEVICE_TO_HOST)
            data.cpu_backup_tensor = cpu_backup_tensor
            unmap_and_release(handle)


CaMemAllocator.sleep = camem_sleep
