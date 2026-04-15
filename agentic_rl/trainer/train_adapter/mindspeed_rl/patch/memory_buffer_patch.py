# -*- coding: utf-8 -*-
#
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
# 
from typing import Dict, Optional

import torch

from mindspeed_rl.workers.resharding.memory_buffer import build_memory_buffer, MemoryBuffer, calc_padded_numel
from agentic_rl.runner.infer_adapter.vllm.extension.custom_worker_extensions import resolve_device
from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def __init__(
    self,
    numel: int,
    numel_padded: int,
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> None:
    """Initialize a MemoryBuffer with padded allocation for alignment."""
    self.numel = numel
    self.numel_padded = numel_padded
    self.dtype = dtype
    self.device = resolve_device(device)
    self.data = torch.zeros(
        self.numel_padded, dtype=self.dtype,
        device=self.device, requires_grad=False,
    )
    self.tensor_indices = {}


def copy_by_name(self, param_name: str, param: torch.Tensor) -> None:
    """Copy a parameter tensor into the buffer slot identified by name.

    Args:
        param_name: Name key used to look up the buffer region.
        param: Source tensor whose data will be copied.

    Raises:
        RuntimeError: If the buffer region cannot be reshaped to match param.
    """
    buffer_tensor = self.get_by_name(param_name)
    try:
        buffer_tensor = buffer_tensor.view(param.shape)
    except RuntimeError as err:
        raise RuntimeError(
            f"Shape mismatch for weight '{param_name}'"
        ) from err
    buffer_tensor.copy_(param)


def build_experts_memory_buffer_patch(
    experts_weight_buffer_meta: Dict[str, Dict],
    experts_memory_expend_N: int,
    device: Optional[torch.device] = None,
) -> Dict[torch.dtype, MemoryBuffer]:
    """Build the experts memory buffer given experts_weight_buffer_meta.

    Args:
        experts_weight_buffer_meta: Mapping from name to a dict
            containing shape and dtype of the tensors.
        experts_memory_expend_N: Expansion factor for expert memory.
        device: Target device for the buffer allocation.

    Returns:
        A large memory buffer for each dtype that can hold all the tensors.
    """
    device = resolve_device(device)
    experts_memory_buffers = {}
    total_numel_map: Dict[torch.dtype, int] = {}

    for _, meta_info in sorted(experts_weight_buffer_meta.items()):
        shape = meta_info['shape']
        shape = torch.Size([experts_memory_expend_N, shape[0], shape[1], shape[2]])
        dtype = meta_info['dtype']

        if not isinstance(shape, torch.Size):
            raise TypeError("Shape must be an instance of torch.Size")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be an instance of torch.dtype")
        if dtype not in total_numel_map:
            total_numel_map[dtype] = 0

        tmp_numel = calc_padded_numel(shape, dtype)
        total_numel_map[dtype] += tmp_numel

    for dtype, total_numel in total_numel_map.items():
        experts_memory_buffers[dtype] = MemoryBuffer(
            total_numel, total_numel, dtype, device=device,
        )

    current_index_map: Dict[torch.dtype, int] = {}
    for name, meta_info in sorted(experts_weight_buffer_meta.items()):
        shape = meta_info['shape']
        shape = torch.Size([experts_memory_expend_N, shape[0], shape[1], shape[2]])
        dtype = meta_info['dtype']
        buffer = experts_memory_buffers[dtype]
        tensor_size = calc_padded_numel(shape, dtype)

        start_index = current_index_map.get(dtype, 0)
        current_index_map[dtype] = start_index + tensor_size

        buffer.tensor_indices[name] = (start_index, shape)

    return experts_memory_buffers


def rebuild_with_device(self, device: Optional[torch.device] = None) -> None:
    """Rebuild or validate memory buffers for the given device.

    Args:
        device: Target device. Resolved via ``resolve_device`` if None.
    """
    dev = resolve_device(device)
    if self.memory_buffers is None:
        self.memory_buffers = build_memory_buffer(self.weight_buffer_meta)
    else:
        for dt, mb in self.memory_buffers.items():
            if mb.data.device != dev:
                logger.warning(f"Buffer dtype {dt} resides on {mb.data.device}, expected {dev}.")