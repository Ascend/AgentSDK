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
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import ExitStack
from unittest.mock import patch
from typing import Any, Union

import torch
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from vllm_ascend.compilation.acl_graph import ACLGraphEntry, ACLGraphWrapper


def weak_ref_tensor(tensor: Any) -> Any:
    """
    Create a weak reference to a tensor.
    The new tensor will share the same data as the original tensor,
    but will not keep the original tensor alive.
    """
    if isinstance(tensor, torch.Tensor):
        # return torch.ops._C_ascend.weak_ref_tensor(tensor)
        return tensor.as_strided(tensor.size(), tensor.stride(), tensor.storage_offset())
    else:
        return tensor


def weak_ref_tensors(
    tensors: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor]],
) -> Union[torch.Tensor, list[Any], tuple[Any], Any]:
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.

    This function should be used in the following scenario:
    When a tensor is created during graph capture, and it's held by a method
    that's not part of the graph, we don't really need to store it, but we
    **do need** its buffer pointer. If we don't handle this, it cannot
    be garbage collected, leading to a memory leak. To avoid this,
    we should create a weak reference to the tensor.
    """
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    raise ValueError("Invalid type for tensors")


def __call__(self, *args, **kwargs):
    forward_context = get_forward_context()
    batch_descriptor = forward_context.batch_descriptor
    aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

    if aclgraph_runtime_mode == CUDAGraphMode.NONE or aclgraph_runtime_mode != self.runtime_mode:
        # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
        # running without aclgraphs.
        # We do not trigger capture/replay if the runtime mode is not
        # matches. This enables properly dispatching to the correct
        # CUDAGraphWrapper when nesting multiple instances with different
        # runtime modes.
        return self.runnable(*args, **kwargs)

    if batch_descriptor not in self.concrete_aclgraph_entries:
        # create a new entry for this batch descriptor
        self.concrete_aclgraph_entries[batch_descriptor] = ACLGraphEntry(batch_descriptor=batch_descriptor)

    entry = self.concrete_aclgraph_entries[batch_descriptor]

    if entry.aclgraph is None:
        if self.aclgraph_options.debug_log_enable:
            # Since we capture aclgraph for many different shapes and
            # capturing is fast, we don't need to log it for every
            # shape. E.g. we only log it for the first subgraph in
            # piecewise mode.
            logger.debug("Capturing a aclgraph on (%s,%s)", self.runtime_mode.name, entry.batch_descriptor)
        # validate that aclgraph capturing is legal at this point.
        validate_cudagraph_capturing_enabled()

        input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
        entry.input_addresses = input_addresses
        aclgraph = torch.npu.NPUGraph()

        with ExitStack() as stack:
            if self.aclgraph_options.gc_disable:
                # during every model forward for piecewise aclgraph
                # mode, we will capture many pieces of aclgraphs
                # (roughly one per layer). running gc again and again
                # across layers will make the aclgraph capture very slow.
                # therefore, we only run gc for the first graph,
                # and disable gc for the rest of the graphs.
                stack.enter_context(patch("gc.collect", lambda: None))
                stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

            # mind-exploding: carefully manage the reference and memory.
            forward_context.capturing = True
            device_id = torch.npu.current_device()
            torch.npu.set_device(device_id)
            tmp_pool = () if self.graph_pool is None else (self.graph_pool,)

            torch.distributed.barrier()
            aclgraph.capture_begin(*tmp_pool)
            # `output` is managed by pytorch's aclgraph pool
            output = self.runnable(*args, **kwargs)
            if self.aclgraph_options.weak_ref_output:
                # by converting it to weak ref,
                # the original `output` will immediately be released
                # to save memory. It is only safe to do this for
                # the last graph in piecewise aclgraph mode, because
                # the output of the last graph will not be used by
                # any other acl graph.
                output = weak_ref_tensors(output)
            aclgraph.capture_end()
            torch.distributed.barrier()

        # here we always use weak ref for the output
        # to save memory
        entry.output = weak_ref_tensors(output)
        entry.aclgraph = aclgraph

        compilation_counter.num_cudagraph_captured += 1

        # important: we need to return the output, rather than
        # the weak ref of the output, so that pytorch can correctly
        # manage the memory during acl graph capture
        return output

    if self.is_debugging_mode:
        # check if the input addresses are the same
        new_input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
        if new_input_addresses != entry.input_addresses:
            raise ValueError(
                f"Input addresses for aclgraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

    logger.info_once("Replaying aclgraph")
    entry.aclgraph.replay()
    return entry.output


ACLGraphWrapper.__call__ = __call__
torch.ops._C_ascend.weak_ref_tensor = weak_ref_tensor
