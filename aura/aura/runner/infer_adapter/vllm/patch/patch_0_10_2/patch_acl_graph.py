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

from contextlib import ExitStack
from unittest.mock import patch

import torch
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.utils import weak_ref_tensors

from vllm_ascend.compilation.acl_graph import ACLGraphEntry, ACLGraphWrapper


def __call__(self, *args, **kwargs):
    forward_context = get_forward_context()
    batch_descriptor = forward_context.batch_descriptor
    aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

    if aclgraph_runtime_mode == CUDAGraphMode.NONE or aclgraph_runtime_mode != self.runtime_mode:
        return self.runnable(*args, **kwargs)

    if batch_descriptor not in self.concrete_aclgraph_entries:
        self.concrete_aclgraph_entries[batch_descriptor] = ACLGraphEntry(batch_descriptor=batch_descriptor)

    entry = self.concrete_aclgraph_entries[batch_descriptor]

    if entry.aclgraph is None:
        if self.aclgraph_options.debug_log_enable:
            logger.debug("Capturing a aclgraph on (%s,%s)", self.runtime_mode.name, entry.batch_descriptor)
        validate_cudagraph_capturing_enabled()

        input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
        entry.input_addresses = input_addresses
        aclgraph = torch.npu.NPUGraph()

        with ExitStack() as stack:
            if self.aclgraph_options.gc_disable:
                stack.enter_context(patch("gc.collect", lambda: None))
                stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

            device_id = torch.npu.current_device()
            torch.npu.set_device(device_id)
            tmp_pool = () if self.graph_pool is None else (self.graph_pool,)

            torch.distributed.barrier()
            aclgraph.capture_begin(*tmp_pool)
            output = self.runnable(*args, **kwargs)
            aclgraph.capture_end()
            torch.distributed.barrier()

            if self.aclgraph_options.weak_ref_output:
                output = weak_ref_tensors(output)

        entry.output = weak_ref_tensors(output)
        entry.aclgraph = aclgraph

        compilation_counter.num_cudagraph_captured += 1

        return output

    if self.is_debugging_mode:
        new_input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
        if new_input_addresses != entry.input_addresses:
            raise RuntimeError(
                f"Input addresses for aclgraphs are different during replay. "
                f"Expected {entry.input_addresses}, got {new_input_addresses}"
            )

    logger.info_once("Replaying aclgraph")
    entry.aclgraph.replay()
    return entry.output


ACLGraphWrapper.__call__ = __call__
