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
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import Handle
from vllm.utils import get_mp_context
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from loguru import logger

from agentic_rl.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat import vllm_output_statics

origin_worker_proc_init = WorkerProc.__init__


def worker_proc_init(
    self,
    vllm_config: VllmConfig,
    local_rank: int,
    rank: int,
    distributed_init_method: str,
    input_shm_handle: Handle,
):
    origin_worker_proc_init(self, vllm_config, local_rank, rank, distributed_init_method, input_shm_handle)
    vllm_output_statics.set_process_name(get_mp_context().current_process().name)


original_shutdown = WorkerProc.shutdown
def shutdown_patch(self):
    vllm_output_statics.write_stats_tofile()
    original_shutdown(self)


def execute_dummy_batch_patch(self) -> None:
    if self.is_sleeping:
        logger.info("Engine is currently sleeping, skipping dummy batch execution.")
    else:
        self.collective_rpc("execute_dummy_batch", unique_reply_rank=self.output_rank)


WorkerProc.__init__ = worker_proc_init
WorkerProc.shutdown = shutdown_patch
MultiprocExecutor.execute_dummy_batch = execute_dummy_batch_patch
