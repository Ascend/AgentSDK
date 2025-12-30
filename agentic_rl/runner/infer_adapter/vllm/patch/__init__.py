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


def apply_patch():
    from vllm_ascend.device_allocator.camem import CaMemAllocator
    from vllm_ascend.worker.worker_v1 import NPUWorker

    from agentic_rl.runner.infer_adapter.vllm.patch.ca_mem_sleep import ca_mem_allocator_sleep
    from agentic_rl.runner.infer_adapter.vllm.patch.worker_v1_sleep import worker_v1_sleep

    CaMemAllocator.sleep = ca_mem_allocator_sleep
    NPUWorker.sleep = worker_v1_sleep
