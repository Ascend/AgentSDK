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


import os
import logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.profiler import GPUMemoryLogger


def mock_patch_vllm_moe_model_weight_loader(model):
    pass 


class MockAgentLoopManager:
    def __init__(self, **kwargs):
        pass


@GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
def mock__exit__(self, exc_type, exc_value, traceback):
    if self.rollout_config.free_cache_engine:
        self.inference_engine.sleep(level=1)
        self.inference_engine.sleep(level=2)

    self.module.train()

    # add empty cache after each compute
    get_torch_device().empty_cache()

    # restore random states
    if self.device_mesh is not None:
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)
