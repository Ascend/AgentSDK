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
    """
    Apply patch for adaptor to verl
    """
    try:
        from verl.utils import vllm_utils
        from .patch_verl_init import mock_patch_vllm_moe_model_weight_loader

        vllm_utils.patch_vllm_moe_model_weight_loader = (
            mock_patch_vllm_moe_model_weight_loader
        )
    except ImportError:
        # Skip vllm patch if vllm is not available or has import errors
        pass

    try:
        from verl.experimental import agent_loop
        from .patch_verl_init import MockAgentLoopManager
        from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
        from .patch_verl_init import mock__exit__

        agent_loop.AgentLoopManager = MockAgentLoopManager
        FSDPVLLMShardingManager.__exit__ = mock__exit__
    except ImportError:
        # Skip agent loop patch if agent_loop is not available
        pass
