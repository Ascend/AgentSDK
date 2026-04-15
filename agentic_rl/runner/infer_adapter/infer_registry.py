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

from agentic_rl.base.utils.globals import is_pd_separate
from agentic_rl.runner.infer_adapter.vllm.vllm_async_server import AsyncVLLMServer
from agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd import AsyncVLLMServerPDSep

class InferBackendRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name, cls):
        self._registry[name] = cls

    def get_class(self, name):
        return self._registry.get(name)


registry = InferBackendRegistry()
registry.register("vllm", AsyncVLLMServer)
registry.register("vllm_pd", AsyncVLLMServerPDSep)

def async_server_class(infer_backend: str):
    if infer_backend == "vllm" and is_pd_separate():
        return registry.get_class("vllm_pd")
    return registry.get_class(infer_backend)
