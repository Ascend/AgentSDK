#!/usr/bin/env python3
# coding=utf-8

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


# Standard library imports
from typing import Dict

# Internal imports
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.runner.infer_service.base_infer_server import BaseInferServer

logger = Loggers(__name__).get_logger()


class VLLMExternalInferServer(BaseInferServer):
    # TODO: Support "shared card" deployment mode, externally pass vLLMExecutor address
    def __init__(self, model_name, *args, **kwargs):
        self.server = None
        self.model_name = model_name

    async def chat_completions(self, request_data: Dict):
        import json
        from starlette.requests import Request
        from starlette.types import Scope

        def make_request_from_dict(data: dict) -> Request:
            scope: Scope = {
                "type": "http",
                "method": "POST",
                "path": "/",
                "headers": [],
                "query_string": b"",
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8000),
            }

            request = Request(scope)

            async def json_func():
                return data
    
            request.json = json_func
            return request

        response = await self.server.chat_completion(make_request_from_dict(request_data))
        return json.loads(response.body)

    async def stream_chat_completions(self, request_data: Dict):
        yield

    async def wake_up(self, *args, **kwargs):
        if self.server is None:
            from agentic_rl.base.utils.run_env import get_vllm_version
            import os
            os.environ['VLLM_VERSION'] = get_vllm_version()
            from agentic_rl.runner.infer_adapter.vllm.vllm_async_server import AsyncVLLMServer
            self.server = AsyncVLLMServer(*args, **kwargs)
            await self.server.init_engine()
            return

        await self.server.wake_up()

    async def sleep(self, *args, **kwargs):
        if self.server is None:
            return
        await self.server.sleep()

    async def get_workload(self):
        """get workload metrics from VLLMExternalInferServer"""
        return {}

    async def cancel_requests(self, *args, **kwargs):
        """cancel all requests in VLLMExternalInferServer"""
        pass