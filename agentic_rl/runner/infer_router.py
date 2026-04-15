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
import os
import random
import asyncio
from typing import Dict, AsyncGenerator, List, Optional

# Internal imports
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.runner.scheduler.req_scheduler import SchedulerFactory

logger = Loggers(__name__).get_logger()


class InferRouter:
    _router = None

    def __init__(self, infer_manager) -> None:
        self.infer_manager = infer_manager
        self.inited = False
        self._lock = asyncio.Lock()
        self.dp_size = int(os.getenv("VLLM_DP_SIZE", "1"))

    async def init(self, model_name):
        self.inited = True

    @classmethod
    async def create(cls) -> "InferRouter":
        if cls._router is None:
            from agentic_rl.runner.infer_manager import get_or_create_infer_manager
            infer_manager = await get_or_create_infer_manager()
            cls._router = InferRouter(infer_manager)
        return cls._router

    def get_application_id(self, request_id: str) -> str:
        return request_id.split('--')[0]

    async def stream_chat_completions(self, request_data: Dict) -> AsyncGenerator[str, None]:
        try:
            infer_instance = await self.infer_manager.get_instance.remote(request_data["model"])
            executor = random.choice(infer_instance.executor_list)
            async for response in executor.stream_execute_method.remote("stream_chat_completions", request_data=request_data):
                response = await response
                if not response:
                    continue
                logger.error(f"Stream chat completion: {response}")
                yield response
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    async def completions(self, request_data: Dict) -> Dict:
        infer_instance = await self.infer_manager.get_instance.remote(request_data["model"])

        await self.init(request_data["model"])
        executor = random.choice(infer_instance.executor_list)
        return await executor.execute_method.remote("completions", request_data=request_data)

    async def chat_completions(self, request_data: Dict) -> Dict:
        # todo secondary router scheduling??
        infer_instance = await self.infer_manager.get_instance.remote(request_data["model"])

        await self.init(request_data["model"])
        executor = random.choice(infer_instance.executor_list)
        return await executor.execute_method.remote("chat_completions", request_data=request_data)


    async def launch_server(self, model_name, kwargs_list: List[Dict] = None, *args, **kwargs):
        infer_instance = await self.infer_manager.get_instance.remote(model_name)
        kwargs_list = kwargs_list if kwargs_list else [{}] * len(infer_instance.executor_list)

        result_list = []
        for executor, kwargs in zip(infer_instance.executor_list, kwargs_list):
            result = await executor.execute_method.remote(method_name="launch_server", **kwargs)
            result_list.append(result)
        return result_list
    
    async def wake_up(self, model_name, kwargs_list: List[Dict] = None, *args, **kwargs):
        infer_instance = await self.infer_manager.get_instance.remote(model_name)
        kwargs_list = kwargs_list if kwargs_list else [{}] * len(infer_instance.executor_list)

        result_list = []
        for executor, kwargs in zip(infer_instance.executor_list, kwargs_list):
            result = await executor.execute_method.remote(method_name="wake_up", **kwargs)
            result_list.append(result)
        return result_list
    
    async def sleep(self, model_name, kwargs_list: List[Dict] = None, *args, **kwargs):
        infer_instance = await self.infer_manager.get_instance.remote(model_name)
        kwargs_list = kwargs_list if kwargs_list else [{}] * len(infer_instance.executor_list)

        result_list = []
        for executor, kwargs in zip(infer_instance.executor_list, kwargs_list):
            result = await executor.execute_method.remote(method_name="sleep", **kwargs)
            result_list.append(result)
        return result_list
    
    async def update_weights(self, model_name, kwargs_list: List[Dict] = None, *args, **kwargs):
        infer_instance = await self.infer_manager.get_instance.remote(model_name)
        kwargs_list = kwargs_list if kwargs_list else [{}] * len(infer_instance.executor_list)

        result_list = []
        for executor, kwargs in zip(infer_instance.executor_list, kwargs_list):
            result = await executor.execute_method.remote(method_name="update_weights", **kwargs)
            result_list.append(result)
        return result_list

    async def get_workload(self):
        infer_instance = await self.infer_manager.get_instance.remote(self.default_model_name)
        kwargs_list = [{}] * len(infer_instance.executor_list)

        executor_idx: int = 0
        result_list = {}
        for executor, kwargs in zip(infer_instance.executor_list, kwargs_list):
            result = await executor.execute_method.remote(method_name="get_workload", **kwargs)
            result_list[str(executor_idx)] = result
            executor_idx = executor_idx + 1
        return result_list

    async def cancel_requests(self):
        pass

    async def stop(self):
        await self.cancel_requests()

    def reset(self):
        pass
