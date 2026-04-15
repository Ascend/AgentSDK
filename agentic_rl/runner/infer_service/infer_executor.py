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


from agentic_rl.base.execution.executor import public_api, Executor
from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class InferExecutor(Executor):
    def __init__(self, engine, engine_kwargs, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if engine == "vllm_ray":
            from agentic_rl.runner.infer_service.infer_server.vllm_ray_infer_server import VLLMRayInferServer
            self.engine = VLLMRayInferServer(**engine_kwargs)
        elif engine == "vllm_mp":
            from agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server import VLLMMPInferServer
            self.engine = VLLMMPInferServer(**engine_kwargs)
        elif engine == "vllm_external":
            from agentic_rl.runner.infer_service.infer_server.vllm_external_infer_server import VLLMExternalInferServer
            self.engine = VLLMExternalInferServer(**engine_kwargs)
        elif engine == "vllm_proxy":
            from agentic_rl.runner.infer_service.infer_server.vllm_proxy_infer_server import VLLMProxyInferServer
            self.engine = VLLMProxyInferServer(**engine_kwargs)
        else:
            raise ValueError(f"{engine} is not supported.")
        logger.info(f"InferExecutor: engine={engine} is initialized.")

    @public_api(name="get_workload")
    async def get_workload(self, *args, **kwargs):
        """get workload metrics from InferExecutor"""
        return await self.engine.get_workload()

    async def cancel_requests(self, *args, **kwargs):
        """cancel all requests in InferExecutor"""
        await self.engine.cancel_requests(*args, **kwargs)

    @public_api(name="completions")
    async def completions(self, *args, **kwargs):
        try:
            return await self.engine.completions(*args, **kwargs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(e)
            raise e

    @public_api(name="chat_completions")
    async def chat_completions(self, *args, **kwargs):
        try:
            return await self.engine.chat_completions(*args, **kwargs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(e)
            raise e

    @public_api(name="stream_chat_completions", is_stream=True)
    async def stream_chat_completions(self, *args, **kwargs):
        try:
            async for chat_response in self.engine.stream_chat_completions(*args, **kwargs):
                yield chat_response
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(e)
            raise e

    @public_api(name="launch_server")
    async def launch_server(self, *args, **kwargs):
        await self.engine.launch_server(*args, **kwargs)

    @public_api(name="wake_up")
    async def wake_up(self, *args, **kwargs):
        await self.engine.wake_up(*args, **kwargs)

    @public_api(name="sleep")
    async def sleep(self, *args, **kwargs):
        await self.engine.sleep(*args, **kwargs)

    @public_api(name="update_weights")
    async def update_weights(self, *args, **kwargs):
        path = kwargs["path"]
        await self.engine.collective_rpc("update_weights", args=path)
