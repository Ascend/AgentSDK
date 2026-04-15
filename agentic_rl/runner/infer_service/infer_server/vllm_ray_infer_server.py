#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------


# Standard library imports
import os
import asyncio
from typing import Dict, Union, Callable, Optional, Tuple, Any
from typing_extensions import TypeVar

# vLLM imports
from vllm import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

# Internal imports
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.runner.infer_service.base_infer_server import BaseInferServer
from agentic_rl.runner.scheduler.load_stat import WorkloadStatLogger, vllm_log_stats_periodically
from agentic_rl.runner.scheduler.workload import InstanceWorkLoad

logger = Loggers(__name__).get_logger()

_R = TypeVar("_R", default=Any)


class VLLMRayInferServer(BaseInferServer):
    def __init__(self, model_name, **kwargs):
        from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.entrypoints.openai.serving_models import BaseModelPath
        from vllm.entrypoints.openai.serving_models import OpenAIServingModels

        # init async llm engine
        kwargs["worker_extension_cls"] = "agentic_rl.runner.infer_adapter.vllm.extension.custom_worker_extensions.CustomWorkerExtensions"
        logger.info(f"VLLMInferServer kwargs={kwargs}")
        dp_size = int(os.getenv("VLLM_DP_SIZE", "1"))
        self.ins_workload = InstanceWorkLoad(dp_size=dp_size)
        engine_args = AsyncEngineArgs(**kwargs)
        vllm_config = engine_args.create_engine_config()
        vllm_config.workload = self.ins_workload
        disable_log_stats = False
        self.engine = AsyncLLM.from_vllm_config(vllm_config, disable_log_stats=disable_log_stats,
            stat_loggers=[WorkloadStatLogger])
        if not disable_log_stats:
            asyncio.create_task(vllm_log_stats_periodically(self))

        # build serving chat
        model_path = kwargs["model"]
        model_config = self.engine.model_config
        base_model_paths = [BaseModelPath(name=model_name, model_path=model_path)]
        models = OpenAIServingModels(self.engine, model_config, base_model_paths)
        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            self.engine.model_config,
            models,
            "assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )

    async def chat_completions(self, request_data: Dict):
        request = ChatCompletionRequest(**request_data)
        response = await self.openai_serving_chat.create_chat_completion(request)

        return response.model_dump()

    async def stream_chat_completions(self, request_data: Dict):
        request = ChatCompletionRequest(**request_data)
        generator = await self.openai_serving_chat.create_chat_completion(request)

        async for response in generator:
            yield response[6:]

    async def collective_rpc(
            self,
            method: Union[str, Callable],
            timeout: Optional[float] = None,
            args: Tuple = (),
            kwargs: Optional[Dict[str, Any]] = None,
    ) -> list[_R]:
        logger.info(f"exec collective_rpc, method={method}, args={args}, kwargs={kwargs}")
        return await self.engine.collective_rpc(method, timeout, args, kwargs)

    async def get_workload(self):
        return self.ins_workload.to_dict()

    async def cancel_requests(self, *args, **kwargs):
        """cancel requests from AsyncVLLMServer"""
        request_list = kwargs["requests"]
        await self.engine.abort(request_list)