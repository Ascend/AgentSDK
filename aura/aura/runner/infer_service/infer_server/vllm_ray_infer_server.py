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

# Copyright 2024 Bytedance Ltd. and/or its affiliates

import os
import asyncio
from typing import Dict, Union, Callable, Optional, Tuple, Any
from typing_extensions import TypeVar

from vllm import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

from aura.base.log.loggers import Loggers
from aura.runner.infer_service.base_infer_server import BaseInferServer
from aura.runner.scheduler.load_stat import WorkloadStatLogger, vllm_log_stats_periodically
from aura.runner.scheduler.workload import InstanceWorkLoad

logger = Loggers(__name__).get_logger()

_R = TypeVar("_R", default=Any)


class VLLMRayInferServer(BaseInferServer):
    def __init__(self, model_name, **kwargs):
        from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.entrypoints.openai.serving_models import BaseModelPath
        from vllm.entrypoints.openai.serving_models import OpenAIServingModels

        from aura.base.utils.run_env import get_vllm_version

        os.environ["VLLM_VERSION"] = get_vllm_version()
        logger.info(f"vllm version: {get_vllm_version()}")

        # init async llm engine
        kwargs["worker_extension_cls"] = (
            "aura.runner.infer_adapter.vllm.extension.custom_worker_extensions.CustomWorkerExtensions"
        )
        logger.info(f"VLLMInferServer kwargs={kwargs}")
        dp_size = int(os.getenv("VLLM_DP_SIZE", "1"))
        self.ins_workload = InstanceWorkLoad(dp_size=dp_size)
        engine_args = AsyncEngineArgs(**kwargs)
        vllm_config = engine_args.create_engine_config()
        vllm_config.workload = self.ins_workload
        disable_log_stats = False
        self.engine = AsyncLLM.from_vllm_config(
            vllm_config, disable_log_stats=disable_log_stats, stat_loggers=[WorkloadStatLogger]
        )
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
