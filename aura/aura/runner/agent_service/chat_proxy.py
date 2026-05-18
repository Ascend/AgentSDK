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

import copy

from openai import AsyncOpenAI
from openai.types import Completion

from aura.base.log.loggers import Loggers
from aura.runner.infer_router import InferRouter

logger = Loggers(__name__).get_logger()


def make_proxy_create_chat(real_create, infer_service_params):
    logger.info(f"# create real_create: {real_create}, {infer_service_params=}")

    async def _proxy_async_create(*args, **kwargs):
        kwargs_update = copy.deepcopy(kwargs)
        kwargs_update = infer_service_params | kwargs_update
        # Use the smaller value between the max_tokens configuration value and the value calculated based on max_model_len
        if "max_tokens" in infer_service_params and "max_tokens" in kwargs_update:
            kwargs_update["max_tokens"] = min(infer_service_params["max_tokens"], kwargs_update["max_tokens"])

        # Completion APIs usually use prompts rather than messages
        kwargs_output = {key: value for key, value in kwargs_update.items() if key not in {'prompt'}}
        logger.info(f"# after update kwargs: {real_create}, {kwargs_output=}")

        if kwargs_update.get("stream", False):

            async def stream():
                infer_router = await InferRouter.create()
                async for chunk in infer_router.stream_chat_completions(kwargs_update):
                    import json
                    from openai.types.chat import ChatCompletionChunk

                    chunk_data = json.loads(chunk)
                    chunk = ChatCompletionChunk.model_validate(chunk_data)
                    yield chunk

            return stream()

        # ---- Non-streaming: returns a ChatCompletion object ----
        async def non_stream():
            infer_router = await InferRouter.create()
            resp = await infer_router.chat_completions(kwargs_update)
            from openai.types.chat import ChatCompletion

            return ChatCompletion(**resp)

        return await non_stream()

    return _proxy_async_create


def make_proxy_create_completion(real_create, infer_service_params):
    logger.info(f"# create real_create: {real_create}, {infer_service_params=}")

    async def _proxy_async_create(*args, **kwargs):
        kwargs_update = copy.deepcopy(kwargs)

        kwargs_update = infer_service_params | kwargs_update
        if "max_tokens" in infer_service_params and "max_tokens" in kwargs_update:
            kwargs_update["max_tokens"] = min(infer_service_params["max_tokens"], kwargs_update["max_tokens"])

        # Completion APIs usually use prompts rather than messages
        kwargs_output = {key: value for key, value in kwargs_update.items() if key not in {'prompt'}}
        logger.info(f"# after update kwargs: {real_create}, {kwargs_output=}")

        if kwargs_update.get("stream", False):

            async def stream():
                infer_router = await InferRouter.create()
                async for chunk in infer_router.stream_chat_completions(kwargs_update):
                    import json
                    from openai.types.chat import ChatCompletionChunk

                    chunk_data = json.loads(chunk)
                    chunk = ChatCompletionChunk.model_validate(chunk_data)
                    yield chunk

            return stream()

        async def non_stream():
            infer_router = await InferRouter.create()
            resp = await infer_router.completions(kwargs_update)
            return Completion(**resp)

        return await non_stream()

    return _proxy_async_create


_PATCHED = False


def patch_async_openai_global(infer_service_params):
    """
    Patch the completions.create method of AsyncOpenAI.
    """
    global _PATCHED

    if _PATCHED:
        return

    real_init = AsyncOpenAI.__init__
    _PATCHED = True

    def mock_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        try:
            # Change path: from self.chat.completions to self.completions
            _ = self.chat
            real_create_chat = self.chat.completions.create
            self.chat.completions.create = make_proxy_create_chat(real_create_chat, infer_service_params)
            real_create_completion = self.completions.create
            self.completions.create = make_proxy_create_completion(real_create_completion, infer_service_params)
        except Exception as e:
            logger.error(f"Failed to patch completions: {e}")
            pass

    AsyncOpenAI.__init__ = mock_init
