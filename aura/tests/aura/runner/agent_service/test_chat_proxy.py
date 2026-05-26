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
import sys
import types
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Fixture: fake module tree for chat_proxy
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_chat_proxy_env():
    """Build a fully isolated environment for chat_proxy."""

    # ---- Fake openai ----
    fake_openai = types.ModuleType("openai")
    fake_openai.__path__ = []
    class FakeAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = MagicMock()
            self.chat.completions = MagicMock()
            self.completions = MagicMock()
    fake_openai.AsyncOpenAI = FakeAsyncOpenAI

    # ---- Fake openai.types / chat ----
    fake_openai_types = types.ModuleType("openai.types")
    fake_openai_types.__path__ = []
    class FakeCompletion:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    fake_openai_types.Completion = FakeCompletion

    fake_openai_types_chat = types.ModuleType("openai.types.chat")
    class FakeChatCompletion:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    class FakeChatCompletionChunk:
        pass
    fake_openai_types_chat.ChatCompletion = FakeChatCompletion
    fake_openai_types_chat.ChatCompletionChunk = FakeChatCompletionChunk

    # ---- Common router mock ----
    router_mock = MagicMock()
    router_mock.stream_chat_completions = MagicMock()
    router_mock.chat_completions = AsyncMock()
    router_mock.completions = AsyncMock()

    # ---- Fake aura.runner.infer_router ----
    fake_infer_router = types.ModuleType("aura.runner.infer_router")
    class FakeInferRouter:
        @classmethod
        async def create(cls):
            return router_mock
    fake_infer_router.InferRouter = FakeInferRouter

    # ---- Fake aura.base.log.loggers ----
    fake_loggers_mod = types.ModuleType("aura.base.log.loggers")
    mock_logger = MagicMock()
    fake_logger_instance = MagicMock()
    fake_logger_instance.get_logger.return_value = mock_logger
    fake_loggers_mod.Loggers = MagicMock(return_value=fake_logger_instance)

    # ---- Aura packages (to locate the real file) ----
    import os
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = [os.path.join(base, "runner")]
    fake_aura_runner_agent_service = types.ModuleType("aura.runner.agent_service")
    fake_aura_runner_agent_service.__path__ = [os.path.join(base, "runner/agent_service")]
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_log = types.ModuleType("aura.base.log")
    fake_aura_base_log.__path__ = []

    fakes = {
        "openai": fake_openai,
        "openai.types": fake_openai_types,
        "openai.types.chat": fake_openai_types_chat,
        "aura.runner.infer_router": fake_infer_router,
        "aura.base.log.loggers": fake_loggers_mod,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.agent_service": fake_aura_runner_agent_service,
        "aura.base": fake_aura_base,
        "aura.base.log": fake_aura_base_log,
    }

    target = "aura.runner.agent_service.chat_proxy"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.runner.agent_service.chat_proxy as mod
        mod._PATCHED = False
        yield {
            "mod": mod,
            "router_mock": router_mock,
            "logger_mock": mock_logger,
            "AsyncOpenAI": fake_openai.AsyncOpenAI,
            "ChatCompletion": fake_openai_types_chat.ChatCompletion,
            "ChatCompletionChunk": fake_openai_types_chat.ChatCompletionChunk,
            "Completion": fake_openai_types.Completion,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Helper async generator
# ---------------------------------------------------------------------------
async def async_gen(items):
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMakeProxyCreateChat:
    @pytest.mark.asyncio
    async def test_streaming_path(self, fake_chat_proxy_env):
        """Streaming: proxy returns an async generator that yields ChatCompletionChunks."""
        mod = fake_chat_proxy_env["mod"]
        router_mock = fake_chat_proxy_env["router_mock"]
        ChatCompletionChunk = fake_chat_proxy_env["ChatCompletionChunk"]

        router_mock.stream_chat_completions.return_value = async_gen(['{"id":"1"}'])
        ChatCompletionChunk.model_validate = MagicMock(return_value="chunk_obj")

        params = {"max_tokens": 1024}
        proxy_func = mod.make_proxy_create_chat("original_create", params)

        result = await proxy_func(stream=True, prompt="hello")
        chunks = [chunk async for chunk in result]
        assert chunks == ["chunk_obj"]

    @pytest.mark.asyncio
    async def test_non_streaming_path(self, fake_chat_proxy_env):
        """Non-streaming: proxy returns a ChatCompletion instance with the expected id."""
        mod = fake_chat_proxy_env["mod"]
        router_mock = fake_chat_proxy_env["router_mock"]

        router_mock.chat_completions.return_value = {"id": "chat_resp"}

        params = {"max_tokens": 100}
        proxy_func = mod.make_proxy_create_chat("original_create", params)

        result = await proxy_func(stream=False, prompt="hi")
        assert isinstance(result, fake_chat_proxy_env["ChatCompletion"])
        assert result.id == "chat_resp"

    @pytest.mark.asyncio
    async def test_max_tokens_min_logic(self, fake_chat_proxy_env):
        """max_tokens in both params and kwargs uses min value."""
        mod = fake_chat_proxy_env["mod"]
        router_mock = fake_chat_proxy_env["router_mock"]
        router_mock.chat_completions.return_value = {"id": "ok"}

        params = {"max_tokens": 300}
        proxy_func = mod.make_proxy_create_chat("orig", params)

        await proxy_func(max_tokens=500, stream=False)
        call_kwargs = router_mock.chat_completions.call_args[0][0]
        assert call_kwargs["max_tokens"] == 300

    @pytest.mark.asyncio
    async def test_no_max_tokens_in_params(self, fake_chat_proxy_env):
        """When max_tokens is not in params, kwarg value is kept as-is."""
        mod = fake_chat_proxy_env["mod"]
        router_mock = fake_chat_proxy_env["router_mock"]
        router_mock.chat_completions.return_value = {"id": "ok"}

        params = {}
        proxy_func = mod.make_proxy_create_chat("orig", params)

        await proxy_func(max_tokens=500, stream=False)
        call_kwargs = router_mock.chat_completions.call_args[0][0]
        assert call_kwargs["max_tokens"] == 500


class TestMakeProxyCreateCompletion:
    @pytest.mark.asyncio
    async def test_streaming_path(self, fake_chat_proxy_env):
        """Streaming for completion: yields ChatCompletionChunk objects."""
        mod = fake_chat_proxy_env["mod"]
        router_mock = fake_chat_proxy_env["router_mock"]
        ChatCompletionChunk = fake_chat_proxy_env["ChatCompletionChunk"]

        router_mock.stream_chat_completions.return_value = async_gen(['{"id":"cmp1"}'])
        ChatCompletionChunk.model_validate = MagicMock(return_value="chunk_obj")

        params = {"max_tokens": 200}
        proxy_func = mod.make_proxy_create_completion("orig_comp", params)

        result = await proxy_func(stream=True, prompt="hello")
        chunks = [chunk async for chunk in result]
        assert chunks == ["chunk_obj"]

    @pytest.mark.asyncio
    async def test_non_streaming_path(self, fake_chat_proxy_env):
        """Non-streaming completion returns a Completion instance."""
        mod = fake_chat_proxy_env["mod"]
        router_mock = fake_chat_proxy_env["router_mock"]
        Completion = fake_chat_proxy_env["Completion"]

        router_mock.completions.return_value = {"id": "complete"}

        params = {}
        proxy_func = mod.make_proxy_create_completion("orig_comp", params)

        result = await proxy_func(stream=False, prompt="test")
        assert isinstance(result, Completion)
        assert result.id == "complete"

    @pytest.mark.asyncio
    async def test_max_tokens_min_logic(self, fake_chat_proxy_env):
        """max_tokens min logic is applied for completion proxy as well."""
        mod = fake_chat_proxy_env["mod"]
        router_mock = fake_chat_proxy_env["router_mock"]
        router_mock.completions.return_value = {"id": "ok"}

        params = {"max_tokens": 150}
        proxy_func = mod.make_proxy_create_completion("orig", params)

        await proxy_func(max_tokens=300, stream=False)
        call_kwargs = router_mock.completions.call_args[0][0]
        assert call_kwargs["max_tokens"] == 150


class TestPatchAsyncOpenAIGlobal:
    def test_patch_successful(self, fake_chat_proxy_env):
        """Patch replaces AsyncOpenAI create methods with proxy functions."""
        mod = fake_chat_proxy_env["mod"]
        AsyncOpenAI = fake_chat_proxy_env["AsyncOpenAI"]
        params = {"max_tokens": 42}
        mod.patch_async_openai_global(params)

        assert mod._PATCHED is True
        client = AsyncOpenAI()
        assert not isinstance(client.chat.completions.create, MagicMock)
        assert not isinstance(client.completions.create, MagicMock)

    def test_already_patched_skips(self, fake_chat_proxy_env):
        """When _PATCHED is True, patch_async_openai_global returns immediately."""
        mod = fake_chat_proxy_env["mod"]
        mod._PATCHED = True
        original_init = mod.AsyncOpenAI.__init__
        mod.patch_async_openai_global({})
        assert mod.AsyncOpenAI.__init__ is original_init

    def test_patch_with_exception(self, fake_chat_proxy_env):
        """If an exception occurs during patching, logger.error is called and _PATCHED stays True."""
        mod = fake_chat_proxy_env["mod"]
        logger_mock = fake_chat_proxy_env["logger_mock"]

        # Cause an exception inside the try block of mock_init
        original_init = mod.AsyncOpenAI.__init__
        def evil_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            del self.chat  # will cause access to self.chat in try block to raise AttributeError

        mod.AsyncOpenAI.__init__ = evil_init
        mod.patch_async_openai_global({"model": "test"})
        assert mod._PATCHED is True

        # Instantiating triggers mock_init -> evil_init -> exception -> logger.error
        mod.AsyncOpenAI()
        logger_mock.error.assert_called_once()
