#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import os
import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Fixture: fake module tree
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_router_env():
    # ---- Get real aura package path ----
    # Import the real aura package temporarily to get its path (must be done before faking)
    import aura
    real_aura_path = aura.__path__

    # Construct the expected real path for aura.serve
    # Assumes aura/serve/ is a subdirectory of the first entry in aura.__path__
    real_serve_path = [os.path.join(real_aura_path[0], "serve")] if real_aura_path else []

    # ---- Package stubs with real paths for loading router.py ----
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = real_aura_path                    # point to real aura package

    fake_aura_serve = types.ModuleType("aura.serve")
    fake_aura_serve.__path__ = real_serve_path             # point to real serve directory

    # Other package stubs remain empty (no real files needed)
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_exceptions = types.ModuleType("aura.base.exceptions")
    fake_aura_base_exceptions.__path__ = []
    fake_aura_base_log = types.ModuleType("aura.base.log")
    fake_aura_base_log.__path__ = []

    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = []
    fake_aura_runner_agent_engine = types.ModuleType("aura.runner.agent_engine_wrapper")
    fake_aura_runner_agent_engine.__path__ = []

    # ---- fastapi ----
    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.Request = MagicMock
    class FakeAPIRouter:
        def __init__(self, *args, **kwargs):
            self.routes = []
        def post(self, path: str):
            def decorator(func):
                self.routes.append((path, func))
                return func
            return decorator
    fake_fastapi.APIRouter = FakeAPIRouter

    # ---- sse_starlette ----
    fake_sse = types.ModuleType("sse_starlette")
    fake_sse.EventSourceResponse = MagicMock(name="EventSourceResponse")

    # ---- aura.base.exceptions.exceptions ----
    fake_exceptions = types.ModuleType("aura.base.exceptions.exceptions")
    fake_exceptions.async_raise_http_exception = lambda fn: fn

    # ---- aura.base.log.loggers ----
    fake_loggers = types.ModuleType("aura.base.log.loggers")
    fake_loggers.Loggers = MagicMock(return_value=MagicMock(name="logger"))

    # ---- aura.runner.agent_engine_wrapper.base_engine_wrapper ----
    fake_agent_engine = types.ModuleType("aura.runner.agent_engine_wrapper.base_engine_wrapper")
    class FakeAgentTask:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    fake_agent_engine.AgentTask = FakeAgentTask

    # ---- aura.runner.agent_router ----
    fake_agent_router_mod = types.ModuleType("aura.runner.agent_router")
    class FakeAgentRouter:
        @classmethod
        async def create(cls):
            return cls()

        def stream_generate_trajectory(self, task):
            return AsyncMock(name="async_gen")
    fake_agent_router_mod.AgentRouter = FakeAgentRouter

    # ---- aura.runner.infer_router ----
    fake_infer_router_mod = types.ModuleType("aura.runner.infer_router")
    class FakeInferRouter:
        @classmethod
        async def create(cls):
            return cls()

        def stream_chat_completions(self, data):
            return AsyncMock(name="stream_chat")

        async def chat_completions(self, data):
            return MagicMock(name="chat")

        def stream_completions(self, data):
            return AsyncMock(name="stream_comp")

        async def completions(self, data):
            return MagicMock(name="comp")
    fake_infer_router_mod.InferRouter = FakeInferRouter

    # ---- Inject all fakes ----
    all_fakes = {
        # Packages (some with real paths)
        "aura": fake_aura,
        "aura.base": fake_aura_base,
        "aura.base.exceptions": fake_aura_base_exceptions,
        "aura.base.log": fake_aura_base_log,
        "aura.runner": fake_aura_runner,
        "aura.runner.agent_engine_wrapper": fake_aura_runner_agent_engine,
        "aura.serve": fake_aura_serve,       # real path, so router.py can be found
        # Leaf modules (all faked)
        "fastapi": fake_fastapi,
        "sse_starlette": fake_sse,
        "aura.base.exceptions.exceptions": fake_exceptions,
        "aura.base.log.loggers": fake_loggers,
        "aura.runner.agent_engine_wrapper.base_engine_wrapper": fake_agent_engine,
        "aura.runner.agent_router": fake_agent_router_mod,
        "aura.runner.infer_router": fake_infer_router_mod,
    }

    with patch.dict(sys.modules, all_fakes):
        import aura.serve.router as router_module   # now loads the real router.py
        yield {
            "module": router_module,
            "router": router_module.router,
            "EventSourceResponse": fake_sse.EventSourceResponse,
            "fastapi": fake_fastapi,
        }

# ---------------------------------------------------------------------------
# Helper: build an async mock request with given JSON data
# ---------------------------------------------------------------------------
def make_request(json_data: dict):
    req = MagicMock(name="Request")
    req.json = AsyncMock(return_value=json_data)
    return req


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestRouterRegistration:
    def test_routes_registered(self, fake_router_env):
        router = fake_router_env["router"]
        paths = [path for path, _ in router.routes]
        assert "/agent/invoke" in paths
        assert "/v1/chat/completions" in paths
        assert "/v1/completions" in paths


class TestAgentInvoke:
    @pytest.mark.asyncio
    async def test_agent_invoke_returns_sse(self, fake_router_env):
        module = fake_router_env["module"]
        EventSourceResponse = fake_router_env["EventSourceResponse"]

        req = make_request({"prompt": "hello"})
        result = await module.agent_invoke(req)

        EventSourceResponse.assert_called_once()
        assert result is EventSourceResponse.return_value


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_streaming_mode(self, fake_router_env):
        module = fake_router_env["module"]
        EventSourceResponse = fake_router_env["EventSourceResponse"]

        req = make_request({"stream": True, "messages": []})
        result = await module.completions(req)

        EventSourceResponse.assert_called_once()
        assert result is EventSourceResponse.return_value

    @pytest.mark.asyncio
    async def test_non_streaming_mode(self, fake_router_env):
        module = fake_router_env["module"]
        EventSourceResponse = fake_router_env["EventSourceResponse"]

        req = make_request({"stream": False, "messages": []})
        result = await module.completions(req)

        EventSourceResponse.assert_not_called()
        assert isinstance(result, MagicMock)  # chat_completions returns a mock


class TestCompletions:
    @pytest.mark.asyncio
    async def test_streaming_mode(self, fake_router_env):
        module = fake_router_env["module"]
        EventSourceResponse = fake_router_env["EventSourceResponse"]

        req = make_request({"stream": True, "prompt": ""})
        result = await module.chat_completions(req)

        EventSourceResponse.assert_called_once()
        assert result is EventSourceResponse.return_value

    @pytest.mark.asyncio
    async def test_non_streaming_mode(self, fake_router_env):
        module = fake_router_env["module"]
        EventSourceResponse = fake_router_env["EventSourceResponse"]

        req = make_request({"stream": False, "prompt": ""})
        result = await module.chat_completions(req)

        EventSourceResponse.assert_not_called()
        assert isinstance(result, MagicMock)  # completions returns a mock
