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
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import os


@pytest.fixture(scope="function")
def mock_dependencies():
    import aura
    base_path = aura.__path__[0]
    infer_server_path = os.path.join(base_path, "runner", "infer_service", "infer_server")

    aura_runner_infer_service_infer_server = types.ModuleType(
        "aura.runner.infer_service.infer_server"
    )
    aura_runner_infer_service_infer_server.__path__ = [infer_server_path]

    fake_modules = {
        # --- aura package hierarchy ---
        "aura.runner": types.ModuleType("aura.runner"),
        "aura.runner.infer_service": types.ModuleType("aura.runner.infer_service"),
        "aura.runner.infer_service.infer_server": aura_runner_infer_service_infer_server,

        "aura.base": types.ModuleType("aura.base"),
        "aura.base.log": types.ModuleType("aura.base.log"),
        "aura.base.log.loggers": types.ModuleType("aura.base.log.loggers"),
        "aura.base.utils": types.ModuleType("aura.base.utils"),
        "aura.base.utils.run_env": types.ModuleType("aura.base.utils.run_env"),

        # scheduler package
        "aura.runner.scheduler": types.ModuleType("aura.runner.scheduler"),
        "aura.runner.scheduler.load_stat": types.ModuleType("aura.runner.scheduler.load_stat"),
        "aura.runner.scheduler.workload": types.ModuleType("aura.runner.scheduler.workload"),

        # base_infer_server module
        "aura.runner.infer_service.base_infer_server": types.ModuleType(
            "aura.runner.infer_service.base_infer_server"
        ),

        # --- vllm module tree ---
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.engine": types.ModuleType("vllm.v1.engine"),
        "vllm.v1.engine.async_llm": types.ModuleType("vllm.v1.engine.async_llm"),
        "vllm.entrypoints": types.ModuleType("vllm.entrypoints"),
        "vllm.entrypoints.openai": types.ModuleType("vllm.entrypoints.openai"),
        "vllm.entrypoints.openai.protocol": types.ModuleType("vllm.entrypoints.openai.protocol"),
        "vllm.entrypoints.openai.serving_chat": types.ModuleType("vllm.entrypoints.openai.serving_chat"),
        "vllm.entrypoints.openai.serving_models": types.ModuleType("vllm.entrypoints.openai.serving_models"),
        "vllm.config": types.ModuleType("vllm.config"),
    }

    # Add necessary attributes to the fake modules
    fake_modules["aura.base.log.loggers"].Loggers = MagicMock(return_value=MagicMock())
    fake_modules["aura.base.utils.run_env"].get_vllm_version = MagicMock(return_value="0.0.0")

    fake_modules["aura.runner.scheduler.load_stat"].WorkloadStatLogger = MagicMock()
    fake_modules["aura.runner.scheduler.load_stat"].vllm_log_stats_periodically = MagicMock()
    fake_modules["aura.runner.scheduler.workload"].InstanceWorkLoad = MagicMock()

    class FakeBaseInferServer:
        pass
    fake_modules["aura.runner.infer_service.base_infer_server"].BaseInferServer = FakeBaseInferServer

    mock_async_engine_args = MagicMock()
    fake_modules["vllm"].AsyncEngineArgs = mock_async_engine_args
    fake_modules["vllm.config"].AsyncEngineArgs = mock_async_engine_args

    fake_modules["vllm.v1.engine.async_llm"].AsyncLLM = MagicMock()
    fake_modules["vllm.entrypoints.openai.protocol"].ChatCompletionRequest = MagicMock()

    mock_openai_chat = MagicMock()
    fake_modules["vllm.entrypoints.openai.serving_chat"].OpenAIServingChat = mock_openai_chat

    mock_openai_models = MagicMock()
    fake_modules["vllm.entrypoints.openai.serving_models"].OpenAIServingModels = mock_openai_models
    mock_base_model_path = MagicMock()
    fake_modules["vllm.entrypoints.openai.serving_models"].BaseModelPath = mock_base_model_path

    # Configure engine behavior
    mock_engine = MagicMock()
    mock_engine.model_config = MagicMock()
    fake_modules["vllm.v1.engine.async_llm"].AsyncLLM.from_vllm_config.return_value = mock_engine
    mock_async_engine_args.return_value.create_engine_config.return_value = MagicMock()

    # Use patch.dict to inject fake modules; sys.modules is restored automatically on exit
    with patch.dict(sys.modules, fake_modules):
        # Force clearing the cached module so that it is re‑imported in the fake environment
        target_mod = "aura.runner.infer_service.infer_server.vllm_ray_infer_server"
        if target_mod in sys.modules:
            del sys.modules[target_mod]
        yield {
            "openai_chat": mock_openai_chat,
            "openai_models": mock_openai_models,
            "base_model_path": mock_base_model_path,
            "engine_args": mock_async_engine_args,
            "async_llm": fake_modules["vllm.v1.engine.async_llm"].AsyncLLM,
            "mock_engine": mock_engine,
            "vllm_log_stats": fake_modules["aura.runner.scheduler.load_stat"].vllm_log_stats_periodically,
            "instance_workload": fake_modules["aura.runner.scheduler.workload"].InstanceWorkLoad,
        }


class TestVLLMRayInferServer:
    """Tests for VLLMRayInferServer class."""

    @pytest.fixture(autouse=True)
    def _import_server(self, mock_dependencies):
        """Import the module under test before each test method to ensure mock environment is in place."""
        from aura.runner.infer_service.infer_server.vllm_ray_infer_server import VLLMRayInferServer
        self.VLLMRayInferServer = VLLMRayInferServer

    def test_init(self, mock_dependencies):
        """Test VLLMRayInferServer initialization."""
        with patch("asyncio.create_task") as mock_create_task:
            server = self.VLLMRayInferServer(model_name="test", model="xxx")
            assert server.engine == mock_dependencies["mock_engine"]
            assert server.openai_serving_chat is not None
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_completions(self, mock_dependencies):
        """Test chat_completions method."""
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"ok": 1}

        server = self.VLLMRayInferServer.__new__(self.VLLMRayInferServer)
        server.openai_serving_chat = AsyncMock()
        server.openai_serving_chat.create_chat_completion = AsyncMock(return_value=mock_response)

        result = await server.chat_completions({"a": 1})
        assert result == {"ok": 1}

    @pytest.mark.asyncio
    async def test_stream_chat_completions(self, mock_dependencies):
        """Test stream_chat_completions method."""
        async def fake_generator():
            yield "xxxxxx1"
            yield "xxxxxx2"

        server = self.VLLMRayInferServer.__new__(self.VLLMRayInferServer)
        server.openai_serving_chat = AsyncMock()
        server.openai_serving_chat.create_chat_completion = AsyncMock(return_value=fake_generator())

        results = []
        async for r in server.stream_chat_completions({"a": 1}):
            results.append(r)

        assert results == ["1", "2"]

    @pytest.mark.asyncio
    async def test_collective_rpc(self, mock_dependencies):
        """Test collective_rpc method."""
        server = self.VLLMRayInferServer.__new__(self.VLLMRayInferServer)
        server.engine = AsyncMock()
        server.engine.collective_rpc = AsyncMock(return_value=["ok"])

        result = await server.collective_rpc("method")
        assert result == ["ok"]

    @pytest.mark.asyncio
    async def test_cancel_requests(self, mock_dependencies):
        """Test cancel_requests method."""
        server = self.VLLMRayInferServer.__new__(self.VLLMRayInferServer)
        server.engine = AsyncMock()
        server.engine.abort = AsyncMock()

        await server.cancel_requests(requests=[1, 2, 3])
        server.engine.abort.assert_awaited_once_with([1, 2, 3])
