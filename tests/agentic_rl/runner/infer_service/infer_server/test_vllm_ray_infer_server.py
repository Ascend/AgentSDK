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
import pytest
import sys
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies(monkeypatch):
    """Mock all external dependencies for VLLMRayInferServer tests."""
    mock_load_stat = MagicMock()
    mock_load_stat.WorkloadStatLogger = MagicMock()
    mock_load_stat.vllm_log_stats_periodically = MagicMock()

    monkeypatch.setitem(sys.modules, "agentic_rl.runner.scheduler.load_stat", mock_load_stat)
    monkeypatch.setitem(sys.modules, "vllm", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm.v1", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.protocol", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.serving_chat", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.serving_models", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm.config", MagicMock())

    monkeypatch.delitem(sys.modules, "agentic_rl.runner.infer_service.infer_server.vllm_ray_infer_server", raising=False)

    with (
        patch("agentic_rl.runner.infer_service.infer_server.vllm_ray_infer_server.asyncio.create_task") as mock_create_task,
        patch("agentic_rl.runner.infer_service.infer_server.vllm_ray_infer_server.vllm_log_stats_periodically") as mock_log,
        patch("agentic_rl.runner.infer_service.infer_server.vllm_ray_infer_server.InstanceWorkLoad") as mock_workload,
        patch("agentic_rl.runner.infer_service.infer_server.vllm_ray_infer_server.AsyncEngineArgs") as mock_engine_args,
        patch("vllm.v1.engine.async_llm.AsyncLLM") as mock_async_llm,
        patch("vllm.entrypoints.openai.serving_chat.OpenAIServingChat") as mock_chat,
        patch("vllm.entrypoints.openai.serving_models.OpenAIServingModels") as mock_models,
        patch("vllm.entrypoints.openai.serving_models.BaseModelPath") as mock_base_model_path,
    ):
        mock_engine_args.return_value.create_engine_config.return_value = MagicMock()

        mock_engine = MagicMock()
        mock_engine.model_config = MagicMock()
        mock_async_llm.from_vllm_config.return_value = mock_engine

        yield {
            "create_task": mock_create_task,
            "log": mock_log,
            "workload": mock_workload,
            "engine_args": mock_engine_args,
            "async_llm": mock_async_llm,
            "chat": mock_chat,
            "models": mock_models,
            "base_model_path": mock_base_model_path,
            "mock_engine": mock_engine,
            "load_stat": mock_load_stat,
        }


class TestVLLMRayInferServer:
    """Tests for VLLMRayInferServer class."""

    def setup_method(self):
        """Setup method to import VLLMRayInferServer before each test."""
        from agentic_rl.runner.infer_service.infer_server.vllm_ray_infer_server import VLLMRayInferServer
        self.VLLMRayInferServer = VLLMRayInferServer

    def test_init(self, mock_dependencies):
        """Test VLLMRayInferServer initialization."""
        server = self.VLLMRayInferServer(model_name="test", model="xxx")

        assert server.engine == mock_dependencies["mock_engine"]
        assert server.openai_serving_chat is not None
        mock_dependencies["create_task"].assert_called_once()

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
