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
import json
import pytest
import sys
from unittest.mock import patch, MagicMock, AsyncMock


def create_mock_async_vllm_server():
    """Create a mock AsyncVLLMServer with all required async methods."""
    mock_server_instance = MagicMock()
    mock_server_instance.init_engine = AsyncMock()
    mock_server_instance.wake_up = AsyncMock()
    mock_server_instance.sleep = AsyncMock()
    mock_server_instance.chat_completion = AsyncMock()

    class MockAsyncVLLMServer:
        def __init__(self, *args, **kwargs):
            self._inner = mock_server_instance

        async def init_engine(self):
            await self._inner.init_engine()

        async def wake_up(self):
            await self._inner.wake_up()

        async def sleep(self):
            await self._inner.sleep()

        async def chat_completion(self, request):
            return await self._inner.chat_completion(request)

    return MockAsyncVLLMServer, mock_server_instance


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies(monkeypatch):
    """Mock all external dependencies for VLLMExternalInferServer tests."""
    monkeypatch.setitem(sys.modules, "agentic_rl.base.utils.run_env", MagicMock())
    monkeypatch.setitem(sys.modules, "agentic_rl.runner.infer_adapter.vllm.vllm_async_server", MagicMock())

    monkeypatch.delitem(sys.modules, "agentic_rl.runner.infer_service.infer_server.vllm_external_infer_server", raising=False)

    with (
        patch("agentic_rl.runner.infer_service.infer_server.vllm_external_infer_server.logger") as mock_logger,
        patch("agentic_rl.base.utils.run_env.get_vllm_version") as mock_get_vllm_version,
    ):
        mock_get_vllm_version.return_value = "0.4.0"

        MockAsyncVLLMServer, mock_server_instance = create_mock_async_vllm_server()

        mock_vllm_module = MagicMock()
        mock_vllm_module.AsyncVLLMServer = MockAsyncVLLMServer
        monkeypatch.setitem(sys.modules, "agentic_rl.runner.infer_adapter.vllm.vllm_async_server", mock_vllm_module)

        yield {
            "logger": mock_logger,
            "get_vllm_version": mock_get_vllm_version,
            "MockAsyncVLLMServer": MockAsyncVLLMServer,
            "mock_server_instance": mock_server_instance,
        }


class TestVLLMExternalInferServer:
    """Tests for VLLMExternalInferServer class."""

    def setup_method(self):
        """Setup method to import VLLMExternalInferServer before each test."""
        from agentic_rl.runner.infer_service.infer_server.vllm_external_infer_server import VLLMExternalInferServer
        self.VLLMExternalInferServer = VLLMExternalInferServer

    @pytest.mark.asyncio
    async def test_chat_completions_success(self, mock_dependencies):
        """Test chat_completions method with successful response."""
        mock_response = MagicMock()
        mock_response.body = json.dumps(
            {"choices": [{"message": {"content": "Hello World"}}]}
        )

        mock_dependencies["mock_server_instance"].chat_completion.return_value = mock_response

        server = self.VLLMExternalInferServer("test-model")
        server.server = mock_dependencies["MockAsyncVLLMServer"]()

        request_data = {"messages": [{"role": "user", "content": "test"}]}
        result = await server.chat_completions(request_data)

        assert result["choices"][0]["message"]["content"] == "Hello World"
        mock_dependencies["mock_server_instance"].chat_completion.assert_awaited_once()

        args, _ = mock_dependencies["mock_server_instance"].chat_completion.call_args
        request = args[0]
        assert hasattr(request, "json")
        data = await request.json()
        assert data == request_data

    @pytest.mark.asyncio
    async def test_chat_completions_invalid_json(self, mock_dependencies):
        """Test chat_completions method with invalid JSON response."""
        mock_response = MagicMock()
        mock_response.body = "invalid json"

        mock_dependencies["mock_server_instance"].chat_completion.return_value = mock_response

        server = self.VLLMExternalInferServer("test-model")
        server.server = mock_dependencies["MockAsyncVLLMServer"]()

        with pytest.raises(json.JSONDecodeError):
            await server.chat_completions({"messages": []})

    @pytest.mark.asyncio
    async def test_chat_completions_server_none(self, mock_dependencies):
        """Test chat_completions method when server is None."""
        server = self.VLLMExternalInferServer("test-model")
        server.server = None

        with pytest.raises(AttributeError):
            await server.chat_completions({"messages": []})

    @pytest.mark.asyncio
    async def test_stream_chat_completions(self, mock_dependencies):
        """Test stream_chat_completions method."""
        server = self.VLLMExternalInferServer("test-model")

        chunks = []
        async for chunk in server.stream_chat_completions({"messages": []}):
            chunks.append(chunk)

        assert chunks == [None]

    @pytest.mark.asyncio
    async def test_wake_up_first_call(self, mock_dependencies, monkeypatch):
        """Test wake_up method when server is None (first call)."""
        monkeypatch.delenv("VLLM_VERSION", raising=False)

        server = self.VLLMExternalInferServer("test-model")
        await server.wake_up(param1="value1")

        assert server.server is not None

        import os
        assert os.environ["VLLM_VERSION"] == "0.4.0"
        mock_dependencies["mock_server_instance"].init_engine.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wake_up_subsequent_call(self, mock_dependencies):
        """Test wake_up method when server already exists."""
        server = self.VLLMExternalInferServer("test-model")
        server.server = mock_dependencies["MockAsyncVLLMServer"]()

        await server.wake_up()

        mock_dependencies["mock_server_instance"].wake_up.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sleep_with_server(self, mock_dependencies):
        """Test sleep method when server exists."""
        server = self.VLLMExternalInferServer("test-model")
        server.server = mock_dependencies["MockAsyncVLLMServer"]()

        await server.sleep()

        mock_dependencies["mock_server_instance"].sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sleep_without_server(self, mock_dependencies):
        """Test sleep method when server is None."""
        server = self.VLLMExternalInferServer("test-model")
        server.server = None

        await server.sleep()

    @pytest.mark.asyncio
    async def test_get_workload(self, mock_dependencies):
        """Test get_workload method."""
        server = self.VLLMExternalInferServer("test-model")
        result = await server.get_workload()

        assert isinstance(result, dict)
        assert result == {}

    @pytest.mark.asyncio
    async def test_cancel_requests(self, mock_dependencies):
        """Test cancel_requests method."""
        server = self.VLLMExternalInferServer("test-model")
        result = await server.cancel_requests(requests=["req1"])

        assert result is None
