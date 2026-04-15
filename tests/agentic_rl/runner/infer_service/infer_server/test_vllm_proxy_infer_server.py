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
from unittest.mock import patch, MagicMock, AsyncMock, call


def create_mock_client():
    """Create a mock OpenAI client with all required async methods."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {"choices": [{"text": "Hello World"}]}
    mock_client.completions.create = AsyncMock(return_value=mock_completion)

    mock_chat_completion = MagicMock()
    mock_chat_completion.model_dump.return_value = {"choices": [{"message": {"content": "Hello World"}}]}
    mock_client.chat.completions.create = AsyncMock(return_value=mock_chat_completion)

    return mock_client


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies(monkeypatch):
    """Mock all external dependencies for VLLMProxyInferServer tests."""
    monkeypatch.delitem(sys.modules, "agentic_rl.runner.infer_service.infer_server.vllm_proxy_infer_server", raising=False)

    with (
        patch("agentic_rl.runner.infer_service.infer_server.vllm_proxy_infer_server.logger") as mock_logger,
        patch("agentic_rl.runner.infer_service.infer_server.vllm_proxy_infer_server.AsyncOpenAI") as mock_async_openai,
        patch("agentic_rl.runner.infer_service.infer_server.vllm_proxy_infer_server.requests") as mock_requests,
    ):
        mock_client = create_mock_client()
        mock_async_openai.return_value = mock_client

        yield {
            "logger": mock_logger,
            "async_openai": mock_async_openai,
            "requests": mock_requests,
            "mock_client": mock_client,
        }


class TestVLLMProxyInferServer:
    """Tests for VLLMProxyInferServer class."""

    def setup_method(self):
        """Setup method to import VLLMProxyInferServer before each test."""
        from agentic_rl.runner.infer_service.infer_server.vllm_proxy_infer_server import VLLMProxyInferServer
        self.VLLMProxyInferServer = VLLMProxyInferServer

    def test_init_single_server(self, mock_dependencies):
        """Test initialization with a single server."""
        server = self.VLLMProxyInferServer(
            model_name="test-model",
            chat_server="http://192.168.1.1:8080"
        )

        assert server.model_name == "test-model"
        assert server.chat_server_list == ["http://192.168.1.1:8080"]
        assert len(server.client_list) == 1

        mock_dependencies["async_openai"].assert_called_once_with(
            base_url="http://192.168.1.1:8080/v1/",
            api_key="EMPTY"
        )

    def test_init_multiple_servers(self, mock_dependencies):
        """Test initialization with multiple servers."""
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()
        mock_dependencies["async_openai"].side_effect = [mock_client1, mock_client2]

        servers = ["http://192.168.1.1:8080", "http://192.168.1.2:8080"]
        server = self.VLLMProxyInferServer(
            model_name="test-model",
            chat_server=servers
        )

        assert server.chat_server_list == servers
        assert len(server.client_list) == 2
        assert mock_dependencies["async_openai"].call_count == 2

        mock_dependencies["async_openai"].assert_has_calls([
            call(base_url="http://192.168.1.1:8080/v1/", api_key="EMPTY"),
            call(base_url="http://192.168.1.2:8080/v1/", api_key="EMPTY")
        ])

    def test_init_with_pd_servers(self, mock_dependencies):
        """Test initialization with prefill/decode servers."""
        server = self.VLLMProxyInferServer(
            model_name="test-model",
            chat_server="http://192.168.1.1:8080",
            prefill_server_list=["http://192.168.1.3:8080"],
            decode_server_list=["http://192.168.1.4:8080"]
        )

        assert server.prefill_server_list == ["http://192.168.1.3:8080"]
        assert server.decode_server_list == ["http://192.168.1.4:8080"]

    def test_choose_server(self, mock_dependencies):
        """Test _choose_server method."""
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()
        mock_dependencies["async_openai"].side_effect = [mock_client1, mock_client2]

        servers = ["http://192.168.1.1:8080", "http://192.168.1.2:8080"]
        server = self.VLLMProxyInferServer(
            model_name="test-model",
            chat_server=servers
        )

        with patch('random.choice', return_value=mock_client1):
            chosen_server = server._choose_server()
            assert chosen_server == mock_client1

    @pytest.mark.asyncio
    async def test_launch_server(self, mock_dependencies):
        """Test launch_server method."""
        server = self.VLLMProxyInferServer(
            model_name="test-model",
            chat_server="http://192.168.1.1:8080"
        )

        server.client_list = []

        await server.launch_server(
            model_name="new-model",
            chat_server=["http://192.168.1.3:8080", "http://192.168.1.4:8080"]
        )

        assert server.model_name == "new-model"
        assert server.chat_server_list == ["http://192.168.1.3:8080", "http://192.168.1.4:8080"]
        assert len(server.client_list) == 2

    @pytest.mark.asyncio
    async def test_completions(self, mock_dependencies):
        """Test completions method."""
        mock_completion = MagicMock()
        mock_completion.model_dump.return_value = {"choices": [{"text": "Hello World"}]}
        mock_dependencies["mock_client"].completions.create = AsyncMock(return_value=mock_completion)

        server = self.VLLMProxyInferServer(
            model_name="test-model",
            chat_server="http://192.168.1.1:8080"
        )

        server._choose_server = MagicMock(return_value=mock_dependencies["mock_client"])

        request_data = {
            "prompt": "test",
            "max_tokens": 100,
            "stream": True,
            "extra_headers": {"X-Test": "value"}
        }

        result = await server.completions(request_data)

        call_args = mock_dependencies["mock_client"].completions.create.call_args[1]

        assert "stream" not in call_args
        assert "extra_headers" not in call_args
        assert call_args["logprobs"] == 1
        assert call_args["extra_body"] == {"return_token_ids": True}
        assert call_args["model"] == "test-model"

        assert result == {"choices": [{"text": "Hello World"}]}

    @pytest.mark.asyncio
    async def test_chat_completions(self, mock_dependencies):
        """Test chat_completions method."""
        mock_completion = MagicMock()
        mock_completion.model_dump.return_value = {"choices": [{"message": {"content": "Hello World"}}]}
        mock_dependencies["mock_client"].chat.completions.create = AsyncMock(return_value=mock_completion)

        server = self.VLLMProxyInferServer(
            model_name="test-model",
            chat_server="http://192.168.1.1:8080"
        )

        server._choose_server = MagicMock(return_value=mock_dependencies["mock_client"])

        request_data = {
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 100,
            "stream": True,
            "extra_headers": {"X-Test": "value"}
        }

        result = await server.chat_completions(request_data)

        call_args = mock_dependencies["mock_client"].chat.completions.create.call_args[1]

        assert "stream" not in call_args
        assert "extra_headers" not in call_args
        assert call_args["logprobs"] == 1
        assert call_args["extra_body"] == {"return_token_ids": True}

        assert result == {"choices": [{"message": {"content": "Hello World"}}]}

    @pytest.mark.asyncio
    async def test_stream_chat_completions(self, mock_dependencies):
        """Test stream_chat_completions method."""
        mock_chunk1 = MagicMock()
        mock_chunk1.model_dump.return_value = {"choices": [{"delta": {"content": "Hello"}}]}

        mock_chunk2 = MagicMock()
        mock_chunk2.model_dump.return_value = {"choices": [{"delta": {"content": " World"}}]}

        async def mock_stream():
            yield mock_chunk1
            yield mock_chunk2

        mock_dependencies["mock_client"].chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        server = self.VLLMProxyInferServer(
            model_name="test-model",
            chat_server="http://192.168.1.1:8080"
        )

        server._choose_server = MagicMock(return_value=mock_dependencies["mock_client"])

        request_data = {
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 100
        }

        chunks = []
        async for chunk in server.stream_chat_completions(request_data):
            chunks.append(json.loads(chunk))

        call_args = mock_dependencies["mock_client"].chat.completions.create.call_args[1]

        assert call_args["stream"] is True
        assert call_args["model"] == "test-model"

        assert len(chunks) == 2
        assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_collective_rpc(self, mock_dependencies):
        """Test collective_rpc method."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_dependencies["requests"].post.return_value = mock_response

        server = self.VLLMProxyInferServer(
            model_name="test-model",
            chat_server=["http://192.168.1.1:8080"]
        )

        result = await server.collective_rpc(method="test_method")

        assert mock_dependencies["requests"].post.called
        assert result == []

    @pytest.mark.asyncio
    async def test_collective_rpc_with_pd_servers(self, mock_dependencies):
        """Test collective_rpc with prefill/decode servers."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_dependencies["requests"].post.return_value = mock_response

        server = self.VLLMProxyInferServer(
            model_name="test-model",
            chat_server="http://192.168.1.1:8080",
            prefill_server_list=["http://192.168.1.2:8080"],
            decode_server_list=["http://192.168.1.3:8080"]
        )

        result = await server.collective_rpc(method="test_method")

        assert mock_dependencies["requests"].post.call_count == 2
        assert result == []

    @pytest.mark.asyncio
    async def test_collective_rpc_non_200_status(self, mock_dependencies):
        """Test collective_rpc with non-200 status code."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status = MagicMock()
        mock_dependencies["requests"].post.return_value = mock_response

        server = self.VLLMProxyInferServer(
            model_name="test-model",
            chat_server=["http://192.168.1.1:8080"]
        )

        with pytest.raises(RuntimeError, match="response=500"):
            await server.collective_rpc(method="test_method")

    @pytest.mark.asyncio
    async def test_collective_rpc_exception(self, mock_dependencies):
        """Test collective_rpc with exception."""
        mock_dependencies["requests"].post.side_effect = Exception("Connection failed")

        server = self.VLLMProxyInferServer(
            model_name="test-model",
            chat_server=["http://192.168.1.1:8080"]
        )

        with pytest.raises(Exception, match="Connection failed"):
            await server.collective_rpc(method="test_method")

    @pytest.mark.asyncio
    async def test_get_workload(self, mock_dependencies):
        """Test get_workload method."""
        server = self.VLLMProxyInferServer(
            model_name="test-model",
            chat_server="http://192.168.1.1:8080"
        )

        assert await server.get_workload() is None

    @pytest.mark.asyncio
    async def test_cancel_requests(self, mock_dependencies):
        """Test cancel_requests method."""
        server = self.VLLMProxyInferServer(
            model_name="test-model",
            chat_server="http://192.168.1.1:8080"
        )

        await server.cancel_requests()
