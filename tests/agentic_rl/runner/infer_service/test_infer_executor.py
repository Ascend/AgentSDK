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
from unittest.mock import patch, MagicMock, AsyncMock


def create_mock_engine():
    """Create a mock engine with all required async methods."""
    mock_engine = MagicMock()
    mock_engine.get_workload = AsyncMock(return_value={"workload": 100})
    mock_engine.cancel_requests = AsyncMock()
    mock_engine.completions = AsyncMock(return_value={"completion": "ok"})
    mock_engine.chat_completions = AsyncMock(return_value={"message": "ok"})
    mock_engine.launch_server = AsyncMock()
    mock_engine.wake_up = AsyncMock()
    mock_engine.sleep = AsyncMock()
    mock_engine.collective_rpc = AsyncMock()
    return mock_engine


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies(monkeypatch):
    """Mock all external dependencies for infer_executor tests."""
    # Mock the modules in sys.modules
    monkeypatch.setitem(sys.modules, "agentic_rl.runner.infer_service.infer_server.vllm_ray_infer_server", MagicMock())
    monkeypatch.setitem(sys.modules, "agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server", MagicMock())
    monkeypatch.setitem(sys.modules, "agentic_rl.runner.infer_service.infer_server.vllm_external_infer_server", MagicMock())
    monkeypatch.setitem(sys.modules, "agentic_rl.runner.infer_service.infer_server.vllm_proxy_infer_server", MagicMock())

    # Delete the infer_executor module if it's already imported
    monkeypatch.delitem(sys.modules, "agentic_rl.runner.infer_service.infer_executor", raising=False)

    # Now patch the classes and logger
    with (
        patch("agentic_rl.runner.infer_service.infer_executor.logger") as mock_logger,
        patch("agentic_rl.runner.infer_service.infer_server.vllm_ray_infer_server.VLLMRayInferServer") as mock_vllm_ray,
        patch("agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.VLLMMPInferServer") as mock_vllm_mp,
        patch("agentic_rl.runner.infer_service.infer_server.vllm_external_infer_server.VLLMExternalInferServer") as mock_vllm_external,
        patch("agentic_rl.runner.infer_service.infer_server.vllm_proxy_infer_server.VLLMProxyInferServer") as mock_vllm_proxy,
    ):
        mock_engine = create_mock_engine()

        mock_vllm_ray.return_value = mock_engine
        mock_vllm_mp.return_value = mock_engine
        mock_vllm_external.return_value = mock_engine
        mock_vllm_proxy.return_value = mock_engine

        yield {
            "logger": mock_logger,
            "vllm_ray": mock_vllm_ray,
            "vllm_mp": mock_vllm_mp,
            "vllm_external": mock_vllm_external,
            "vllm_proxy": mock_vllm_proxy,
            "mock_engine": mock_engine,
        }


class TestInferExecutor:
    """Tests for InferExecutor class."""

    def setup_method(self):
        """Setup method to import InferExecutor before each test."""
        from agentic_rl.runner.infer_service.infer_executor import InferExecutor
        self.InferExecutor = InferExecutor

    @pytest.mark.parametrize("engine_name, mock_key", [
        ("vllm_ray", "vllm_ray"),
        ("vllm_mp", "vllm_mp"),
        ("vllm_external", "vllm_external"),
        ("vllm_proxy", "vllm_proxy"),
    ])
    @pytest.mark.asyncio
    async def test_init_engines(self, engine_name, mock_key, mock_dependencies):
        """Test InferExecutor initialization with different engines."""
        executor = self.InferExecutor(engine_name, {"a": 1}, resource_set=MagicMock())

        mock_dependencies[mock_key].assert_called_once_with(a=1)
        assert executor.engine == mock_dependencies["mock_engine"]
        mock_dependencies["logger"].info.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_workload(self, mock_dependencies):
        """Test get_workload method."""
        executor = self.InferExecutor("vllm_ray", {}, resource_set=MagicMock())
        result = await executor.get_workload()

        mock_dependencies["mock_engine"].get_workload.assert_awaited_once()
        assert result == {"workload": 100}

    @pytest.mark.asyncio
    async def test_cancel_requests(self, mock_dependencies):
        """Test cancel_requests method."""
        executor = self.InferExecutor("vllm_ray", {}, resource_set=MagicMock())
        await executor.cancel_requests(request_id="test_id")

        mock_dependencies["mock_engine"].cancel_requests.assert_awaited_once_with(request_id="test_id")

    @pytest.mark.asyncio
    async def test_completions(self, mock_dependencies):
        """Test completions method."""
        executor = self.InferExecutor("vllm_ray", {}, resource_set=MagicMock())
        result = await executor.completions(prompt="test prompt")

        mock_dependencies["mock_engine"].completions.assert_awaited_once_with(prompt="test prompt")
        assert result == {"completion": "ok"}

    @pytest.mark.asyncio
    async def test_completions_with_exception(self, mock_dependencies):
        """Test completions method with exception."""
        mock_dependencies["mock_engine"].completions.side_effect = Exception("Test exception")

        executor = self.InferExecutor("vllm_ray", {}, resource_set=MagicMock())

        with pytest.raises(Exception) as excinfo:
            await executor.completions(prompt="test prompt")

        mock_dependencies["mock_engine"].completions.assert_awaited_once_with(prompt="test prompt")
        mock_dependencies["logger"].error.assert_called_once()
        assert "Test exception" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_chat_completions(self, mock_dependencies):
        """Test chat_completions method."""
        executor = self.InferExecutor("vllm_ray", {}, resource_set=MagicMock())
        result = await executor.chat_completions(
            messages=[{"role": "user", "content": "test"}]
        )

        mock_dependencies["mock_engine"].chat_completions.assert_awaited_once_with(
            messages=[{"role": "user", "content": "test"}]
        )
        assert result == {"message": "ok"}

    @pytest.mark.asyncio
    async def test_stream_chat_completions(self, mock_dependencies):
        """Test stream_chat_completions method."""
        async def async_gen():
            for i in range(3):
                yield {"chunk": f"test chunk {i}"}

        mock_dependencies["mock_engine"].stream_chat_completions.return_value = async_gen()

        executor = self.InferExecutor("vllm_ray", {}, resource_set=MagicMock())

        chunks = []
        async for chunk in executor.stream_chat_completions(messages=[{"role": "user", "content": "test"}]):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0] == {"chunk": "test chunk 0"}
        assert chunks[1] == {"chunk": "test chunk 1"}
        assert chunks[2] == {"chunk": "test chunk 2"}

    @pytest.mark.asyncio
    async def test_launch_wake_sleep(self, mock_dependencies):
        """Test launch_server, wake_up, and sleep methods."""
        executor = self.InferExecutor("vllm_ray", {}, resource_set=MagicMock())

        await executor.launch_server(port=8080)
        mock_dependencies["mock_engine"].launch_server.assert_awaited_once_with(port=8080)

        await executor.wake_up()
        mock_dependencies["mock_engine"].wake_up.assert_awaited_once()

        await executor.sleep()
        mock_dependencies["mock_engine"].sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_weights(self, mock_dependencies):
        """Test update_weights method."""
        executor = self.InferExecutor("vllm_ray", {}, resource_set=MagicMock())

        await executor.update_weights(path="/path/to/weights")
        mock_dependencies["mock_engine"].collective_rpc.assert_awaited_once_with("update_weights", args="/path/to/weights")

        await executor.update_weights(path=None)
        mock_dependencies["mock_engine"].collective_rpc.assert_awaited_with("update_weights", args=None)

    def test_init_unsupported_engine(self, mock_dependencies):
        """Test InferExecutor initialization with unsupported engine."""
        with pytest.raises(ValueError, match="unsupported_engine is not supported"):
            self.InferExecutor("unsupported_engine", {}, resource_set=MagicMock())

    @pytest.mark.asyncio
    async def test_chat_completions_with_exception(self, mock_dependencies):
        """Test chat_completions method with exception."""
        mock_dependencies["mock_engine"].chat_completions.side_effect = Exception("Chat exception")

        executor = self.InferExecutor("vllm_ray", {}, resource_set=MagicMock())

        with pytest.raises(Exception) as excinfo:
            await executor.chat_completions(messages=[{"role": "user", "content": "test"}])

        mock_dependencies["mock_engine"].chat_completions.assert_awaited_once_with(messages=[{"role": "user", "content": "test"}])
        mock_dependencies["logger"].error.assert_called_once()
        assert "Chat exception" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_stream_chat_completions_with_exception(self, mock_dependencies):
        """Test stream_chat_completions method with exception."""
        async def async_gen_with_error():
            yield {"chunk": "first"}
            raise Exception("Stream error")

        mock_dependencies["mock_engine"].stream_chat_completions.return_value = async_gen_with_error()

        executor = self.InferExecutor("vllm_ray", {}, resource_set=MagicMock())

        chunks = []
        with pytest.raises(Exception) as excinfo:
            async for chunk in executor.stream_chat_completions(messages=[{"role": "user", "content": "test"}]):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == {"chunk": "first"}
        mock_dependencies["logger"].error.assert_called_once()
        assert "Stream error" in str(excinfo.value)

