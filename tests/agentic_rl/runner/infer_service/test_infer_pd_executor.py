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
import os
import pytest
import sys
from types import ModuleType
from unittest.mock import patch, MagicMock, AsyncMock


def create_mock_executor_module():
    """Create mock executor module with Executor and public_api."""
    mock_executor_module = ModuleType("executor")

    class DummyExecutor:
        def __init__(self, *args, **kwargs):
            self.resource_set = kwargs.get("resource_set")

    def public_api(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

    mock_executor_module.Executor = DummyExecutor
    mock_executor_module.public_api = public_api
    return mock_executor_module


def create_mock_infer_executor_module():
    """Create mock infer_executor module with InferExecutor."""
    mock_infer_executor_module = ModuleType("infer_executor")

    class MockInferExecutor:
        def __init__(self, engine, engine_kwargs, *args, **kwargs):
            self.resource_set = kwargs.get("resource_set")
            self.engine = AsyncMock()

    mock_infer_executor_module.InferExecutor = MockInferExecutor
    return mock_infer_executor_module


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies(monkeypatch):
    """Mock all external dependencies for infer_pd_executor tests."""
    # Create mock modules
    mock_executor_module = create_mock_executor_module()
    mock_infer_executor_module = create_mock_infer_executor_module()

    # Mock the modules in sys.modules
    monkeypatch.setitem(sys.modules, "agentic_rl.base.execution.executor", mock_executor_module)
    monkeypatch.setitem(sys.modules, "agentic_rl.runner.infer_service.infer_executor", mock_infer_executor_module)

    # Delete the infer_pd_executor module if it's already imported
    monkeypatch.delitem(sys.modules, "agentic_rl.runner.infer_service.infer_pd_executor", raising=False)

    # Now patch the logger
    with patch("agentic_rl.runner.infer_service.infer_pd_executor.logger") as mock_logger:
        yield {
            "logger": mock_logger,
        }


class TestInferPrefillExecutor:
    """Tests for InferPrefillExecutor class."""

    def setup_method(self):
        """Setup method to import InferPrefillExecutor before each test."""
        from agentic_rl.runner.infer_service.infer_pd_executor import InferPrefillExecutor
        self.InferPrefillExecutor = InferPrefillExecutor

    @pytest.mark.asyncio
    async def test_chat_completions(self, mock_dependencies):
        """Test chat_completions method."""
        executor = self.InferPrefillExecutor(
            engine="vllm_ray",
            engine_kwargs={},
            resource_set=MagicMock()
        )

        executor.engine.chat_completions = AsyncMock(return_value={"ok": True})

        request_data = {"prompt": "hi", "max_tokens": 10}

        result = await executor.chat_completions(request_data=request_data)

        called_kwargs = executor.engine.chat_completions.call_args.kwargs
        req = called_kwargs["request_data"]

        assert req["max_tokens"] == 1
        assert req["stream"] is False
        assert "kv_transfer_params" in req

        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_chat_completions_with_max_completion_tokens(self, mock_dependencies):
        """Test chat_completions method with max_completion_tokens."""
        executor = self.InferPrefillExecutor(
            engine="vllm_ray",
            engine_kwargs={},
            resource_set=MagicMock()
        )

        executor.engine.chat_completions = AsyncMock(return_value={"ok": True})

        request_data = {"prompt": "hi", "max_completion_tokens": 100}

        result = await executor.chat_completions(request_data=request_data)

        called_kwargs = executor.engine.chat_completions.call_args.kwargs
        req = called_kwargs["request_data"]

        assert req["max_completion_tokens"] == 1
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_chat_completions_with_stream_options(self, mock_dependencies):
        """Test chat_completions method with stream_options."""
        executor = self.InferPrefillExecutor(
            engine="vllm_ray",
            engine_kwargs={},
            resource_set=MagicMock()
        )

        executor.engine.chat_completions = AsyncMock(return_value={"ok": True})

        request_data = {"prompt": "hi", "stream_options": {"include_usage": True}}

        result = await executor.chat_completions(request_data=request_data)

        called_kwargs = executor.engine.chat_completions.call_args.kwargs
        req = called_kwargs["request_data"]

        assert "stream_options" not in req
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_stream_not_supported(self, mock_dependencies):
        """Test stream_chat_completions raises NotImplementedError."""
        executor = self.InferPrefillExecutor(
            engine="vllm_ray",
            engine_kwargs={},
            resource_set=MagicMock()
        )

        with pytest.raises(NotImplementedError):
            await executor.stream_chat_completions()


class TestInferDecodeExecutor:
    """Tests for InferDecodeExecutor class."""

    def setup_method(self):
        """Setup method to import InferDecodeExecutor before each test."""
        from agentic_rl.runner.infer_service.infer_pd_executor import InferDecodeExecutor
        self.InferDecodeExecutor = InferDecodeExecutor

    def test_init(self, mock_dependencies):
        """Test InferDecodeExecutor initialization."""
        with patch(
                "agentic_rl.runner.infer_service.infer_executor.InferExecutor.__init__",
                return_value=None
        ) as mock_super:

            self.InferDecodeExecutor(
                engine="vllm_ray",
                engine_kwargs={},
                resource_set=MagicMock()
            )

            mock_super.assert_called_once()


class TestInferPDSepExecutor:
    """Tests for InferPDSepExecutor class."""

    def setup_method(self):
        """Setup method to import InferPDSepExecutor before each test."""
        from agentic_rl.runner.infer_service.infer_pd_executor import InferPDSepExecutor
        self.InferPDSepExecutor = InferPDSepExecutor

    @pytest.fixture
    def mock_pd_executor(self):
        """Create a mock InferPDSepExecutor instance for testing."""
        executor = self.InferPDSepExecutor(
            engine="vllm_ray",
            engine_kwargs={},
            resource_set=MagicMock(),
            p_num=1,
            d_num=1
        )
        return executor

    @pytest.mark.asyncio
    async def test_chat_completions_pd(self, mock_pd_executor, mock_dependencies):
        """Test chat_completions method for PD executor."""
        executor = mock_pd_executor

        prefill = MagicMock()
        decode = MagicMock()

        prefill.chat_completions.remote = AsyncMock(
            return_value={"kv_transfer_params": {"k": "v"}}
        )
        decode.chat_completions.remote = AsyncMock(
            return_value={"final": "ok"}
        )

        executor.executors = {
            "prefill": [prefill],
            "decode": [decode]
        }

        request_data = {}

        result = await executor.chat_completions(request_data=request_data)

        prefill.chat_completions.remote.assert_awaited_once()
        decode.chat_completions.remote.assert_awaited_once()

        assert request_data["kv_transfer_params"] == {"k": "v"}
        assert result == {"final": "ok"}

    @pytest.mark.asyncio
    async def test_stream_chat_completions_pd(self, mock_pd_executor, mock_dependencies):
        """Test stream_chat_completions method for PD executor."""
        executor = mock_pd_executor

        prefill = MagicMock()
        decode = MagicMock()

        prefill.chat_completions.remote = AsyncMock(
            return_value={"kv_transfer_params": {"k": "v"}}
        )

        async def async_gen():
            for i in range(3):
                yield AsyncMock(return_value={"chunk": i})()

        decode.stream_chat_completions.remote = MagicMock(
            return_value=async_gen()
        )

        executor.executors = {
            "prefill": [prefill],
            "decode": [decode]
        }

        request_data = {}

        chunks = []
        async for c in executor.stream_chat_completions(request_data=request_data):
            chunks.append(c)

        assert len(chunks) == 3
        assert chunks[0] == {"chunk": 0}
        assert request_data["kv_transfer_params"] == {"k": "v"}

    def test_alloc_resources_from_ranktable(self, mock_pd_executor, mock_dependencies):
        """Test alloc_resources_from_ranktable method."""
        executor = mock_pd_executor

        executor.get_ranktable = MagicMock(return_value={
            "prefill_device_list": [{"server_id": "1.1.1.1"}],
            "decode_device_list": [{"server_id": "2.2.2.2"}],
        })

        executor.get_node_info = MagicMock(return_value=[
            {"node_id": "n1", "node_ip": "1.1.1.1"},
            {"node_id": "n2", "node_ip": "2.2.2.2"},
        ])

        result = executor.alloc_resources_from_ranktable()

        assert len(result["prefill"]) == 1
        assert len(result["decode"]) == 1

    @pytest.mark.asyncio
    async def test_create_single_executor(self, mock_pd_executor, mock_dependencies):
        """Test create_single_infer_executor method."""
        executor = mock_pd_executor

        executor.resource_set = MagicMock()
        executor.resource_set.model_copy.return_value = MagicMock()

        with patch("ray.remote") as mock_remote:
            with patch("agentic_rl.runner.infer_service.infer_pd_executor.NodeAffinitySchedulingStrategy"):
                mock_actor = MagicMock()
                mock_actor.options.return_value.remote.return_value = mock_actor
                mock_actor.setup.remote = AsyncMock()

                mock_remote.return_value = mock_actor

                result = await executor.create_single_infer_executor(
                    "prefill", 0, "node1"
                )

                assert result == mock_actor
                mock_actor.setup.remote.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wake_sleep(self, mock_pd_executor, mock_dependencies):
        """Test wake_up and sleep methods."""
        executor = mock_pd_executor

        engine = MagicMock()
        engine.wake_up = AsyncMock()
        engine.sleep = AsyncMock()

        executor.executors = {
            "prefill": [engine],
            "decode": [engine]
        }

        await executor.wake_up()
        await executor.sleep()

        assert engine.wake_up.await_count == 2
        assert engine.sleep.await_count == 2

    @pytest.mark.asyncio
    async def test_setup(self, mock_pd_executor, mock_dependencies):
        """Test setup method."""
        executor = mock_pd_executor

        executor.alloc_resources_from_ranktable = MagicMock(return_value={
            "prefill": [{"node_id": "n1"}],
            "decode": [{"node_id": "n2"}],
        })

        mock_prefill_executor = MagicMock()
        mock_decode_executor = MagicMock()

        executor.create_single_infer_executor = AsyncMock()
        executor.create_single_infer_executor.side_effect = [mock_prefill_executor, mock_decode_executor]

        await executor.setup()

        assert len(executor.executors["prefill"]) == 1
        assert len(executor.executors["decode"]) == 1
        assert executor.executors["prefill"][0] == mock_prefill_executor
        assert executor.executors["decode"][0] == mock_decode_executor

    def test_get_ranktable(self, mock_pd_executor, tmp_path, mock_dependencies):
        """Test get_ranktable method."""
        executor = mock_pd_executor

        ranktable_data = {
            "prefill_device_list": [{"server_id": "1.1.1.1"}],
            "decode_device_list": [{"server_id": "2.2.2.2"}],
        }

        ranktable_file = tmp_path / "ranktable.json"
        with open(ranktable_file, 'w') as f:
            json.dump(ranktable_data, f)

        with patch.dict(os.environ, {"DISAGGREGATED_PREFILL_RANK_TABLE_PATH": str(ranktable_file)}):
            result = executor.get_ranktable()

        assert result == ranktable_data

    def test_get_ranktable_no_env(self, mock_pd_executor, mock_dependencies):
        """Test get_ranktable method without environment variable."""
        executor = mock_pd_executor

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Can't find ranktable file"):
                executor.get_ranktable()

    def test_get_node_info(self, mock_pd_executor, mock_dependencies):
        """Test get_node_info method."""
        executor = mock_pd_executor

        mock_nodes = [
            {
                "NodeID": "node-1",
                "NodeManagerAddress": "192.168.1.1",
                "Resources": {"CPU": 8, "GPU": 1}
            },
            {
                "NodeID": "node-2",
                "NodeManagerAddress": "192.168.1.2",
                "Resources": {"CPU": 8, "GPU": 1}
            }
        ]

        with patch("ray.nodes", return_value=mock_nodes):
            result = executor.get_node_info()

        assert len(result) == 2
        assert result[0]["node_id"] == "node-1"
        assert result[0]["node_ip"] == "192.168.1.1"
        assert result[1]["node_id"] == "node-2"

    def test_alloc_resources(self, mock_pd_executor, mock_dependencies):
        """Test alloc_resources method."""
        executor = mock_pd_executor

        mock_instance_conf = MagicMock()
        mock_instance_conf.executor_num = 1
        mock_instance_conf.resource_info = [{"NPU": 8}]
        mock_instance_conf.role = "prefill"

        mock_conf = MagicMock()
        mock_conf.infer_pd_instances = [mock_instance_conf]

        mock_node_info = [
            {"node_id": "n1", "node_ip": "1.1.1.1"},
            {"node_id": "n2", "node_ip": "2.2.2.2"},
        ]

        executor.get_node_info = MagicMock(return_value=mock_node_info)
        executor.conf = mock_conf

        with patch.dict(os.environ, {"ASCEND_PLATFORM": "A2"}):
            with patch("agentic_rl.base.conf.conf.AgenticRLConf") as mock_conf_cls:
                mock_conf_cls.load_config.return_value = mock_conf
                result = executor.alloc_resources()

        assert len(result["prefill"]) == 1
        assert result["prefill"][0]["node_id"] == "n1"

    def test_alloc_resources_insufficient_nodes(self, mock_pd_executor, mock_dependencies):
        """Test alloc_resources method with insufficient nodes."""
        executor = mock_pd_executor

        mock_instance_conf = MagicMock()
        mock_instance_conf.executor_num = 2
        mock_instance_conf.resource_info = [{"NPU": 8}]
        mock_instance_conf.role = "prefill"

        mock_conf = MagicMock()
        mock_conf.infer_pd_instances = [mock_instance_conf]

        mock_node_info = [
            {"node_id": "n1", "node_ip": "1.1.1.1"},
        ]

        executor.get_node_info = MagicMock(return_value=mock_node_info)
        executor.conf = mock_conf

        with patch.dict(os.environ, {"ASCEND_PLATFORM": "A2"}):
            with patch("agentic_rl.base.conf.conf.AgenticRLConf") as mock_conf_cls:
                mock_conf_cls.load_config.return_value = mock_conf
                with pytest.raises(ValueError, match="Resources are insufficient"):
                    executor.alloc_resources()
