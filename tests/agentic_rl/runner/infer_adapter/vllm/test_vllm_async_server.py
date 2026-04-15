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

import os
import sys
import json
import unittest
import types
import importlib
import cloudpickle
from unittest.mock import patch, MagicMock, AsyncMock


# ============================================================
# Fake module builder for dependency isolation
# ============================================================

# List of module names to be replaced with MagicMock
MOCK_MODULES = [
    "ray", "ray.state", "ray.runtime_context", "vllm",
    "vllm.engine.arg_utils", "vllm.entrypoints.openai.protocol",
    "vllm.entrypoints.openai.serving_chat", "vllm.entrypoints.openai.serving_completion",
    "vllm.entrypoints.openai.serving_models", "vllm.v1.engine.async_llm",
    "vllm.v1.executor.abstract", "vllm.config", "vllm.executor",
    "agentic_rl.base.log.loggers", "agentic_rl.runner.infer_adapter.async_server",
    "agentic_rl.runner.scheduler.workload", "agentic_rl.runner.scheduler.load_stat",
    "uvicorn", "fastapi", "omegaconf", "torch", "transformers"
]


def _build_fake_modules():
    """Create and return a dict of fake modules to replace real dependencies."""
    fake_modules = {mod: MagicMock(name=f"fake_{mod}") for mod in MOCK_MODULES}

    # Fake protocol module with required classes for OpenAI endpoints
    protocol_mod = types.ModuleType("vllm.entrypoints.openai.protocol")

    class FakeErrorResponse:
        pass

    class FakeChatCompletionResponse:
        pass

    class FakeCompletionResponse:
        pass

    class FakeChatRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            if not hasattr(self, "stream"):
                self.stream = False

    protocol_mod.ErrorResponse = FakeErrorResponse
    protocol_mod.ChatCompletionResponse = FakeChatCompletionResponse
    protocol_mod.CompletionResponse = FakeCompletionResponse
    protocol_mod.ChatCompletionRequest = FakeChatRequest
    protocol_mod.CompletionRequest = FakeChatRequest

    fake_modules["vllm.entrypoints.openai.protocol"] = protocol_mod

    # Fake vLLM v1 executor base class
    abstract_mod = types.ModuleType("vllm.v1.executor.abstract")

    class MockExecutor:
        def __init__(self, *args, **kwargs):
            pass

        @property
        def uses_ray(self):
            return False

    abstract_mod.Executor = MockExecutor
    fake_modules["vllm.v1.executor.abstract"] = abstract_mod

    # Fake AsyncServerBase for server address retrieval
    async_server_mod = types.ModuleType("agentic_rl.runner.infer_adapter.async_server")

    class MockAsyncServerBase:
        def __init__(self, *args, **kwargs):
            pass

        async def get_server_address(self):
            return "127.0.0.1:8000"

    async_server_mod.AsyncServerBase = MockAsyncServerBase
    fake_modules["agentic_rl.runner.infer_adapter.async_server"] = async_server_mod

    return fake_modules


def _reload_async_server_module():
    """Reload the module under test after fakes are installed."""
    mod_name = "agentic_rl.runner.infer_adapter.vllm.vllm_async_server"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


# ============================================================
# Dummy request and response helpers for testing
# ============================================================

class DummyRequest:
    """Fake HTTP request with async json() method."""
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class DummyErrorObj:
    """Simple error object with code attribute."""
    def __init__(self, code=400):
        self.code = code


# ------------------------------------------------------------------
# Response factories that inherit from fake protocol base classes
# ------------------------------------------------------------------

class DummyErrorResponse:
    """Factory for error responses that mimic ErrorResponse."""
    def __init__(self, base_cls, code=400):
        class _Dummy(base_cls):
            def __init__(self, code=400):
                self.error = DummyErrorObj(code=code)

            def model_dump(self):
                return {"error": {"code": self.error.code}}

            def model_dump_json(self, **kwargs):
                return json.dumps(self.model_dump())

        self.cls = _Dummy
        self.code = code

    def __call__(self, code=None):
        return self.cls(code=code if code is not None else self.code)


class DummyChatResponse:
    """Factory for chat completion responses."""
    def __init__(self, base_cls):
        class _Dummy(base_cls):
            def model_dump(self):
                return {"id": "chat_xxx"}

            def model_dump_json(self, **kwargs):
                return json.dumps(self.model_dump())

        self.cls = _Dummy

    def __call__(self):
        return self.cls()


class DummyLogprobs:
    """Dummy logprobs structure."""
    def __init__(self):
        self.token_logprobs = [1, 2]
        self.top_logprobs = [3, 4]
        self.text_offset = [0, 1]


class DummyChoice:
    """Dummy choice for completion responses."""
    def __init__(self, with_logprobs=True):
        self.logprobs = DummyLogprobs() if with_logprobs else None


class DummyCompletionResponse:
    """Factory for completion responses with optional logprobs."""
    def __init__(self, base_cls, with_logprobs=True):
        class _Dummy(base_cls):
            def __init__(self, with_logprobs=True):
                self.choices = [DummyChoice(with_logprobs=with_logprobs)]

            def model_dump(self):
                return {"id": "comp_xxx"}

            def model_dump_json(self, **kwargs):
                return json.dumps(self.model_dump())

        self.cls = _Dummy
        self.with_logprobs = with_logprobs

    def __call__(self, with_logprobs=None):
        return self.cls(with_logprobs=self.with_logprobs if with_logprobs is None else with_logprobs)


class DummyAsyncChunkGen:
    """Async generator that yields chunks for streaming responses."""
    def __init__(self, chunks):
        self._chunks = chunks
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        val = self._chunks[self._idx]
        self._idx += 1
        return val


# ============================================================
# Base test class with module isolation
# ============================================================

class BaseVllmAsyncServerTest(unittest.TestCase):
    """Base class that sets up fake modules and reloads the target module."""

    def setUp(self):
        self.module_patcher = patch.dict(sys.modules, _build_fake_modules())
        self.module_patcher.start()

        self.async_mod = _reload_async_server_module()

        # Export protocol base classes for isinstance checks in tests
        self.FakeErrorResponse = self.async_mod.ErrorResponse
        self.FakeChatCompletionResponse = self.async_mod.ChatCompletionResponse
        self.FakeCompletionResponse = self.async_mod.CompletionResponse

    def tearDown(self):
        self.module_patcher.stop()


# ============================================================
# Tests for safe_ray_init
# ============================================================

class TestSafeRayInit(BaseVllmAsyncServerTest):
    """Test the safe_ray_init helper function."""

    def test_safe_ray_init_connect_success(self):
        """When Ray connects successfully, it should log info."""
        safe_ray_init = self.async_mod.safe_ray_init

        with patch.object(self.async_mod, "ray") as mock_ray, \
             patch.object(self.async_mod, "logger") as mock_logger:

            safe_ray_init(namespace="ns1")
            mock_ray.init.assert_called_once_with(
                address="auto", namespace="ns1", ignore_reinit_error=True
            )
            mock_logger.info.assert_called_once()

    def test_safe_ray_init_fallback_to_local(self):
        """If auto-connect fails, fall back to local Ray init."""
        safe_ray_init = self.async_mod.safe_ray_init

        with patch.object(self.async_mod, "ray") as mock_ray, \
             patch.object(self.async_mod, "logger") as mock_logger:

            mock_ray.init.side_effect = [RuntimeError("connect failed"), None]

            safe_ray_init(namespace="ns2")

            self.assertEqual(mock_ray.init.call_count, 2)
            mock_ray.init.assert_any_call(
                address="auto", namespace="ns2", ignore_reinit_error=True
            )
            mock_ray.init.assert_any_call(namespace="ns2", ignore_reinit_error=True)

            mock_logger.warning.assert_called_once()
            self.assertGreaterEqual(mock_logger.info.call_count, 1)

    def test_safe_ray_init_default_namespace(self):
        """When no namespace provided, default to 'default'."""
        safe_ray_init = self.async_mod.safe_ray_init

        with patch.object(self.async_mod, "ray") as mock_ray, \
             patch.object(self.async_mod, "logger") as mock_logger:

            safe_ray_init()
            mock_ray.init.assert_called_once_with(
                address="auto", namespace="default", ignore_reinit_error=True
            )


# ============================================================
# Tests for ExternalRayDistributedExecutor
# ============================================================

class TestExternalRayDistributedExecutor(BaseVllmAsyncServerTest):
    """Test the custom Ray-based distributed executor."""

    def _make_config(self, instance_id="ns:wg:2:0", tp_size=2, dp_rank=0):
        """Create a mock vLLM config with given parameters."""
        cfg = MagicMock()
        cfg.instance_id = instance_id
        cfg.parallel_config.tensor_parallel_size = tp_size
        cfg.parallel_config.data_parallel_rank = dp_rank
        return cfg

    def _make_executor(self, cfg):
        """Create an uninitialized executor instance."""
        ExternalRayDistributedExecutor = self.async_mod.ExternalRayDistributedExecutor
        executor = ExternalRayDistributedExecutor.__new__(ExternalRayDistributedExecutor)
        executor.vllm_config = cfg
        executor.workers = []
        executor.actor_names = []
        executor.is_first_wake_up = False
        executor.is_sleeping = False
        return executor

    def test_init_executor_main_flow(self):
        """Normal initialization: find workers, call collective RPC methods."""
        ExternalRayDistributedExecutor = self.async_mod.ExternalRayDistributedExecutor

        with patch.object(self.async_mod, "ray") as mock_ray, \
             patch.object(self.async_mod, "logger") as mock_logger, \
             patch.object(self.async_mod, "safe_ray_init") as mock_safe_init:

            # Simulate existing Ray actors
            mock_ray.state.actors.return_value = {
                "a0": {"Name": "ActorHybridWorker_0"},
                "a1": {"Name": "ActorHybridWorker_1"},
                "a2": {"Name": "IntegratedWorker_2"},
                "a3": {"Name": "OtherWorker_9"},
            }

            w0, w1 = MagicMock(), MagicMock()
            mock_ray.get_actor.side_effect = [w0, w1]

            cfg = self._make_config(instance_id="ns:wg:2:0", tp_size=2, dp_rank=0)
            executor = self._make_executor(cfg)
            executor.collective_rpc = MagicMock(return_value=[None])

            with patch.dict(os.environ, {"VLLM_DP_SIZE": "1"}):
                ExternalRayDistributedExecutor._init_executor(executor)

            mock_safe_init.assert_called_once_with("ns")
            self.assertEqual(executor.actor_names, ["ActorHybridWorker_0", "ActorHybridWorker_1"])
            self.assertEqual(executor.workers, [w0, w1])

            # Three collective calls: init_worker, init_device, load_model
            self.assertEqual(executor.collective_rpc.call_count, 3)
            self.assertEqual(executor.collective_rpc.mock_calls[0].args[0], "init_worker")
            self.assertEqual(executor.collective_rpc.mock_calls[1].args[0], "init_device")
            self.assertEqual(executor.collective_rpc.mock_calls[2].args[0], "load_model")

            mock_logger.info.assert_called()

    def test_init_executor_invalid_instance_id_format(self):
        """instance_id must follow format 'namespace:wg_prefix:tp_size:dp_rank'."""
        ExternalRayDistributedExecutor = self.async_mod.ExternalRayDistributedExecutor
        cfg = self._make_config(instance_id="invalid_format")
        executor = self._make_executor(cfg)

        with self.assertRaises(ValueError) as ctx:
            ExternalRayDistributedExecutor._init_executor(executor)

        self.assertIn("must be in the format of", str(ctx.exception))

    def test_init_executor_missing_instance_id(self):
        """instance_id must not be None."""
        ExternalRayDistributedExecutor = self.async_mod.ExternalRayDistributedExecutor
        cfg = self._make_config(instance_id=None)
        executor = self._make_executor(cfg)

        with self.assertRaises(ValueError) as ctx:
            ExternalRayDistributedExecutor._init_executor(executor)

        self.assertIn("instance_id must be set", str(ctx.exception))

    def test_init_executor_dp_rank_expand_with_env_dp_size(self):
        """When VLLM_DP_SIZE > 1, select workers based on dp_rank."""
        ExternalRayDistributedExecutor = self.async_mod.ExternalRayDistributedExecutor

        with patch.object(self.async_mod, "ray") as mock_ray, \
             patch.object(self.async_mod, "safe_ray_init"):

            # 8 workers total
            mock_ray.state.actors.return_value = {
                f"a{i}": {"Name": f"ActorHybridWorker_{i}"}
                for i in range(8)
            }

            w4, w5 = MagicMock(), MagicMock()
            mock_ray.get_actor.side_effect = [w4, w5]

            cfg = self._make_config(instance_id="ns:wg:2:1", tp_size=2, dp_rank=0)
            executor = self._make_executor(cfg)
            executor.collective_rpc = MagicMock(return_value=[None])

            with patch.dict(os.environ, {"VLLM_DP_SIZE": "2"}):
                ExternalRayDistributedExecutor._init_executor(executor)

            self.assertEqual(executor.actor_names, ["ActorHybridWorker_4", "ActorHybridWorker_5"])
            self.assertEqual(executor.workers, [w4, w5])

    def test_collective_rpc_string_method(self):
        """collective_rpc with string method name calls remote on each worker."""
        ExternalRayDistributedExecutor = self.async_mod.ExternalRayDistributedExecutor

        with patch.object(self.async_mod, "ray") as mock_ray:
            executor = ExternalRayDistributedExecutor.__new__(ExternalRayDistributedExecutor)

            w1, w2 = MagicMock(), MagicMock()
            fut1, fut2 = MagicMock(), MagicMock()
            w1.execute_method.remote.return_value = fut1
            w2.execute_method.remote.return_value = fut2

            executor.workers = [w1, w2]
            mock_ray.get.return_value = ["r1", "r2"]

            out = ExternalRayDistributedExecutor.collective_rpc(
                executor, method="init_device", args=("AAA",), kwargs={"k": 1}
            )

            w1.execute_method.remote.assert_called_once_with("init_device", "AAA", k=1)
            w2.execute_method.remote.assert_called_once_with("init_device", "AAA", k=1)

            mock_ray.get.assert_called_once_with([fut1, fut2])
            self.assertEqual(out, ["r1", "r2"])

    def test_collective_rpc_callable_method_cloudpickle(self):
        """Callable method is pickled and sent as bytes."""
        ExternalRayDistributedExecutor = self.async_mod.ExternalRayDistributedExecutor

        with patch.object(self.async_mod, "ray") as mock_ray:
            executor = ExternalRayDistributedExecutor.__new__(ExternalRayDistributedExecutor)

            w1 = MagicMock()
            fut = MagicMock()
            w1.execute_method.remote.return_value = fut
            executor.workers = [w1]

            mock_ray.get.return_value = ["ok"]

            def my_fn(x):
                return x + 1

            out = ExternalRayDistributedExecutor.collective_rpc(
                executor, method=my_fn, args=(3,), kwargs=None
            )

            called_args = w1.execute_method.remote.call_args[0]
            sent_method = called_args[0]
            self.assertIsInstance(sent_method, (bytes, bytearray))
            self.assertEqual(sent_method, cloudpickle.dumps(my_fn))

            mock_ray.get.assert_called_once_with([fut])
            self.assertEqual(out, ["ok"])

    def test_check_health_noop(self):
        """check_health does nothing."""
        ExternalRayDistributedExecutor = self.async_mod.ExternalRayDistributedExecutor
        executor = ExternalRayDistributedExecutor.__new__(ExternalRayDistributedExecutor)
        ret = ExternalRayDistributedExecutor.check_health(executor)
        self.assertIsNone(ret)

    def test_uses_ray_flag(self):
        """uses_ray property is False for this executor."""
        ExternalRayDistributedExecutor = self.async_mod.ExternalRayDistributedExecutor
        self.assertFalse(ExternalRayDistributedExecutor.uses_ray)


# ============================================================
# Tests for AsyncVLLMServer
# ============================================================

class TestAsyncVLLMServer(unittest.IsolatedAsyncioTestCase):
    """Test the main async server class."""

    def setUp(self):
        self.module_patcher = patch.dict(sys.modules, _build_fake_modules())
        self.module_patcher.start()
        self.async_mod = _reload_async_server_module()

    def tearDown(self):
        self.module_patcher.stop()

    def _make_config(self):
        """Create a mock configuration with typical values."""
        cfg = MagicMock()
        cfg.trust_remote_code = True
        cfg.infer_tensor_parallel_size = 2
        cfg.infer_pipeline_parallel_size = 1
        cfg.max_model_len = 2048
        cfg.max_num_batched_tokens = 1024
        cfg.enable_sleep_mode = False
        cfg.dtype = "float16"
        cfg.enforce_eager = False
        cfg.gpu_memory_utilization = 0.8
        cfg.enable_chunked_prefill = True
        cfg.enable_prefix_caching = False
        cfg.enable_expert_parallel = False
        cfg.disable_log_stats = False
        cfg.max_num_seqs = 8
        cfg.load_format = "megatron"
        cfg.cudagraph_capture_sizes = "1,2,4"

        sampling = MagicMock()
        sampling.logprobs = 0
        sampling.max_tokens = 16
        sampling.top_p = 1.0
        sampling.top_k = -1
        sampling.min_p = 0.0
        sampling.temperature = 1.0
        cfg.sampling_config = sampling

        return cfg

    async def test_init_constructor_sets_fields(self):
        """Constructor should store all provided attributes."""
        AsyncVLLMServer = self.async_mod.AsyncVLLMServer

        cfg = self._make_config()
        server = AsyncVLLMServer(cfg, "/models/xx", 2, 1, "wg")

        self.assertEqual(server.config, cfg)
        self.assertEqual(server.tokenizer_name_or_path, "/models/xx")
        self.assertEqual(server.vllm_dp_size, 2)
        self.assertEqual(server.vllm_dp_rank, 1)
        self.assertEqual(server.wg_prefix, "wg")
        self.assertIsNone(server.engine)
        self.assertIsNone(server.ins_workload)

    async def test_init_engine_main_flow_disable_log_stats_false(self):
        """Normal engine initialization (disable_log_stats=False) creates background task."""
        AsyncVLLMServer = self.async_mod.AsyncVLLMServer

        cfg = self._make_config()
        server = AsyncVLLMServer(cfg, "/models/test_model", 2, 1, "wg")
        server.get_server_address = AsyncMock(return_value="127.0.0.1:8000")

        fake_engine_args = MagicMock()
        fake_engine_args.create_engine_config.return_value = MagicMock()
        fake_engine = MagicMock()
        fake_engine.model_config = MagicMock()

        fake_async_llm_cls = MagicMock()
        fake_async_llm_cls.from_vllm_config.return_value = fake_engine

        with patch.dict(os.environ, {"VLLM_DP_SIZE": "2"}), \
             patch.object(self.async_mod, "AsyncEngineArgs", return_value=fake_engine_args) as mock_engine_args_cls, \
             patch.object(self.async_mod, "AsyncLLM", fake_async_llm_cls), \
             patch.object(self.async_mod, "OpenAIServingModels", return_value=MagicMock()), \
             patch.object(self.async_mod, "OpenAIServingChat", return_value=MagicMock()), \
             patch.object(self.async_mod, "OpenAIServingCompletion", return_value=MagicMock()), \
             patch.object(self.async_mod.ray, "get_runtime_context") as mock_ctx, \
             patch.object(self.async_mod.asyncio, "create_task") as mock_create_task:

            ctx = MagicMock()
            ctx.namespace = "ns"
            mock_ctx.return_value = ctx

            await server.init_engine()

            mock_engine_args_cls.assert_called_once()
            fake_engine_args.create_engine_config.assert_called_once()
            fake_async_llm_cls.from_vllm_config.assert_called_once()
            mock_create_task.assert_called_once()

            self.assertIsNotNone(server.engine)
            self.assertIsNotNone(server.openai_serving_chat)
            self.assertIsNotNone(server.openai_serving_completion)
            self.assertIsNotNone(server.ins_workload)

    async def test_init_engine_disable_log_stats_true_no_create_task(self):
        """When disable_log_stats=True, no background stat logger task is created."""
        AsyncVLLMServer = self.async_mod.AsyncVLLMServer

        cfg = self._make_config()
        cfg.disable_log_stats = True

        server = AsyncVLLMServer(cfg, "/models/test_model", 2, 1, "wg")
        server.get_server_address = AsyncMock(return_value="127.0.0.1:8000")

        fake_engine_args = MagicMock()
        fake_engine_args.create_engine_config.return_value = MagicMock()
        fake_engine = MagicMock()
        fake_engine.model_config = MagicMock()

        fake_async_llm_cls = MagicMock()
        fake_async_llm_cls.from_vllm_config.return_value = fake_engine

        with patch.dict(os.environ, {"VLLM_DP_SIZE": "1"}), \
             patch.object(self.async_mod, "AsyncEngineArgs", return_value=fake_engine_args), \
             patch.object(self.async_mod, "AsyncLLM", fake_async_llm_cls), \
             patch.object(self.async_mod, "OpenAIServingModels", return_value=MagicMock()), \
             patch.object(self.async_mod, "OpenAIServingChat", return_value=MagicMock()), \
             patch.object(self.async_mod, "OpenAIServingCompletion", return_value=MagicMock()), \
             patch.object(self.async_mod.ray, "get_runtime_context") as mock_ctx, \
             patch.object(self.async_mod.asyncio, "create_task") as mock_create_task:

            ctx = MagicMock()
            ctx.namespace = "ns"
            mock_ctx.return_value = ctx

            await server.init_engine()
            mock_create_task.assert_not_called()

    async def test_chat_completion_error_response(self):
        """When chat completion returns an ErrorResponse, return error with status code."""
        AsyncVLLMServer = self.async_mod.AsyncVLLMServer

        cfg = self._make_config()
        server = AsyncVLLMServer(cfg, "/models/x", 1, 0, "wg")

        dummy_error_cls = DummyErrorResponse(self.async_mod.ErrorResponse)
        DummyErr = dummy_error_cls.cls

        server.openai_serving_chat = MagicMock()
        server.openai_serving_chat.create_chat_completion = AsyncMock(return_value=DummyErr(code=401))

        raw_request = DummyRequest({"model": "xxx", "messages": [], "stream": False})

        with patch.object(self.async_mod, "ChatCompletionRequest") as mock_req:
            mock_req.side_effect = lambda **kwargs: MagicMock(stream=False)

            with patch.object(self.async_mod, "ErrorResponse", DummyErr):
                resp = await server.chat_completion(raw_request)

        self.assertEqual(resp.status_code, 401)

    async def test_chat_completion_streaming_response(self):
        """Streaming chat completion returns a streaming response."""
        AsyncVLLMServer = self.async_mod.AsyncVLLMServer

        cfg = self._make_config()
        server = AsyncVLLMServer(cfg, "/models/x", 1, 0, "wg")

        dummy_chat_cls = DummyChatResponse(self.async_mod.ChatCompletionResponse)
        DummyChat = dummy_chat_cls.cls

        server.openai_serving_chat = MagicMock()
        server.openai_serving_chat.create_chat_completion = AsyncMock(
            return_value=DummyAsyncChunkGen(["c1", "c2"])
        )

        raw_request = DummyRequest({"model": "xxx", "messages": [], "stream": True})

        with patch.object(self.async_mod, "ChatCompletionRequest") as mock_req:
            mock_req.side_effect = lambda **kwargs: MagicMock(stream=True)

            with patch.object(self.async_mod, "ChatCompletionResponse", DummyChat):
                resp = await server.chat_completion(raw_request)

        self.assertTrue(hasattr(resp, "body_iterator"))

    async def test_chat_completion_non_stream_json_response(self):
        """Non-streaming chat completion returns JSON response."""
        AsyncVLLMServer = self.async_mod.AsyncVLLMServer

        cfg = self._make_config()
        server = AsyncVLLMServer(cfg, "/models/x", 1, 0, "wg")

        dummy_chat_cls = DummyChatResponse(self.async_mod.ChatCompletionResponse)
        DummyChat = dummy_chat_cls.cls

        server.openai_serving_chat = MagicMock()
        server.openai_serving_chat.create_chat_completion = AsyncMock(return_value=DummyChat())

        raw_request = DummyRequest({"model": "xxx", "messages": [], "stream": False})

        with patch.object(self.async_mod, "ChatCompletionRequest") as mock_req:
            mock_req.side_effect = lambda **kwargs: MagicMock(stream=False)

            with patch.object(self.async_mod, "ChatCompletionResponse", DummyChat):
                resp = await server.chat_completion(raw_request)

        self.assertEqual(resp.status_code, 200)

    async def test_completions_error_response(self):
        """Error response from completions endpoint returns correct status code."""
        AsyncVLLMServer = self.async_mod.AsyncVLLMServer

        cfg = self._make_config()
        server = AsyncVLLMServer(cfg, "/models/x", 1, 0, "wg")

        dummy_error_cls = DummyErrorResponse(self.async_mod.ErrorResponse)
        DummyErr = dummy_error_cls.cls

        server.openai_serving_completion = MagicMock()
        server.openai_serving_completion.create_completion = AsyncMock(return_value=DummyErr(code=500))

        raw_request = DummyRequest({"model": "xxx", "prompt": "hi", "stream": False})

        with patch.object(self.async_mod, "CompletionRequest") as mock_req:
            mock_req.side_effect = lambda **kwargs: MagicMock(stream=False)

            with patch.object(self.async_mod, "ErrorResponse", DummyErr):
                resp = await server.completions(raw_request)

        self.assertEqual(resp.status_code, 500)

    async def test_completions_streaming_response(self):
        """Streaming completions returns streaming response."""
        AsyncVLLMServer = self.async_mod.AsyncVLLMServer

        cfg = self._make_config()
        server = AsyncVLLMServer(cfg, "/models/x", 1, 0, "wg")

        dummy_comp_cls = DummyCompletionResponse(self.async_mod.CompletionResponse, with_logprobs=True)
        DummyComp = dummy_comp_cls.cls

        server.openai_serving_completion = MagicMock()
        server.openai_serving_completion.create_completion = AsyncMock(
            return_value=DummyAsyncChunkGen(["x1", "x2"])
        )

        raw_request = DummyRequest({"model": "xxx", "prompt": "hi", "stream": True})

        with patch.object(self.async_mod, "CompletionRequest") as mock_req:
            mock_req.side_effect = lambda **kwargs: MagicMock(stream=True)

            with patch.object(self.async_mod, "CompletionResponse", DummyComp):
                resp = await server.completions(raw_request)

        self.assertTrue(hasattr(resp, "body_iterator"))

    async def test_completions_non_stream_cleanup_logprobs(self):
        """Non-streaming completions: logprobs fields are cleared for JSON serialization."""
        AsyncVLLMServer = self.async_mod.AsyncVLLMServer

        cfg = self._make_config()
        server = AsyncVLLMServer(cfg, "/models/x", 1, 0, "wg")

        dummy_comp_cls = DummyCompletionResponse(self.async_mod.CompletionResponse, with_logprobs=True)
        DummyComp = dummy_comp_cls.cls
        dummy_resp = DummyComp(with_logprobs=True)

        server.openai_serving_completion = MagicMock()
        server.openai_serving_completion.create_completion = AsyncMock(return_value=dummy_resp)

        raw_request = DummyRequest({"model": "xxx", "prompt": "hi", "stream": False})

        with patch.object(self.async_mod, "CompletionRequest") as mock_req:
            mock_req.side_effect = lambda **kwargs: MagicMock(stream=False)

            with patch.object(self.async_mod, "CompletionResponse", DummyComp):
                resp = await server.completions(raw_request)

        self.assertEqual(dummy_resp.choices[0].logprobs.token_logprobs, [])
        self.assertEqual(dummy_resp.choices[0].logprobs.top_logprobs, [])
        self.assertEqual(dummy_resp.choices[0].logprobs.text_offset, [])
        self.assertEqual(resp.status_code, 200)

    async def test_wake_up_calls_engine(self):
        """wake_up delegates to engine.wake_up."""
        AsyncVLLMServer = self.async_mod.AsyncVLLMServer

        cfg = self._make_config()
        server = AsyncVLLMServer(cfg, "/models/x", 1, 0, "wg")

        server.engine = MagicMock()
        server.engine.wake_up = AsyncMock(return_value=None)

        await server.wake_up(tags=["a"])
        server.engine.wake_up.assert_awaited_once_with(["a"])

    async def test_sleep_calls_reset_prefix_cache_then_sleep(self):
        """sleep resets prefix cache before calling engine.sleep."""
        AsyncVLLMServer = self.async_mod.AsyncVLLMServer

        cfg = self._make_config()
        server = AsyncVLLMServer(cfg, "/models/x", 1, 0, "wg")

        server.engine = MagicMock()
        server.engine.reset_prefix_cache = AsyncMock(return_value=None)
        server.engine.sleep = AsyncMock(return_value=None)

        await server.sleep()

        server.engine.reset_prefix_cache.assert_awaited_once()
        server.engine.sleep.assert_awaited_once()

    async def test_collective_rpc_delegates_to_engine(self):
        """collective_rpc forwards to engine.collective_rpc."""
        AsyncVLLMServer = self.async_mod.AsyncVLLMServer

        cfg = self._make_config()
        server = AsyncVLLMServer(cfg, "/models/x", 1, 0, "wg")

        server.engine = MagicMock()
        server.engine.collective_rpc = AsyncMock(return_value=["ok"])

        out = await server.collective_rpc("m", timeout=1.0, args=("a",), kwargs={"k": 1})
        self.assertEqual(out, ["ok"])
        server.engine.collective_rpc.assert_awaited_once()

    async def test_get_workload_returns_json_response(self):
        """get_workload returns workload stats as JSON."""
        AsyncVLLMServer = self.async_mod.AsyncVLLMServer

        cfg = self._make_config()
        server = AsyncVLLMServer(cfg, "/models/x", 1, 0, "wg")

        server.ins_workload = MagicMock()
        server.ins_workload.to_dict.return_value = {"dp_size": 2}

        resp = await server.get_workload(DummyRequest({}))
        self.assertEqual(resp.status_code, 200)

    async def test_cancel_requests_calls_engine_abort(self):
        """cancel_requests calls engine.abort with the request IDs."""
        AsyncVLLMServer = self.async_mod.AsyncVLLMServer

        cfg = self._make_config()
        server = AsyncVLLMServer(cfg, "/models/x", 1, 0, "wg")

        server.engine = MagicMock()
        server.engine.abort = AsyncMock(return_value=None)

        raw_request = DummyRequest({"requests": ["r1", "r2"]})

        with patch.object(self.async_mod, "logger") as mock_logger:
            await server.cancel_requests(raw_request)

        server.engine.abort.assert_awaited_once_with(["r1", "r2"])
        mock_logger.info.assert_called_once()

    async def test_reset_prefix_cache_calls_engine(self):
        """reset_prefix_cache delegates to engine.reset_prefix_cache."""
        AsyncVLLMServer = self.async_mod.AsyncVLLMServer

        cfg = self._make_config()
        server = AsyncVLLMServer(cfg, "/models/x", 1, 0, "wg")

        server.engine = MagicMock()
        server.engine.reset_prefix_cache = AsyncMock(return_value=None)

        await server.reset_prefix_cache()
        server.engine.reset_prefix_cache.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()