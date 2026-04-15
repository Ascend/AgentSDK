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

"""
Unit tests for vllm_async_server_pd module (prefill-decode disaggregated server).

This test suite covers:
- Automatic inference mode detection (prefill vs decode) via rank table.
- Dynamic config updates for prefill/decode modes.
- Request modifications for prefilling.
- Server endpoints (chat/completions) with streaming and error handling.
- Ray-based initialization and custom executor integration.
"""

import os
import sys
import json
import unittest
import importlib
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from starlette.responses import JSONResponse, StreamingResponse


# ============================================================
# Helper: build fake external modules
# ============================================================

def _build_fake_modules():
    """
    Build a fake sys.modules dict for heavy dependencies:
    - ray, vllm, uvicorn, fastapi, transformers, torch
    - agentic_rl base loggers and scheduler modules
    - async_server base class
    """
    fake_modules = {}

    def _mk(name):
        return MagicMock(name=name)

    # ---------------- ray ----------------
    ray_mod = _mk("ray")
    ray_state_mod = _mk("ray.state")
    ray_runtime_ctx_mod = _mk("ray.runtime_context")
    ray_util_mod = _mk("ray.util")

    ray_mod.state = ray_state_mod
    ray_mod.runtime_context = ray_runtime_ctx_mod
    ray_mod.util = ray_util_mod

    ray_util_mod.get_node_ip_address = MagicMock(return_value="127.0.0.1")
    ray_mod.get_runtime_context = MagicMock()

    fake_modules["ray"] = ray_mod
    fake_modules["ray.state"] = ray_state_mod
    fake_modules["ray.runtime_context"] = ray_runtime_ctx_mod
    fake_modules["ray.util"] = ray_util_mod

    # ---------------- vllm ----------------
    vllm_mod = _mk("vllm")
    fake_modules["vllm"] = vllm_mod

    fake_modules["vllm.engine.arg_utils"] = _mk("vllm.engine.arg_utils")
    fake_modules["vllm.entrypoints.openai.protocol"] = _mk("vllm.entrypoints.openai.protocol")
    fake_modules["vllm.entrypoints.openai.serving_chat"] = _mk("vllm.entrypoints.openai.serving_chat")
    fake_modules["vllm.entrypoints.openai.serving_completion"] = _mk("vllm.entrypoints.openai.serving_completion")
    fake_modules["vllm.entrypoints.openai.serving_models"] = _mk("vllm.entrypoints.openai.serving_models")
    fake_modules["vllm.v1.engine.async_llm"] = _mk("vllm.v1.engine.async_llm")
    fake_modules["vllm.v1.executor.abstract"] = _mk("vllm.v1.executor.abstract")
    fake_modules["vllm.config"] = _mk("vllm.config")
    fake_modules["vllm.executor"] = _mk("vllm.executor")

    # Provide Executor base for isinstance checks
    class MockExecutor:
        def __init__(self, *args, **kwargs):
            pass

        @property
        def uses_ray(self):
            return False

    fake_modules["vllm.v1.executor.abstract"].Executor = MockExecutor

    # Provide protocol base classes for isinstance checks
    protocol_mod = fake_modules["vllm.entrypoints.openai.protocol"]

    class FakeErrorResponse:
        pass

    class FakeChatCompletionResponse:
        pass

    class FakeCompletionResponse:
        pass

    protocol_mod.ErrorResponse = FakeErrorResponse
    protocol_mod.ChatCompletionResponse = FakeChatCompletionResponse
    protocol_mod.CompletionResponse = FakeCompletionResponse

    class FakeChatRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            if not hasattr(self, "stream"):
                self.stream = False

    protocol_mod.ChatCompletionRequest = FakeChatRequest
    protocol_mod.CompletionRequest = FakeChatRequest

    # ---------------- other heavy deps ----------------
    for name in [
        "uvicorn", "fastapi", "omegaconf", "torch", "transformers",
        "agentic_rl.base.log.loggers",
        "agentic_rl.runner.scheduler.workload",
        "agentic_rl.runner.scheduler.load_stat",
    ]:
        fake_modules[name] = _mk(name)

    # ---------------- async_server base ----------------
    try:
        async_server_mod = _mk("agentic_rl.runner.infer_adapter.async_server")
    except Exception:
        raise

    class MockAsyncServerBase:
        def __init__(self, *args, **kwargs):
            pass

        async def get_server_address(self):
            return "127.0.0.1:8000"

    async_server_mod.AsyncServerBase = MockAsyncServerBase
    fake_modules["agentic_rl.runner.infer_adapter.async_server"] = async_server_mod

    return fake_modules


def _reload_target_module():
    """
    Reload the target module after fakes are installed to ensure all imports use mocks.
    """
    mod_name = "agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


def bind_dummy_response_base_classes(target_mod):
    """
    Inject dummy response classes that inherit from the real base classes imported by the module.
    This ensures isinstance checks in the production code pass during tests.
    """
    global DummyErrorResponse, DummyChatCompletionResponse, DummyCompletionResponse

    class DummyErrorResponse(target_mod.ErrorResponse):
        def __init__(self, code=400):
            self.error = MagicMock()
            self.error.code = code

        def model_dump(self):
            return {"error": {"code": self.error.code, "message": "dummy error"}}

    class DummyChatCompletionResponse(target_mod.ChatCompletionResponse):
        def model_dump(self):
            return {"id": "dummy_chat_response"}

    class DummyCompletionResponse(target_mod.CompletionResponse):
        def __init__(self, with_logprobs=False):
            self.choices = []
            if with_logprobs:
                lp = MagicMock()
                lp.token_logprobs = [1]
                lp.top_logprobs = [2]
                lp.text_offset = [3]
                choice = MagicMock()
                choice.logprobs = lp
                self.choices.append(choice)

        def model_dump(self):
            return {"id": "dummy_completion_response"}


# ============================================================
# Dummy Request and Async Generator
# ============================================================

class DummyRequest:
    """Minimal Starlette-like request mock."""
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class DummyAsyncChunkGen:
    """Async generator that yields chunks for streaming response simulation."""
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
# Test Cases for AsyncVLLMServerPDSep
# ============================================================

class TestAsyncVLLMServerPDSep(unittest.IsolatedAsyncioTestCase):
    """Main test suite for the disaggregated prefill-decode server."""

    def setUp(self):
        # Patch sys.modules with fake dependencies
        self.module_patcher = patch.dict(sys.modules, _build_fake_modules())
        self.module_patcher.start()

        self.target_mod = _reload_target_module()
        bind_dummy_response_base_classes(self.target_mod)

        # Create a base mock config with all necessary attributes
        self.config = MagicMock()

        # Remove kv_transfer_config if present to simulate uninitialized state
        if hasattr(self.config, "kv_transfer_config"):
            delattr(self.config, "kv_transfer_config")

        self.config.trust_remote_code = True
        self.config.infer_tensor_parallel_size = 1
        self.config.infer_pipeline_parallel_size = 1
        self.config.max_model_len = 2048
        self.config.max_num_batched_tokens = 1024
        self.config.enable_sleep_mode = False
        self.config.dtype = "float16"
        self.config.enforce_eager = False
        self.config.gpu_memory_utilization = 0.8
        self.config.enable_chunked_prefill = False
        self.config.enable_prefix_caching = False
        self.config.enable_expert_parallel = False
        self.config.disable_log_stats = True
        self.config.max_num_seqs = 8
        self.config.load_format = "megatron"
        self.config.cudagraph_capture_sizes = None

        sampling = MagicMock()
        sampling.logprobs = 0
        sampling.max_tokens = 16
        sampling.top_p = 1.0
        sampling.top_k = -1
        sampling.min_p = 0.0
        sampling.temperature = 1.0
        self.config.sampling_config = sampling

        # Prefill-specific overrides
        self.config.prefill_enforce_eager = True
        self.config.prefill_max_num_seqs = 2
        self.config.prefill_max_num_batched_tokens = 128
        self.config.prefill_gpu_memory_utilization = 0.5
        self.config.prefill_max_model_len = 1024

        self.tokenizer_path = "/models/test_model"
        self.dp_size = 2
        self.dp_rank = 0
        self.wg_prefix = "wg"

    def tearDown(self):
        self.module_patcher.stop()

    def _create_server_skip_update_config(self, infer_mode=None):
        """
        Create a server instance while bypassing __init__ and update_config,
        then manually set required attributes.
        """
        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep

        with patch(
            "agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.AsyncVLLMServer.__init__",
            return_value=None,
        ), patch(
            "agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.AsyncVLLMServerPDSep.update_config",
            return_value=None,
        ):
            server = AsyncVLLMServerPDSep(
                config=self.config,
                tokenizer_name_or_path=self.tokenizer_path,
                vllm_dp_size=self.dp_size,
                vllm_dp_rank=self.dp_rank,
                wg_prefix=self.wg_prefix,
                infer_mode=infer_mode,
            )

        server.config = self.config
        server.tokenizer_name_or_path = self.tokenizer_path
        server.vllm_dp_size = self.dp_size
        server.vllm_dp_rank = self.dp_rank
        server.wg_prefix = self.wg_prefix
        server.server_ready = MagicMock()
        server.server_ready.wait = AsyncMock()

        if infer_mode is not None:
            server.infer_mode = infer_mode

        return server

    # ------------------------------------------------------------------
    # Tests for rank table parsing and auto mode detection
    # ------------------------------------------------------------------

    @patch.dict(os.environ, {"DISAGGREGATED_PREFILL_RANK_TABLE_PATH": "/tmp/ranktable.json"})
    def test_get_node_ip_from_ranktable(self):
        """Parsing rank table JSON should return sets of prefill/decode IPs."""
        server = self._create_server_skip_update_config(infer_mode="prefill")
        ranktable_data = {
            "prefill_device_list": [{"server_id": "10.0.0.1"}, {"server_id": "10.0.0.2"}],
            "decode_device_list": [{"server_id": "10.0.0.3"}, {"server_id": "10.0.0.4"}],
        }

        m = mock_open(read_data=json.dumps(ranktable_data))
        with patch("builtins.open", m):
            node_ips = server.get_node_ip_from_ranktable()

        self.assertEqual(node_ips["prefill"], {"10.0.0.1", "10.0.0.2"})
        self.assertEqual(node_ips["decode"], {"10.0.0.3", "10.0.0.4"})

    @patch.dict(os.environ, {"DISAGGREGATED_PREFILL_RANK_TABLE_PATH": "/tmp/ranktable.json"})
    def test_update_config_infer_mode_auto_prefill(self):
        """When local IP matches prefill list, infer_mode becomes 'prefill'."""
        server = self._create_server_skip_update_config(infer_mode=None)

        if hasattr(server.config, "kv_transfer_config"):
            delattr(server.config, "kv_transfer_config")

        ranktable_data = {
            "prefill_device_list": [{"server_id": "1.1.1.1"}],
            "decode_device_list": [{"server_id": "2.2.2.2"}],
        }
        m = mock_open(read_data=json.dumps(ranktable_data))

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep

        with patch("builtins.open", m), patch(
            "agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.ray.util.get_node_ip_address",
            return_value="1.1.1.1",
        ):
            AsyncVLLMServerPDSep.update_config(server, infer_mode=None)

        self.assertEqual(server.infer_mode, "prefill")
        self.assertEqual(server.config.kv_transfer_config["kv_role"], "kv_producer")
        self.assertEqual(server.config.max_model_len, 1024)
        self.assertEqual(server.config.max_num_seqs, 2)
        self.assertEqual(server.config.max_num_batched_tokens, 128)
        self.assertEqual(server.config.gpu_memory_utilization, 0.5)
        self.assertTrue(server.config.enforce_eager)

    @patch.dict(os.environ, {"DISAGGREGATED_PREFILL_RANK_TABLE_PATH": "/tmp/ranktable.json"})
    def test_update_config_infer_mode_auto_decode(self):
        """When local IP matches decode list, infer_mode becomes 'decode'."""
        server = self._create_server_skip_update_config(infer_mode=None)

        if hasattr(server.config, "kv_transfer_config"):
            delattr(server.config, "kv_transfer_config")

        ranktable_data = {
            "prefill_device_list": [{"server_id": "1.1.1.1"}],
            "decode_device_list": [{"server_id": "2.2.2.2"}],
        }
        m = mock_open(read_data=json.dumps(ranktable_data))

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep

        with patch("builtins.open", m), patch(
            "agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.ray.util.get_node_ip_address",
            return_value="2.2.2.2",
        ):
            AsyncVLLMServerPDSep.update_config(server, infer_mode=None)

        self.assertEqual(server.infer_mode, "decode")
        self.assertEqual(server.config.kv_transfer_config["kv_role"], "kv_consumer")

    @patch.dict(os.environ, {"DISAGGREGATED_PREFILL_RANK_TABLE_PATH": "/tmp/ranktable.json"})
    def test_update_config_node_ip_not_found_raises(self):
        """If local IP not found in either list, raise ValueError."""
        server = self._create_server_skip_update_config(infer_mode=None)

        if hasattr(server.config, "kv_transfer_config"):
            delattr(server.config, "kv_transfer_config")

        ranktable_data = {
            "prefill_device_list": [{"server_id": "1.1.1.1"}],
            "decode_device_list": [{"server_id": "2.2.2.2"}],
        }
        m = mock_open(read_data=json.dumps(ranktable_data))

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep

        with patch("builtins.open", m), patch(
            "agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.ray.util.get_node_ip_address",
            return_value="9.9.9.9",
        ):
            with self.assertRaises(ValueError):
                AsyncVLLMServerPDSep.update_config(server, infer_mode=None)

    def test_update_config_skip_if_kv_transfer_config_exists(self):
        """If kv_transfer_config already present, do not reinitialize."""
        server = self._create_server_skip_update_config(infer_mode=None)

        setattr(server.config, "kv_transfer_config", {"already": "exists"})

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep

        with patch.object(server, "get_node_ip_from_ranktable") as mock_get:
            AsyncVLLMServerPDSep.update_config(server, infer_mode=None)
            mock_get.assert_not_called()

        self.assertEqual(server.config.kv_transfer_config, {"already": "exists"})

    # ------------------------------------------------------------------
    # Request modification for prefill mode
    # ------------------------------------------------------------------

    async def test_update_request_for_prefill(self):
        """In prefill mode, request should be modified: disable streaming, set tokens to 1, add KV params."""
        server = self._create_server_skip_update_config(infer_mode="prefill")

        request_json = {
            "model": "abc",
            "stream": True,
            "max_tokens": 100,
            "max_completion_tokens": 200,
            "stream_options": {"include_usage": True},
        }

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep
        await AsyncVLLMServerPDSep.update_request_for_prefill(server, request_json)

        self.assertIn("kv_transfer_params", request_json)
        self.assertFalse(request_json["stream"])
        self.assertEqual(request_json["max_tokens"], 1)
        self.assertEqual(request_json["max_completion_tokens"], 1)
        self.assertNotIn("stream_options", request_json)

    # ------------------------------------------------------------------
    # Server endpoint tests
    # ------------------------------------------------------------------

    async def test_get_infer_mode(self):
        """get_infer_mode waits for server_ready and returns the mode."""
        server = self._create_server_skip_update_config(infer_mode="decode")
        server.infer_mode = "decode"

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep
        mode = await AsyncVLLMServerPDSep.get_infer_mode(server)

        server.server_ready.wait.assert_awaited_once()
        self.assertEqual(mode, "decode")

    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.ChatCompletionRequest")
    async def test_chat_completion_prefill_updates_request(self, mock_chat_req_cls):
        """Chat completion in prefill mode should modify the request before forwarding."""
        server = self._create_server_skip_update_config(infer_mode="prefill")

        server.openai_serving_chat = MagicMock()
        server.openai_serving_chat.create_chat_completion = AsyncMock(
            return_value=DummyChatCompletionResponse()
        )

        request_payload = {"model": "xxx", "messages": [], "stream": False}
        raw_request = DummyRequest(request_payload)

        mock_chat_req_cls.side_effect = lambda **kwargs: MagicMock(stream=kwargs.get("stream", False))

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep
        resp = await AsyncVLLMServerPDSep.chat_completion(server, raw_request)

        self.assertIsInstance(resp, JSONResponse)
        server.openai_serving_chat.create_chat_completion.assert_awaited_once()

    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.ChatCompletionRequest")
    async def test_chat_completion_streaming_response(self, mock_chat_req_cls):
        """Streaming chat completion should return a StreamingResponse."""
        server = self._create_server_skip_update_config(infer_mode="decode")

        async def fake_generator():
            yield b"data: hello\n\n"

        server.openai_serving_chat = MagicMock()
        server.openai_serving_chat.create_chat_completion = AsyncMock(return_value=fake_generator())

        request_payload = {"model": "xxx", "messages": [], "stream": True}
        raw_request = DummyRequest(request_payload)

        mock_chat_req_cls.side_effect = lambda **kwargs: MagicMock(stream=True)

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep
        resp = await AsyncVLLMServerPDSep.chat_completion(server, raw_request)

        self.assertIsInstance(resp, StreamingResponse)

    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.ChatCompletionRequest")
    async def test_chat_completion_error_response(self, mock_chat_req_cls):
        """Error responses from serving layer should be returned with correct status code."""
        server = self._create_server_skip_update_config(infer_mode="decode")

        server.openai_serving_chat = MagicMock()
        server.openai_serving_chat.create_chat_completion = AsyncMock(
            return_value=DummyErrorResponse(code=401)
        )

        request_payload = {"model": "xxx", "messages": [], "stream": False}
        raw_request = DummyRequest(request_payload)

        mock_chat_req_cls.side_effect = lambda **kwargs: MagicMock(stream=False)

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep
        resp = await AsyncVLLMServerPDSep.chat_completion(server, raw_request)

        self.assertIsInstance(resp, JSONResponse)
        self.assertEqual(resp.status_code, 401)

    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.CompletionRequest")
    async def test_completions_non_stream_response_and_logprobs_cleanup(self, mock_comp_req_cls):
        """Non-stream completion should clear logprobs fields for JSON serialization."""
        server = self._create_server_skip_update_config(infer_mode="decode")

        dummy_resp = DummyCompletionResponse(with_logprobs=True)

        server.openai_serving_completion = MagicMock()
        server.openai_serving_completion.create_completion = AsyncMock(return_value=dummy_resp)

        request_payload = {"model": "xxx", "prompt": "hi", "stream": False}
        raw_request = DummyRequest(request_payload)

        mock_comp_req_cls.side_effect = lambda **kwargs: MagicMock(stream=False)

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep
        resp = await AsyncVLLMServerPDSep.completions(server, raw_request)

        self.assertIsInstance(resp, JSONResponse)
        self.assertEqual(dummy_resp.choices[0].logprobs.token_logprobs, [])
        self.assertEqual(dummy_resp.choices[0].logprobs.top_logprobs, [])
        self.assertEqual(dummy_resp.choices[0].logprobs.text_offset, [])

    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.CompletionRequest")
    async def test_completions_streaming_response(self, mock_comp_req_cls):
        """Streaming completion returns StreamingResponse."""
        server = self._create_server_skip_update_config(infer_mode="decode")

        async def fake_generator():
            yield b"data: hi\n\n"

        server.openai_serving_completion = MagicMock()
        server.openai_serving_completion.create_completion = AsyncMock(return_value=fake_generator())

        request_payload = {"model": "xxx", "prompt": "hi", "stream": True}
        raw_request = DummyRequest(request_payload)

        mock_comp_req_cls.side_effect = lambda **kwargs: MagicMock(stream=True)

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep
        resp = await AsyncVLLMServerPDSep.completions(server, raw_request)

        self.assertIsInstance(resp, StreamingResponse)

    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.CompletionRequest")
    async def test_completions_error_response(self, mock_comp_req_cls):
        """Error responses from completions endpoint return correct status."""
        server = self._create_server_skip_update_config(infer_mode="decode")

        server.openai_serving_completion = MagicMock()
        server.openai_serving_completion.create_completion = AsyncMock(
            return_value=DummyErrorResponse(code=500)
        )

        request_payload = {"model": "xxx", "prompt": "hi", "stream": False}
        raw_request = DummyRequest(request_payload)

        mock_comp_req_cls.side_effect = lambda **kwargs: MagicMock(stream=False)

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep
        resp = await AsyncVLLMServerPDSep.completions(server, raw_request)

        self.assertIsInstance(resp, JSONResponse)
        self.assertEqual(resp.status_code, 500)

    async def test_get_workload(self):
        """get_workload returns workload statistics as JSON."""
        server = self._create_server_skip_update_config(infer_mode="decode")

        server.ins_workload = MagicMock()
        server.ins_workload.to_dict.return_value = {"a": 1}

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep
        resp = await AsyncVLLMServerPDSep.get_workload(server, DummyRequest({}))

        self.assertIsInstance(resp, JSONResponse)
        self.assertEqual(resp.status_code, 200)

    async def test_cancel_requests(self):
        """cancel_requests forwards to engine.abort."""
        server = self._create_server_skip_update_config(infer_mode="decode")

        server.engine = MagicMock()
        server.engine.abort = AsyncMock()

        req = DummyRequest({"requests": ["r1", "r2"]})

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep
        await AsyncVLLMServerPDSep.cancel_requests(server, req)

        server.engine.abort.assert_awaited_once_with(["r1", "r2"])

    # ------------------------------------------------------------------
    # Helper method tests
    # ------------------------------------------------------------------

    def test_update_prefill_params_not_prefill(self):
        """When not in prefill mode, update_prefill_params does nothing."""
        server = self._create_server_skip_update_config(infer_mode="decode")

        original_enforce_eager = server.config.enforce_eager
        original_max_num_seqs = server.config.max_num_seqs
        original_max_num_batched_tokens = server.config.max_num_batched_tokens
        original_gpu_memory_utilization = server.config.gpu_memory_utilization
        original_max_model_len = server.config.max_model_len

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep
        AsyncVLLMServerPDSep.update_prefill_params(server)

        self.assertEqual(server.config.enforce_eager, original_enforce_eager)
        self.assertEqual(server.config.max_num_seqs, original_max_num_seqs)
        self.assertEqual(server.config.max_num_batched_tokens, original_max_num_batched_tokens)
        self.assertEqual(server.config.gpu_memory_utilization, original_gpu_memory_utilization)
        self.assertEqual(server.config.max_model_len, original_max_model_len)

    def test_update_prefill_params_with_none_overrides(self):
        """If prefill override values are None, keep original config."""
        server = self._create_server_skip_update_config(infer_mode="prefill")

        server.config.prefill_enforce_eager = None
        server.config.prefill_max_num_seqs = None
        server.config.prefill_max_num_batched_tokens = None
        server.config.prefill_gpu_memory_utilization = None
        server.config.prefill_max_model_len = None

        original_enforce_eager = server.config.enforce_eager
        original_max_num_seqs = server.config.max_num_seqs
        original_max_num_batched_tokens = server.config.max_num_batched_tokens
        original_gpu_memory_utilization = server.config.gpu_memory_utilization
        original_max_model_len = server.config.max_model_len

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep
        AsyncVLLMServerPDSep.update_prefill_params(server)

        self.assertEqual(server.config.enforce_eager, original_enforce_eager)
        self.assertEqual(server.config.max_num_seqs, original_max_num_seqs)
        self.assertEqual(server.config.max_num_batched_tokens, original_max_num_batched_tokens)
        self.assertEqual(server.config.gpu_memory_utilization, original_gpu_memory_utilization)
        self.assertEqual(server.config.max_model_len, original_max_model_len)

    @patch.dict(os.environ, {"DISAGGREGATED_PREFILL_RANK_TABLE_PATH": "/tmp/ranktable.json"})
    def test_update_config_with_explicit_infer_mode(self):
        """When infer_mode is explicitly provided, skip auto-detection."""
        server = self._create_server_skip_update_config(infer_mode=None)

        if hasattr(server.config, "kv_transfer_config"):
            delattr(server.config, "kv_transfer_config")

        ranktable_data = {
            "prefill_device_list": [{"server_id": "1.1.1.1"}],
            "decode_device_list": [{"server_id": "2.2.2.2"}],
        }
        m = mock_open(read_data=json.dumps(ranktable_data))

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep

        with patch("builtins.open", m), patch(
            "agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.ray.util.get_node_ip_address",
            return_value="9.9.9.9",  # IP not in ranktable, but explicit mode overrides
        ):
            AsyncVLLMServerPDSep.update_config(server, infer_mode="prefill")

        self.assertEqual(server.infer_mode, "prefill")
        self.assertEqual(server.config.kv_transfer_config["kv_role"], "kv_producer")
        self.assertEqual(server.config.max_model_len, 1024)

    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.ChatCompletionRequest")
    async def test_chat_completion_no_prefill_update(self, mock_req):
        """When not in prefill mode, chat request is passed through unmodified."""
        server = self._create_server_skip_update_config(infer_mode="decode")

        server.openai_serving_chat = MagicMock()
        server.openai_serving_chat.create_chat_completion = AsyncMock(
            return_value=DummyChatCompletionResponse()
        )

        request_payload = {"model": "xxx", "messages": [], "stream": False, "max_tokens": 100}
        raw_request = DummyRequest(request_payload)

        mock_req.side_effect = lambda **kwargs: MagicMock(stream=kwargs.get("stream", False))

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep
        resp = await AsyncVLLMServerPDSep.chat_completion(server, raw_request)

        self.assertIsInstance(resp, JSONResponse)

        call_args = mock_req.call_args[1]
        self.assertNotIn("kv_transfer_params", call_args)
        self.assertEqual(call_args.get("max_tokens"), 100)

    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.CompletionRequest")
    async def test_completions_no_logprobs_cleanup(self, mock_req):
        """If response has no logprobs, cleanup step does nothing."""
        server = self._create_server_skip_update_config(infer_mode="decode")

        dummy_resp = DummyCompletionResponse(with_logprobs=False)
        server.openai_serving_completion = MagicMock()
        server.openai_serving_completion.create_completion = AsyncMock(return_value=dummy_resp)

        request_payload = {"model": "xxx", "prompt": "hi", "stream": False}
        raw_request = DummyRequest(request_payload)

        mock_req.side_effect = lambda **kwargs: MagicMock(stream=False)

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep
        resp = await AsyncVLLMServerPDSep.completions(server, raw_request)

        self.assertIsInstance(resp, JSONResponse)
        self.assertEqual(len(dummy_resp.choices), 0)

    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.CompletionRequest")
    async def test_completions_with_choices_but_no_logprobs(self, mock_req):
        """If choice exists but logprobs is None, do not attempt to clear fields."""
        server = self._create_server_skip_update_config(infer_mode="decode")

        dummy_resp = DummyCompletionResponse(with_logprobs=False)
        choice = MagicMock()
        choice.logprobs = None
        dummy_resp.choices = [choice]

        server.openai_serving_completion = MagicMock()
        server.openai_serving_completion.create_completion = AsyncMock(return_value=dummy_resp)

        request_payload = {"model": "xxx", "prompt": "hi", "stream": False}
        raw_request = DummyRequest(request_payload)

        mock_req.side_effect = lambda **kwargs: MagicMock(stream=False)

        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep
        resp = await AsyncVLLMServerPDSep.completions(server, raw_request)

        self.assertIsInstance(resp, JSONResponse)
        self.assertIsNone(choice.logprobs)

    def test_get_node_ip_from_ranktable_missing_env_var(self):
        """If environment variable is not set, raise AssertionError."""
        server = self._create_server_skip_update_config()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                server.get_node_ip_from_ranktable()
            self.assertIn("Can't find ranktable file", str(ctx.exception))

    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.CompilationConfig")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.AsyncEngineArgs")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.AsyncLLM")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.ray.get_runtime_context")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.asyncio.create_task")
    async def test_init_engine_disable_log_stats_true_no_task(
        self, mock_create_task, mock_ctx, mock_async_llm, mock_engine_args, mock_comp_config
    ):
        server = self._create_server_skip_update_config(infer_mode="decode")
        server.config.disable_log_stats = True
        server.config.cudagraph_capture_sizes = None
        server.config.kv_transfer_config = {"kv_role": "kv_consumer"}
        server.get_server_address = AsyncMock(return_value="127.0.0.1:8000")

        fake_engine_args = MagicMock()
        fake_engine_args.create_engine_config.return_value = MagicMock()
        mock_engine_args.return_value = fake_engine_args

        fake_engine = MagicMock()
        fake_engine.model_config = MagicMock()
        mock_async_llm.from_vllm_config.return_value = fake_engine

        ctx = MagicMock()
        ctx.namespace = "ns"
        mock_ctx.return_value = ctx

        with patch.dict(os.environ, {"VLLM_DP_SIZE": "2"}):
            await server.init_engine()

        mock_create_task.assert_not_called()

    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.CompilationConfig")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.AsyncEngineArgs")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.AsyncLLM")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.ray.get_runtime_context")
    async def test_init_engine_with_cudagraph_capture_sizes_empty_string(
        self, mock_ctx, mock_async_llm, mock_engine_args, mock_comp_config
    ):
        server = self._create_server_skip_update_config(infer_mode="decode")
        server.config.cudagraph_capture_sizes = ""
        server.config.kv_transfer_config = {"kv_role": "kv_consumer"}
        server.get_server_address = AsyncMock(return_value="127.0.0.1:8000")

        fake_engine_args = MagicMock()
        fake_engine_args.create_engine_config.return_value = MagicMock()
        mock_engine_args.return_value = fake_engine_args

        fake_engine = MagicMock()
        fake_engine.model_config = MagicMock()
        mock_async_llm.from_vllm_config.return_value = fake_engine

        ctx = MagicMock()
        ctx.namespace = "ns"
        mock_ctx.return_value = ctx

        with patch.dict(os.environ, {"VLLM_DP_SIZE": "2"}):
            await server.init_engine()

        mock_comp_config.assert_called_once()
        call_kwargs = mock_comp_config.call_args[1]
        self.assertEqual(call_kwargs.get("cudagraph_capture_sizes"), [])

    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.CompilationConfig")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.AsyncEngineArgs")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.AsyncLLM")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.ray.get_runtime_context")
    async def test_init_engine_load_format_not_megatron(
        self, mock_ctx, mock_async_llm, mock_engine_args, mock_comp_config
    ):
        server = self._create_server_skip_update_config(infer_mode="decode")
        server.config.load_format = "auto"
        server.config.kv_transfer_config = {"kv_role": "kv_consumer"}
        server.get_server_address = AsyncMock(return_value="127.0.0.1:8000")

        fake_engine_args = MagicMock()
        fake_engine_args.create_engine_config.return_value = MagicMock()
        mock_engine_args.return_value = fake_engine_args

        fake_engine = MagicMock()
        fake_engine.model_config = MagicMock()
        mock_async_llm.from_vllm_config.return_value = fake_engine

        ctx = MagicMock()
        ctx.namespace = "ns"
        mock_ctx.return_value = ctx

        with patch.dict(os.environ, {"VLLM_DP_SIZE": "2"}):
            await server.init_engine()

        call_kwargs = mock_engine_args.call_args[1]
        self.assertEqual(call_kwargs.get("load_format"), "auto")


class TestAsyncVLLMServerPDSepInitEngine(unittest.IsolatedAsyncioTestCase):
    """Separate test class for init_engine method to avoid interference."""

    def setUp(self):
        self.module_patcher = patch.dict(sys.modules, _build_fake_modules())
        self.module_patcher.start()

        self.target_mod = _reload_target_module()
        bind_dummy_response_base_classes(self.target_mod)

    def tearDown(self):
        self.module_patcher.stop()

    async def test_init_engine_main_flow(self):
        """
        Full initialization flow: create engine, serving objects, and optional stat logger.
        """
        AsyncVLLMServerPDSep = self.target_mod.AsyncVLLMServerPDSep

        # Create server instance without running __init__ or update_config
        with patch(
            "agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.AsyncVLLMServer.__init__",
            return_value=None,
        ), patch(
            "agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.AsyncVLLMServerPDSep.update_config",
            return_value=None,
        ):
            server = AsyncVLLMServerPDSep(
                config=MagicMock(),
                tokenizer_name_or_path="/models/test_model",
                vllm_dp_size=2,
                vllm_dp_rank=1,
                wg_prefix="wg",
                infer_mode="decode",
            )

        # Set up a realistic config
        config = MagicMock()
        config.trust_remote_code = True
        config.infer_tensor_parallel_size = 2
        config.infer_pipeline_parallel_size = 1
        config.max_model_len = 2048
        config.max_num_batched_tokens = 1024
        config.enable_sleep_mode = False
        config.dtype = "float16"
        config.enforce_eager = False
        config.gpu_memory_utilization = 0.8
        config.enable_chunked_prefill = True
        config.enable_prefix_caching = False
        config.enable_expert_parallel = False
        config.disable_log_stats = False
        config.max_num_seqs = 8
        config.load_format = "megatron"
        config.cudagraph_capture_sizes = "1, 2, 4"

        sampling = MagicMock()
        sampling.logprobs = 0
        sampling.max_tokens = 16
        sampling.top_p = 1.0
        sampling.top_k = -1
        sampling.min_p = 0.0
        sampling.temperature = 1.0
        config.sampling_config = sampling

        config.kv_transfer_config = {"kv_role": "kv_consumer"}

        server.config = config
        server.tokenizer_name_or_path = "/models/test_model"
        server.vllm_dp_rank = 1
        server.vllm_dp_size = 2
        server.wg_prefix = "wg"
        server.get_server_address = AsyncMock(return_value="127.0.0.1:8000")

        fake_engine_args = MagicMock()
        fake_engine_args.create_engine_config.return_value = MagicMock()

        fake_engine = MagicMock()
        fake_engine.model_config = MagicMock()
        fake_async_llm_cls = MagicMock()
        fake_async_llm_cls.from_vllm_config.return_value = fake_engine

        fake_models = MagicMock()
        fake_chat = MagicMock()
        fake_completion = MagicMock()

        def fake_create_task(coro):
            coro.close()
            return MagicMock()

        with patch.dict(os.environ, {"VLLM_DP_SIZE": "2"}), \
             patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.AsyncEngineArgs",
                   return_value=fake_engine_args) as mock_engine_args_cls, \
             patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.AsyncLLM",
                   fake_async_llm_cls), \
             patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.OpenAIServingModels",
                   return_value=fake_models) as mock_models_cls, \
             patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.OpenAIServingChat",
                   return_value=fake_chat) as mock_chat_cls, \
             patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.OpenAIServingCompletion",
                   return_value=fake_completion) as mock_completion_cls, \
             patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.ray.get_runtime_context") as mock_ctx, \
             patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.asyncio.create_task",
                   side_effect=fake_create_task) as mock_create_task, \
             patch("agentic_rl.runner.infer_adapter.vllm.vllm_async_server_pd.vllm_log_stats_periodically",
                   new_callable=AsyncMock) as mock_log_stats:

            ctx = MagicMock()
            ctx.namespace = "ns"
            mock_ctx.return_value = ctx

            await server.init_engine()

            mock_engine_args_cls.assert_called_once()
            fake_engine_args.create_engine_config.assert_called_once()
            fake_async_llm_cls.from_vllm_config.assert_called_once()
            mock_models_cls.assert_called_once()
            mock_chat_cls.assert_called_once()
            mock_completion_cls.assert_called_once()
            mock_create_task.assert_called_once()
            mock_log_stats.assert_called_once_with(server)


if __name__ == "__main__":
    unittest.main()