#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import asyncio
import importlib
import sys
import types
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

from unittest.mock import patch, MagicMock
import pytest

from agentic_rl.configs.agentic_rl_config import AgenticRLConfig, GenConfig, SamplingConfig

FAKE_IP = "192.168.1.1"


class AsyncEngineArgs:
    """Stub AsyncEngineArgs capturing initialization kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_engine_config(self):
        # Minimal config object; real fields are set by AsyncVLLMServer
        return SimpleNamespace(instance_id=None, train_backend=None)


class Executor:
    """Minimal base class for ExternalRayDistributedExecutor."""
    pass


class _CompletionResult:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


class OpenAIServingCompletion:
    def __init__(self, engine, model_config, models, *args, **kwargs):
        self.engine = engine
        self.model_config = model_config
        self.models = models
        self.args = args
        self.kwargs = kwargs
        self.return_value = {"choices": []}

    async def create_completion(self, request):
        # Echo a simple static payload for tests
        return _CompletionResult(self.return_value)


@dataclass
class BaseModelPath:
    name: str
    model_path: str


class OpenAIServingModels:
    def __init__(self, engine, model_config, base_model_paths):
        self.engine = engine
        self.model_config = model_config
        self.base_model_paths = base_model_paths


class CompletionRequest:
    """Stub CompletionRequest capturing initialization kwargs."""

    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.kwargs = kwargs


class ChatCompletionRequest:
    """Stub ChatCompletionRequest capturing initialization kwargs."""
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.kwargs = kwargs


class OpenAIServingChat:
    def __init__(self, engine, model_config, models, *args, **kwargs):
        self.engine = engine
        self.model_config = model_config
        self.models = models
        self.args = args
        self.kwargs = kwargs


class AsyncLLM:
    """Stub AsyncLLM used by AsyncVLLMServer.init_engine."""

    def __init__(self, config):
        self.config = config
        self.model_config = SimpleNamespace()

    @classmethod
    def from_vllm_config(cls, config):
        return cls(config)

    async def wake_up(self, _tags=None):
        return None

    async def sleep(self):
        return None


class _RuntimeContext:
    def __init__(self, namespace: str = "test_ns"):
        self.namespace = namespace


class RayError(Exception):
    """Stub RayError for ExternalRayDistributedExecutor."""


class _Router:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def add_api_route(self, *args, **kwargs):
        pass


class _FastAPI:
    def __init__(self, lifespan, *args, **kwargs):
        self.lifespan = lifespan
        self.router = _Router(args, kwargs)


class _UviConfig:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _UviServer:
    def __init__(self, config: _UviConfig):
        self.config = config
        self.should_exit = False

    async def serve(self):
        pass


def get_runtime_context():
    return _RuntimeContext()


def _create_task(obj=None):
    return (lambda x: x) if obj is None else obj


def _remote(obj=None, **_kwargs):
    # Behave like identity decorator: @ray.remote returns original class
    return (lambda x: x) if obj is None else obj


def _import_after_mocking(monkeypatch, module_name, preinstall):
    """Helper to install mock modules and import target module cleanly."""
    for fullname, module in preinstall.items():
        monkeypatch.setitem(sys.modules, fullname, module)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name)


@pytest.fixture
def vllm_async_server_mod(monkeypatch):
    """Fixture providing the vllm_async_server module with mocked dependencies."""
    # Mock ray
    ray_mod = ModuleType("ray")
    ray_mod.remote = _remote
    ray_mod.get_runtime_context = get_runtime_context
    ray_ex_mod = ModuleType("ray.exceptions")
    ray_mod._private = MagicMock()
    ray_mod._private.services = MagicMock()
    # ray_mod._private.services.get_node_ip_address.return_value = "127.0.0.1"

    ray_ex_mod.RayError = RayError

    preinstall = {
        "ray": ray_mod,
        "ray.exceptions": ray_ex_mod,
    }

    asyncio.create_task = _create_task

    vllm_pkg = ModuleType("vllm")
    vllm_pkg.__path__ = []

    engine_pkg = ModuleType("vllm.engine")
    engine_pkg.__path__ = []
    arg_utils_mod = ModuleType("vllm.engine.arg_utils")
    arg_utils_mod.AsyncEngineArgs = AsyncEngineArgs

    v1_pkg = ModuleType("vllm.v1")
    v1_pkg.__path__ = []
    v1_engine_pkg = ModuleType("vllm.v1.engine")
    v1_engine_pkg.__path__ = []
    v1_executor_pkg = ModuleType("vllm.v1.executor")
    v1_executor_pkg.__path__ = []
    v1_executor_abstract_mod = ModuleType("vllm.v1.executor.abstract")
    v1_executor_abstract_mod.Executor = Executor

    v1_async_llm_mod = ModuleType("vllm.v1.engine.async_llm")
    v1_async_llm_mod.AsyncLLM = AsyncLLM

    entrypoints_pkg = ModuleType("vllm.entrypoints")
    entrypoints_pkg.__path__ = []
    openai_pkg = ModuleType("vllm.entrypoints.openai")
    openai_pkg.__path__ = []

    protocol_mod = ModuleType("vllm.entrypoints.openai.protocol")
    protocol_mod.CompletionRequest = CompletionRequest
    protocol_mod.ChatCompletionRequest = ChatCompletionRequest

    serving_models_mod = ModuleType("vllm.entrypoints.openai.serving_models")
    serving_models_mod.BaseModelPath = BaseModelPath
    serving_models_mod.OpenAIServingModels = OpenAIServingModels
    serving_chat_mod = ModuleType("vllm.entrypoints.openai.serving_chat")
    serving_chat_mod.OpenAIServingChat = OpenAIServingChat
    serving_completion_mod = ModuleType("vllm.entrypoints.openai.serving_completion")
    serving_completion_mod.OpenAIServingCompletion = OpenAIServingCompletion

    fastapi_pkg = ModuleType("fastapi")
    fastapi_pkg.__path__ = []
    fastapi_pkg.FastAPI = _FastAPI
    uvicorn_pkg = ModuleType("uvicorn")
    uvicorn_pkg.__path__ = []
    uvicorn_pkg.Config = _UviConfig
    uvicorn_pkg.Server = _UviServer

    preinstall.update({
        "vllm": vllm_pkg,
        "vllm.engine": engine_pkg,
        "vllm.engine.arg_utils": arg_utils_mod,
        "vllm.v1": v1_pkg,
        "vllm.v1.engine": v1_engine_pkg,
        "vllm.v1.engine.async_llm": v1_async_llm_mod,
        "vllm.v1.executor": v1_executor_pkg,
        "vllm.v1.executor.abstract": v1_executor_abstract_mod,
        "vllm.entrypoints": entrypoints_pkg,
        "vllm.entrypoints.openai": openai_pkg,
        "vllm.entrypoints.openai.protocol": protocol_mod,
        "vllm.entrypoints.openai.serving_models": serving_models_mod,
        "vllm.entrypoints.openai.serving_chat": serving_chat_mod,
        "vllm.entrypoints.openai.serving_completion": serving_completion_mod,
        "fastapi": fastapi_pkg,
        "uvicorn": uvicorn_pkg,
    })

    mod = _import_after_mocking(
        monkeypatch,
        "agentic_rl.runner.infer_adapter.vllm.vllm_async_server",
        preinstall)
    return mod


@pytest.fixture
def valid_configs(monkeypatch):
    """Provide valid GenConfig and AgenticRLConfig instance"""
    monkeypatch.setattr(
        "agentic_rl.configs.agentic_rl_config.FileCheck.check_data_path_is_valid",
        lambda _path: None,
    )

    gen_cfg = GenConfig()
    gen_cfg.infer_tensor_parallel_size = 2
    gen_cfg.infer_pipeline_parallel_size = 1
    gen_cfg.max_model_len = 4096
    gen_cfg.max_num_batched_tokens = 1024
    gen_cfg.enable_prefix_caching = True
    gen_cfg.trust_remote_code = True
    gen_cfg.dtype = "bfloat16"
    gen_cfg.gpu_memory_utilization = 0.6
    gen_cfg.max_num_seqs = 3

    gen_cfg.sampling_config = SamplingConfig(
        logprobs=2,
        max_tokens=256,
        top_p=0.9,
        top_k=32,
        min_p=0.1,
        temperature=0.7,
    )

    agent_cfg = AgenticRLConfig()
    agent_cfg.load_format = "megatron"
    agent_cfg.enable_sleep_mode = True
    agent_cfg.train_backend = "mindspeed_rl"

    return gen_cfg, agent_cfg


@pytest.fixture
@patch('agentic_rl.runner.infer_adapter.async_server_base.ray._private.services.get_node_ip_address')
def server_instance(mock_get_node_ip_address, vllm_async_server_mod, valid_configs):
    """Construct a real AsyncVLLMServer instance with mocked deps."""
    gen_cfg, agent_cfg = valid_configs
    mock_get_node_ip_address.return_value = FAKE_IP
    AsyncVLLMServer = vllm_async_server_mod.AsyncVLLMServer

    server = AsyncVLLMServer(
        gen_cfg,
        agent_cfg,
        "/path/to/tokenizer",
        vllm_dp_size=4,
        vllm_dp_rank=1,
        wg_prefix="wg",
    )

    return server


class TestBuildHelpers:
    """Tests for helper methods on AsyncVLLMServer."""

    def test_build_generation_config(self, vllm_async_server_mod):
        SamplingCfg = SamplingConfig(
            logprobs=3,
            max_tokens=512,
            top_p=0.8,
            top_k=64,
            min_p=0.2,
            temperature=0.5,
        )

        cfg = vllm_async_server_mod.AsyncVLLMServer._build_generation_config(SamplingCfg)

        assert cfg['n'] == 1
        assert cfg['logprobs'] == SamplingCfg.logprobs
        assert cfg['max_new_tokens'] == SamplingCfg.max_tokens
        assert cfg['top_p'] == SamplingCfg.top_p
        assert cfg['top_k'] == SamplingCfg.top_k
        assert cfg['min_p'] == SamplingCfg.min_p
        assert cfg['temperature'] == SamplingCfg.temperature

    def test_build_vllm_config_success(self, server_instance, valid_configs, vllm_async_server_mod):
        gen_cfg, agent_cfg = valid_configs
        vllm_config = server_instance._build_vllm_config(gen_cfg, "/model/path")

        assert isinstance(vllm_config, vllm_async_server_mod.VLLMConfig)
        assert vllm_config.model_path == "/model/path"
        assert vllm_config.tensor_parallel_size == gen_cfg.infer_tensor_parallel_size
        assert vllm_config.pipeline_parallel_size == gen_cfg.infer_pipeline_parallel_size
        assert vllm_config.max_model_len == gen_cfg.max_model_len
        assert vllm_config.max_num_batched_tokens == gen_cfg.max_num_batched_tokens
        assert vllm_config.dtype == gen_cfg.dtype
        assert vllm_config.enforce_eager == gen_cfg.enforce_eager
        assert vllm_config.gpu_memory_utilization == gen_cfg.gpu_memory_utilization
        assert vllm_config.load_format == agent_cfg.load_format
        assert vllm_config.enable_sleep_mode == agent_cfg.enable_sleep_mode
        assert vllm_config.enable_prefix_caching == gen_cfg.enable_prefix_caching
        assert vllm_config.trust_remote_code == gen_cfg.trust_remote_code
        assert vllm_config.max_num_seqs == gen_cfg.max_num_seqs
        assert vllm_config.sampling_config is gen_cfg.sampling_config

    def test_build_vllm_config_missing_attribute_raises(self, server_instance):
        bad_cfg = types.SimpleNamespace()

        with pytest.raises(AttributeError, match="Missing required config field"):
            server_instance._build_vllm_config(bad_cfg, "/model/path")

    def test_build_engine_args_success(self, server_instance, valid_configs, vllm_async_server_mod):
        gen_cfg, _ = valid_configs
        vllm_config = server_instance._build_vllm_config(gen_cfg, "/model/path")
        generation_config = server_instance._build_generation_config(vllm_config.sampling_config)

        engine_args = server_instance._build_engine_args(vllm_config, generation_config)

        kwargs = engine_args.kwargs
        assert kwargs['model'] == vllm_config.model_path
        assert kwargs['enable_sleep_mode'] == vllm_config.enable_sleep_mode
        assert kwargs['tensor_parallel_size'] == vllm_config.tensor_parallel_size
        assert kwargs['pipeline_parallel_size'] == vllm_config.pipeline_parallel_size
        assert kwargs['dtype'] == vllm_config.dtype
        assert kwargs['enforce_eager'] == vllm_config.enforce_eager
        assert kwargs['gpu_memory_utilization'] == vllm_config.gpu_memory_utilization
        assert kwargs['max_model_len'] == vllm_config.max_model_len
        assert kwargs['max_num_batched_tokens'] == vllm_config.max_num_batched_tokens
        assert kwargs['enable_prefix_caching'] == vllm_config.enable_prefix_caching
        assert kwargs['trust_remote_code'] == vllm_config.trust_remote_code
        assert kwargs['seed'] == server_instance.server_config.vllm_dp_rank
        assert kwargs['max_num_seqs'] == vllm_config.max_num_seqs
        assert kwargs['override_generation_config'] == generation_config
        assert kwargs['distributed_executor_backend'] is vllm_async_server_mod.ExternalRayDistributedExecutor
        assert kwargs['load_format'] == 'dummy'

    def test_build_engine_args_large_batch_logs_warning(
            self, server_instance, valid_configs, vllm_async_server_mod, monkeypatch):
        gen_cfg, _ = valid_configs
        gen_cfg.max_num_batched_tokens = vllm_async_server_mod.MAX_BATCHED_TOKENS_WARNING_THRESHOLD + 1
        vllm_config = server_instance._build_vllm_config(gen_cfg, "/model/path")
        generation_config = server_instance._build_generation_config(vllm_config.sampling_config)

        warnings = []

        def _warn(msg):
            warnings.append(msg)

        monkeypatch.setattr(vllm_async_server_mod.logger, "warning", _warn)

        _ = server_instance._build_engine_args(vllm_config, generation_config)

        assert any("max_num_batched_tokens" in msg for msg in warnings)
        assert any(str(vllm_async_server_mod.MAX_BATCHED_TOKENS_WARNING_THRESHOLD) in msg for msg in warnings)


class TestInitEngine:
    """Tests for AsyncVLLMServer.init_engine.
    These tests are written as synchronous tests that drive the underlying
    async API via asyncio.run so that they do not require pytest-asyncio or
    any other async test plugin.
    """

    def test_init_engine_happy_path(self, server_instance, vllm_async_server_mod):
        asyncio.run(server_instance.init_engine())

        assert server_instance.engine is not None
        engine = server_instance.engine

        expected_instance_id = (
            f"test_ns:{server_instance.server_config.wg_prefix}:"
            f"{server_instance.server_config.vllm_dp_size}:{server_instance.server_config.vllm_dp_rank}"
        )
        assert engine.config.instance_id == expected_instance_id

        assert server_instance.openai_serving_chat is not None
        assert server_instance.openai_serving_completion is not None

    def test_init_engine_failure_create_engine_config(self, server_instance, vllm_async_server_mod, monkeypatch):
        class BadArgs:
            def create_engine_config(self):
                raise ValueError("bad config")

        monkeypatch.setattr(server_instance, "_build_engine_args", lambda *_args, **_kwargs: BadArgs())

        with pytest.raises(RuntimeError, match="Failed to create engine config"):
            asyncio.run(server_instance.init_engine())

    def test_init_engine_failure_async_llm_init(self, server_instance, vllm_async_server_mod, monkeypatch):
        class DummyArgs:
            def create_engine_config(self):
                return SimpleNamespace(instance_id=None, train_backend=None)

        monkeypatch.setattr(server_instance, "_build_engine_args", lambda *_args, **_kwargs: DummyArgs())

        def _raise(_config):
            raise ValueError("engine init failed")

        monkeypatch.setattr(vllm_async_server_mod.AsyncLLM, "from_vllm_config", staticmethod(_raise))

        with pytest.raises(RuntimeError, match="Failed to initialize AsyncLLM engine"):
            asyncio.run(server_instance.init_engine())


class TestCompletions:
    """Tests for AsyncVLLMServer.completions."""

    def test_completions_happy_path(self, server_instance, vllm_async_server_mod, monkeypatch):
        # Ensure validation is a no-op
        monkeypatch.setattr(
            vllm_async_server_mod.CompletionRequestChecker,
            "validate_input",
            staticmethod(lambda _req: None),
        )

        class StubCompletion:
            def __init__(self):
                self.last_request = None

            async def create_completion(self, request):
                self.last_request = request
                return types.SimpleNamespace(model_dump=lambda: {"choices": ["ok"]})

        server_instance.openai_serving_completion = StubCompletion()
        raw_request = {"prompt": "hello", "model_name": "foo"}
        result = asyncio.run(server_instance.completions(dict(raw_request)))

        assert result == {"choices": ["ok"]}
        assert "model_name" not in vllm_async_server_mod.CompletionRequest.last_kwargs
        assert vllm_async_server_mod.CompletionRequest.last_kwargs["prompt"] == "hello"

    def test_completions_validation_failure(self, server_instance, vllm_async_server_mod, monkeypatch):
        def _raise(_req):
            raise ValueError("invalid")

        monkeypatch.setattr(
            vllm_async_server_mod.CompletionRequestChecker,
            "validate_input",
            staticmethod(_raise),
        )

        with pytest.raises(ValueError, match="Input validation failed"):
            asyncio.run(server_instance.completions({"prompt": "bad"}))

    def test_completions_missing_serving_completion(self, server_instance):
        server_instance.openai_serving_completion = None

        with pytest.raises(RuntimeError, match="OpenAI serving completion not initialized"):
            asyncio.run(server_instance.completions({"prompt": "hello"}))

    def test_completions_request_parsing_failure(self, server_instance, vllm_async_server_mod, monkeypatch):
        def _raise(**_kwargs):
            raise ValueError("bad request")

        monkeypatch.setattr(vllm_async_server_mod, "CompletionRequest", _raise)
        monkeypatch.setattr(
            vllm_async_server_mod.CompletionRequestChecker,
            "validate_input",
            staticmethod(lambda _req: None),
        )
        server_instance.openai_serving_completion = types.SimpleNamespace(
            create_completion=lambda _req: types.SimpleNamespace(model_dump=lambda: {})
        )

        with pytest.raises(ValueError, match="Failed to parse completion request"):
            asyncio.run(server_instance.completions({"prompt": "hello"}))


class TestChatCompletions:
    """Tests for AsyncVLLMServer.chat_completions."""

    def test_chat_completions_happy_path(self, server_instance, vllm_async_server_mod, monkeypatch):
        # Ensure validation is a no-op
        monkeypatch.setattr(
            vllm_async_server_mod.CompletionRequestChecker,
            "validate_chat_input",
            staticmethod(lambda _req: None),
        )

        class StubChatCompletion:
            def __init__(self):
                self.last_request = None

            async def create_chat_completion(self, request):
                self.last_request = request
                return types.SimpleNamespace(model_dump=lambda: {"choices": ["ok"]})

        server_instance.openai_serving_chat = StubChatCompletion()
        raw_request = {"messages": [{"role": "user", "content": "hello"}], "model_name": "foo"}
        result = asyncio.run(server_instance.chat_completions(dict(raw_request)))

        assert result == {"choices": ["ok"]}
        assert "model_name" not in vllm_async_server_mod.ChatCompletionRequest.last_kwargs
        assert vllm_async_server_mod.ChatCompletionRequest.last_kwargs["messages"] == [
            {"role": "user", "content": "hello"}]

    def test_chat_completions_validation_failure(self, server_instance, vllm_async_server_mod, monkeypatch):
        def _raise(_req):
            raise ValueError("invalid")

        monkeypatch.setattr(
            vllm_async_server_mod.CompletionRequestChecker,
            "validate_chat_input",
            staticmethod(_raise),
        )

        with pytest.raises(ValueError, match="Input validation failed"):
            asyncio.run(server_instance.chat_completions({"messages": [{"role": "user", "content": "bad"}]}))

    def test_chat_completions_missing_serving_chat(self, server_instance):
        server_instance.openai_serving_chat = None

        with pytest.raises(RuntimeError, match="OpenAI serving chat is not initialized"):
            asyncio.run(server_instance.chat_completions({"messages": [{"role": "user", "content": "hello"}]}))

    def test_chat_completions_request_parsing_failure(self, server_instance, vllm_async_server_mod, monkeypatch):
        def _raise(**_kwargs):
            raise ValueError("bad request")

        monkeypatch.setattr(vllm_async_server_mod, "ChatCompletionRequest", _raise)
        monkeypatch.setattr(
            vllm_async_server_mod.CompletionRequestChecker,
            "validate_chat_input",
            staticmethod(lambda _req: None),
        )
        server_instance.openai_serving_chat = types.SimpleNamespace(
            create_chat_completion=lambda _req: types.SimpleNamespace(model_dump=lambda: {})
        )

        with pytest.raises(ValueError, match="Failed to parse chat completion request"):
            asyncio.run(server_instance.chat_completions({"messages": [{"role": "user", "content": "hello"}]}))


class TestStartFastapiServer:
    """Tests for start FastAPI Server."""

    def test_start_fastapi_server_success(self, server_instance, vllm_async_server_mod, monkeypatch):
        monkeypatch.setattr(server_instance, "_get_free_port", lambda: 8000)
        asyncio.run(server_instance._start_fastapi_server())

        assert server_instance.port == 8000

    def test_start_fastapi_server_cancelled_error(self, server_instance, vllm_async_server_mod, monkeypatch):
        monkeypatch.setattr(server_instance, "_get_free_port", lambda: 8000)

        async def _raise(self):
            raise asyncio.CancelledError()

        monkeypatch.setattr(vllm_async_server_mod.uvicorn.Server, "serve", _raise)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(server_instance._start_fastapi_server())
            assert vllm_async_server_mod.uvicorn.Server.should_exit is True

    def test_start_fastapi_server_other_error(self, server_instance, vllm_async_server_mod, monkeypatch):
        monkeypatch.setattr(server_instance, "_get_free_port", lambda: 8000)
        monkeypatch.setattr(vllm_async_server_mod.uvicorn.Server, "serve", RuntimeError)
        with pytest.raises(RuntimeError, match="Unexpected error during start vllm inference"):
            asyncio.run(server_instance._start_fastapi_server())

    @patch('socket.socket')
    def test_get_free_port(self, mock_socket, server_instance):
        import socket
        mock_sock = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock

        mock_sock.getsockname.return_value = (FAKE_IP, 12345)
        port = server_instance._get_free_port()

        mock_socket.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        mock_sock.bind.assert_called_once_with((FAKE_IP, 0))
        assert port == 12345


class TestWakeUpAndSleep:
    """Tests for AsyncVLLMServer.wake_up and sleep."""

    def test_wake_up_success_and_validation(self, server_instance):
        class StubEngine:
            def __init__(self):
                self.calls = []

            async def wake_up(self, tags=None):
                self.calls.append(("wake_up", tags))

        engine = StubEngine()
        server_instance.engine = engine

        asyncio.run(server_instance.wake_up(["tag1"]))
        asyncio.run(server_instance.wake_up(tags=None))

        assert ("wake_up", ["tag1"]) in engine.calls
        assert ("wake_up", None) in engine.calls

        with pytest.raises(ValueError, match="tags must be a list"):
            asyncio.run(server_instance.wake_up(tags="not-a-list"))

    def test_wake_up_missing_engine(self, server_instance):
        server_instance.engine = None

        with pytest.raises(RuntimeError, match="Engine not initialized"):
            asyncio.run(server_instance.wake_up())

    def test_sleep_success_and_errors(self, server_instance):
        class StubEngine:
            def __init__(self):
                self.slept = False
                self.reset = False

            async def sleep(self):
                self.slept = True

            async def reset_prefix_cache(self):
                self.reset = True

        engine = StubEngine()
        server_instance.engine = engine

        asyncio.run(server_instance.sleep())
        assert engine.slept is True
        assert engine.reset is True

        class ValueErrorSleepEngine:
            def __init__(self):
                self.reset = False

            async def reset_prefix_cache(self):
                self.reset = True

            async def sleep(self):
                raise ValueError("sleep error")

        server_instance.engine = ValueErrorSleepEngine()
        with pytest.raises(RuntimeError, match="Failed to put engine to sleep"):
            asyncio.run(server_instance.sleep())
        assert server_instance.engine.reset is True

        class ValueErrorResetEngine:
            async def reset_prefix_cache(self):
                raise ValueError("reset error")

        server_instance.engine = ValueErrorResetEngine()
        with pytest.raises(RuntimeError, match="Failed to put engine to sleep"):
            asyncio.run(server_instance.sleep())

        class GenericErrorEngine:
            def __init__(self):
                self.reset = False

            async def reset_prefix_cache(self):
                self.reset = True

            async def sleep(self):
                raise RuntimeError("boom")

        server_instance.engine = GenericErrorEngine()
        with pytest.raises(RuntimeError, match="Unexpected error occurred when putting engine to sleep"):
            asyncio.run(server_instance.sleep())
        assert server_instance.engine.reset is True

    def test_sleep_missing_engine(self, server_instance):
        server_instance.engine = None

        with pytest.raises(RuntimeError, match="Engine not initialized"):
            asyncio.run(server_instance.sleep())


class TestInitValidation:
    """Minimal tests for __init__ parameter validation."""

    def test_init_invalid_agentic_rl_config(self, vllm_async_server_mod, valid_configs):
        gen_cfg, _ = valid_configs
        AsyncVLLMServer = vllm_async_server_mod.AsyncVLLMServer

        with pytest.raises(ValueError, match="agentic_rl_config must be a AgenticRLConfig"):
            AsyncVLLMServer(
                gen_cfg,
                object(),  # invalid config
                "/path/to/tokenizer",
                vllm_dp_size=1,
                vllm_dp_rank=0,
                wg_prefix="wg",
            )

    @pytest.mark.parametrize(
        "field, value, msg",
        [
            ("vllm_dp_size", "1", "vllm_dp_size must be an integer"),
            ("vllm_dp_rank", "0", "vllm_dp_rank must be an integer"),
            ("wg_prefix", 123, "wg_prefix must be a string"),
        ],
    )
    def test_init_invalid_numeric_and_string_params(self, vllm_async_server_mod, valid_configs, field, value, msg):
        gen_cfg, agent_cfg = valid_configs
        AsyncVLLMServer = vllm_async_server_mod.AsyncVLLMServer

        kwargs = {
            "config": gen_cfg,
            "agentic_rl_config": agent_cfg,
            "tokenizer_name_or_path": "/path/to/tokenizer",
            "vllm_dp_size": 1,
            "vllm_dp_rank": 0,
            "wg_prefix": "wg",
        }
        kwargs[field] = value

        with pytest.raises(ValueError, match=msg):
            AsyncVLLMServer(**kwargs)
