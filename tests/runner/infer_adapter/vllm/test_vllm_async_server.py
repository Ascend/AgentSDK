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
def server_instance(vllm_async_server_mod, valid_configs):
    """Construct a real AsyncVLLMServer instance with mocked deps."""
    gen_cfg, agent_cfg = valid_configs
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
