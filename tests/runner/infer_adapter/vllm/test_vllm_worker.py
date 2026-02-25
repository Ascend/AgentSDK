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

import sys
from typing import Any, Dict, List, Union
from types import ModuleType
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture(autouse=True, scope="function")
def mock_vllm_environment(monkeypatch):
    """Fixture to mock all vLLM and related dependencies."""
    has_torch = True
    try:
        import torch
    except ImportError:
        has_torch = False

    if has_torch:
        if hasattr(torch.jit, "script"):
            original_jit_script = torch.jit.script
        else:
            original_jit_script = None

        if hasattr(torch, "compile"):
            original_compile = torch.compile
        else:
            original_compile = None
    else:
        original_jit_script = None
        original_compile = None

    # Mock vLLM modules
    vllm_mod = ModuleType("vllm")
    vllm_mod.__path__ = []

    # Mock vllm.worker.worker_base
    worker_pkg = ModuleType("vllm.worker")
    worker_pkg.__path__ = []
    worker_base_mod = ModuleType("vllm.worker.worker_base")

    class MockWorkerWrapperBase:
        def __init__(self, *args, **kwargs):
            self.worker = MagicMock()
            self.worker.model_runner.get_model.return_value = MagicMock()
            self.worker.model_runner.kv_caches = []
            self.worker.vllm_config = MagicMock()

        def init_worker(self, *args, **kwargs):
            pass

        def load_model(self, *args, **kwargs):
            pass

        def execute_method(self, *args, **kwargs):
            return None

        def sleep(self, *args, **kwargs):
            pass

        def wake_up(self, *args, **kwargs):
            pass

        def get_kv_cache_spec(self):
            return MagicMock()

        def determine_available_memory(self):
            return 1024 * 1024 * 1024

    worker_base_mod.WorkerWrapperBase = MockWorkerWrapperBase

    # Mock vllm.model_executor
    model_executor_pkg = ModuleType("vllm.model_executor")
    model_executor_pkg.__path__ = []

    # Mock vllm.model_executor.layers
    layers_pkg = ModuleType("vllm.model_executor.layers")
    layers_pkg.__path__ = []

    # Mock vllm.model_executor.layers.fused_moe
    fused_moe_pkg = ModuleType("vllm.model_executor.layers.fused_moe")
    fused_moe_pkg.__path__ = []
    fused_moe_layer_mod = ModuleType("vllm.model_executor.layers.fused_moe.layer")
    fused_moe_layer_mod.FusedMoE = MagicMock()

    # Mock vllm.model_executor.models
    models_mod = ModuleType("vllm.model_executor.models")
    models_mod.ModelRegistry = MagicMock()

    # Mock vllm.model_executor.layers.linear
    linear_mod = ModuleType("vllm.model_executor.layers.linear")
    linear_mod.ColumnParallelLinear = MagicMock()
    linear_mod.MergedColumnParallelLinear = MagicMock()
    linear_mod.QKVParallelLinear = MagicMock()
    linear_mod.RowParallelLinear = MagicMock()
    linear_mod.ReplicatedLinear = MagicMock()

    # Mock vllm.model_executor.layers.vocab_parallel_embedding
    vocab_mod = ModuleType("vllm.model_executor.layers.vocab_parallel_embedding")
    vocab_mod.ParallelLMHead = MagicMock()
    vocab_mod.VocabParallelEmbedding = MagicMock()

    # Mock vllm_ascend
    vllm_ascend_pkg = ModuleType("vllm_ascend")
    vllm_ascend_pkg.__path__ = []
    patch_pkg = ModuleType("vllm_ascend.patch")
    patch_pkg.__path__ = []
    platform_mod = ModuleType("vllm_ascend.patch.platform")
    worker_mod = ModuleType("vllm_ascend.patch.worker")

    # Mock mindspeed_rl
    mindspeed_rl_pkg = ModuleType("mindspeed_rl")
    mindspeed_rl_pkg.__path__ = []
    models_pkg = ModuleType("mindspeed_rl.models")
    models_pkg.__path__ = []
    rollout_pkg = ModuleType("mindspeed_rl.models.rollout")
    rollout_pkg.__path__ = []
    vllm_adapter_pkg = ModuleType("mindspeed_rl.models.rollout.vllm_adapter")
    vllm_adapter_pkg.__path__ = []
    parallel_state_mod = ModuleType("mindspeed_rl.models.rollout.vllm_adapter.vllm_parallel_state")

    def mock_initialize_parallel_state(**kwargs):
        pass

    parallel_state_mod.initialize_parallel_state = mock_initialize_parallel_state

    # Mock torch_npu
    torch_npu_mod = ModuleType("torch_npu")
    torch_npu_mod._autoload = MagicMock()
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_mod)

    # Mock additional vllm_ascend modules
    vllm_ascend_worker_mod = ModuleType("vllm_ascend.worker")
    vllm_ascend_worker_mod.__path__ = []
    vllm_ascend_attention_mod = ModuleType("vllm_ascend.attention")
    vllm_ascend_attention_mod.__path__ = []
    vllm_ascend_attention_attention_mod = ModuleType("vllm_ascend.attention.attention")
    vllm_ascend_attention_attention_mod.AscendMetadata = MagicMock()
    vllm_ascend_worker_draft_mod = ModuleType("vllm_ascend.worker.draft_model_runner")
    vllm_ascend_worker_draft_mod.TP1DraftModelRunner = MagicMock()
    vllm_ascend_patch_platform_mod = ModuleType("vllm_ascend.patch.platform.patch_0_9_1")
    vllm_ascend_patch_worker_common_mod = ModuleType("vllm_ascend.patch.worker.patch_common")
    vllm_ascend_patch_worker_common_mod.__path__ = []
    vllm_ascend_patch_worker_common_utils_mod = ModuleType("vllm_ascend.patch.worker.patch_common.patch_utils")
    vllm_ascend_patch_worker_common_multi_step_mod = ModuleType(
        "vllm_ascend.patch.worker.patch_common.patch_multi_step_worker"
    )

    # Mock transformers
    transformers_mod = ModuleType("transformers")

    class MockAutoConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            config = MagicMock()
            config.hidden_size = 768
            return config

    transformers_mod.AutoConfig = MockAutoConfig

    # Install all mocks
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.worker", worker_pkg)
    monkeypatch.setitem(sys.modules, "vllm.worker.worker_base", worker_base_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor_pkg)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", layers_pkg)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe", fused_moe_pkg)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe.layer", fused_moe_layer_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", models_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", linear_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.vocab_parallel_embedding", vocab_mod)
    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend_pkg)
    monkeypatch.setitem(sys.modules, "vllm_ascend.patch", patch_pkg)
    monkeypatch.setitem(sys.modules, "vllm_ascend.patch.platform", platform_mod)
    monkeypatch.setitem(sys.modules, "vllm_ascend.patch.platform.patch_0_9_1", vllm_ascend_patch_platform_mod)
    monkeypatch.setitem(sys.modules, "vllm_ascend.patch.worker", worker_mod)
    monkeypatch.setitem(sys.modules, "vllm_ascend.patch.worker.patch_common", vllm_ascend_patch_worker_common_mod)
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.patch.worker.patch_common.patch_utils", vllm_ascend_patch_worker_common_utils_mod
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.patch.worker.patch_common.patch_multi_step_worker",
        vllm_ascend_patch_worker_common_multi_step_mod,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker", vllm_ascend_worker_mod)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.draft_model_runner", vllm_ascend_worker_draft_mod)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention", vllm_ascend_attention_mod)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.attention", vllm_ascend_attention_attention_mod)
    monkeypatch.setitem(sys.modules, "mindspeed_rl", mindspeed_rl_pkg)
    monkeypatch.setitem(sys.modules, "mindspeed_rl.models", models_pkg)
    monkeypatch.setitem(sys.modules, "mindspeed_rl.models.rollout", rollout_pkg)
    monkeypatch.setitem(sys.modules, "mindspeed_rl.models.rollout.vllm_adapter", vllm_adapter_pkg)
    monkeypatch.setitem(sys.modules, "mindspeed_rl.models.rollout.vllm_adapter.vllm_parallel_state", parallel_state_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    # Mock base inference engine
    base_pkg = ModuleType("mindspeed_rl.models.base")
    base_pkg.__path__ = []
    base_engine_mod = ModuleType("mindspeed_rl.models.base.base_inference_engine")

    class MockBaseInferEngine:
        def __init__(self, **kwargs):
            # Store all kwargs as attributes
            for key, value in kwargs.items():
                setattr(self, key, value)

    base_engine_mod.BaseInferEngine = MockBaseInferEngine

    monkeypatch.setitem(sys.modules, "mindspeed_rl.models.base", base_pkg)
    monkeypatch.setitem(sys.modules, "mindspeed_rl.models.base.base_inference_engine", base_engine_mod)

    # Clean import of vllm_worker and related modules
    # Remove any previously imported agentic_rl modules that might have the old dependencies
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("agentic_rl.runner.infer_adapter.vllm"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    with patch.dict(sys.modules, {
        "transformers.configuration_utils": MagicMock(),
        "vllm.config": MagicMock()
    }):
        with patch("agentic_rl.runner.infer_adapter.vllm.patch.apply_patch"):
            yield

    if has_torch:
        if original_jit_script is not None:
            torch.jit.script = original_jit_script

        if original_compile is not None:
            torch.compile = original_compile


@pytest.fixture(scope="function")
def mock_async_base_vllminfer_engine():
    from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncBaseVLLMInferEngine

    class ConcreteVLLMInferEngine(AsyncBaseVLLMInferEngine):
        def __init__(self, enable_sleep_mode: bool, load_format: str = "megatron", *args, **kwargs):
            super().__init__(enable_sleep_mode, load_format, *args, **kwargs)

        def init_worker(self, all_kwargs: List[Dict[str, Any]]):
            pass
        
        def load_model(self, *args, **kwargs):
            pass

        def sleep(self, *args, **kwargs):
            pass

        def wake_up(self, *args, **kwargs):
            pass

        def init_cache_engine(self):
            pass

        def free_cache_engine(self):
            pass

        def offload_model_weights(self):
            pass

        def sync_model_weights(self, params, load_format='megatron'):
            pass

        def generate_sequences(self,
                            prompts=None,
                            sampling_params=None,
                            prompt_token_ids=None,
                            use_tqdm=None,
                            **kwargs):
            pass
    yield ConcreteVLLMInferEngine


@pytest.fixture
def base_init_params():
    """Fixture providing base initialization parameters for BaseInferEngine."""
    return {
        "tokenizer_name_or_path": "/valid/tokenizer/path",
        "train_tensor_parallel_size": 4,
        "train_pipeline_parallel_size": 2,
        "infer_tensor_parallel_size": 8,
        "infer_pipeline_parallel_size": 1,
    }


@pytest.fixture
def mock_managers(monkeypatch):
    """Fixture to mock CacheManager, MemoryManager, and WeightManager."""
    # Mock CacheManager
    cache_manager_mock = MagicMock()
    cache_manager_mock.kv_cache_configs = None

    # Mock MemoryManager
    memory_manager_mock = MagicMock()
    memory_manager_mock.cpu_model = None
    memory_manager_mock.create_cpu_model_copy.return_value = {}

    # Mock WeightManager
    weight_manager_mock = MagicMock()

    return {
        "cache_manager": cache_manager_mock,
        "memory_manager": memory_manager_mock,
        "weight_manager": weight_manager_mock,
    }


class TestAsyncBaseVLLMInferEngineInitialization:
    """Test suite for AsyncBaseVLLMInferEngine initialization."""

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    def test_init_default_params(
            self,
            mock_file_check, mock_vllm_environment, base_init_params, mock_async_base_vllminfer_engine
    ):
        """Test initialization with default parameters."""
        mock_file_check.return_value = None
        engine = mock_async_base_vllminfer_engine(enable_sleep_mode=True, load_format="megatron", **base_init_params)

        assert engine.enable_sleep_mode is True
        assert engine.is_sleep is True
        assert engine.first_wake_up is True
        assert engine.model is None
        assert engine.inference_engine is None

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    def test_init_custom_load_format(
            self, mock_file_check, mock_vllm_environment, base_init_params, mock_async_base_vllminfer_engine
    ):
        """Test initialization with custom load_format."""
        mock_file_check.return_value = None
        engine = mock_async_base_vllminfer_engine(
            enable_sleep_mode=False,
            load_format="custom_format",
            **base_init_params
        )

        assert engine.enable_sleep_mode is False

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @pytest.mark.parametrize("invalid_param,error_match", [
        ({"enable_sleep_mode": "true"}, "enable_sleep_mode must be a boolean"),
        ({"enable_sleep_mode": 1}, "enable_sleep_mode must be a boolean"),
        ({"enable_sleep_mode": True, "load_format": 123}, "load_format must be a string"),
        ({"enable_sleep_mode": True, "load_format": None}, "load_format must be a string"),
    ])
    def test_init_invalid_params(
            self, mock_file_check, mock_vllm_environment, base_init_params, mock_async_base_vllminfer_engine,
            invalid_param, error_match
    ):
        """Test initialization with invalid parameters."""
        mock_file_check.return_value = None

        params = {**base_init_params}
        enable_sleep_mode = invalid_param.get("enable_sleep_mode", True)
        load_format = invalid_param.get("load_format", "megatron")
        with pytest.raises(ValueError, match=error_match):
            mock_async_base_vllminfer_engine(enable_sleep_mode=enable_sleep_mode, load_format=load_format, **params)


class TestAsyncBaseVLLMInferEngineExecuteMethod:
    """Test suite for AsyncBaseVLLMInferEngine.execute_method."""

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @pytest.mark.parametrize("method_name", [
        "init_worker",
        "load_model",
        "sleep",
        "wake_up",
    ])
    def test_execute_method_internal_dispatch(
            self, mock_file_check, mock_vllm_environment,
            base_init_params, method_name, mock_async_base_vllminfer_engine
    ):
        """Test execute_method dispatches to internal methods."""
        mock_file_check.return_value = None
        engine = mock_async_base_vllminfer_engine(enable_sleep_mode=True, **base_init_params)

        # Mock the internal method
        with patch.object(engine, method_name) as mock_method:
            engine.execute_method(method_name, "arg1", kwarg1="value1")
            mock_method.assert_called_once_with("arg1", kwarg1="value1")

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    def test_execute_method_bytes_method_name(
            self, mock_file_check, mock_vllm_environment, base_init_params, mock_async_base_vllminfer_engine
    ):
        """Test execute_method with bytes method name."""
        mock_file_check.return_value = None
        engine = mock_async_base_vllminfer_engine(enable_sleep_mode=True, **base_init_params)

        with patch.object(engine, "sleep") as mock_sleep:
            engine.execute_method(b"sleep")
            mock_sleep.assert_called_once()

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    def test_execute_method_delegation_to_inference_engine(
            self, mock_file_check, mock_vllm_environment, base_init_params, mock_async_base_vllminfer_engine
    ):
        """Test execute_method delegates unknown methods to inference_engine."""
        mock_file_check.return_value = None

        engine = mock_async_base_vllminfer_engine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = MagicMock()
        engine.inference_engine.execute_method.return_value = "result"

        result = engine.execute_method("custom_method", "arg1")

        engine.inference_engine.execute_method.assert_called_once_with("custom_method", "arg1")
        assert result == "result"

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    def test_execute_method_uninitialized_inference_engine(
            self, mock_file_check, mock_vllm_environment, base_init_params, mock_async_base_vllminfer_engine
    ):
        """Test execute_method with uninitialized inference_engine."""
        mock_file_check.return_value = None

        engine = mock_async_base_vllminfer_engine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = None

        with pytest.raises(RuntimeError, match="Inference engine is not initialized"):
            engine.execute_method("unknown_method")

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @pytest.mark.parametrize("invalid_method,expected_match", [
        (123, "method must be a string or bytes"),
        (None, "method must be a string or bytes"),
        ([], "method must be a string or bytes"),
    ])
    def test_execute_method_invalid_params(
            self, mock_file_check, mock_vllm_environment, base_init_params,
            mock_async_base_vllminfer_engine, invalid_method, expected_match
    ):
        """Test execute_method with invalid method parameter."""
        mock_file_check.return_value = None

        engine = mock_async_base_vllminfer_engine(enable_sleep_mode=True, **base_init_params)

        with pytest.raises(ValueError, match=expected_match):
            engine.execute_method(invalid_method)


class TestAsyncBaseVLLMInferEnginePrivateHelpers:
    """Test suite for AsyncBaseVLLMInferEngine private helper methods."""

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    def test_setup_worker_config_with_env_vars(
            self, mock_file_check, mock_vllm_environment, 
            base_init_params, mock_async_base_vllminfer_engine, monkeypatch
    ):
        """Test _setup_worker_config with RANK and LOCAL_RANK environment variables."""
        mock_file_check.return_value = None

        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("LOCAL_RANK", "1")

        engine = mock_async_base_vllminfer_engine(enable_sleep_mode=True, **base_init_params)

        all_kwargs = [{"vllm_config": MagicMock()}]
        engine._setup_worker_config(all_kwargs)

        assert all_kwargs[0]["rank"] == 2
        assert all_kwargs[0]["local_rank"] == 1
        assert engine.vllm_config is not None

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    def test_setup_worker_config_missing_vllm_config(
            self, mock_file_check, mock_vllm_environment, base_init_params, mock_async_base_vllminfer_engine
    ):
        """Test _setup_worker_config with missing vllm_config."""
        mock_file_check.return_value = None

        engine = mock_async_base_vllminfer_engine(enable_sleep_mode=True, **base_init_params)
        all_kwargs = [{}]

        with pytest.raises(RuntimeError, match="Missing or invalid vllm_config"):
            engine._setup_worker_config(all_kwargs)