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
from types import ModuleType
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture(autouse=True, scope="function")
def mock_vllm_environment(monkeypatch):
    """Fixture to mock all vLLM and related dependencies."""
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


class TestAsyncVLLMInferEngineInitialization:
    """Test suite for AsyncVLLMInferEngine initialization."""
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_init_default_params(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test initialization with default parameters."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        
        assert engine.enable_sleep_mode is True
        assert engine.is_sleep is True
        assert engine.first_wake_up is True
        assert engine.model is None
        assert engine.inference_engine is None
        mock_cache_mgr_cls.assert_called_once()
        mock_mem_mgr_cls.assert_called_once()
        mock_weight_mgr_cls.assert_called_once()
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_init_custom_load_format(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test initialization with custom load_format."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(
            enable_sleep_mode=False,
            load_format="custom_format",
            **base_init_params
        )
        
        assert engine.enable_sleep_mode is False
        mock_weight_mgr_cls.assert_called_once_with(
            infer_tensor_parallel_size=8,
            infer_pipeline_parallel_size=1,
            infer_expert_parallel_size=1,
            load_format="custom_format"
        )
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    @pytest.mark.parametrize("invalid_param,error_match", [
        ({"enable_sleep_mode": "true"}, "enable_sleep_mode must be a boolean"),
        ({"enable_sleep_mode": 1}, "enable_sleep_mode must be a boolean"),
        ({"enable_sleep_mode": True, "load_format": 123}, "load_format must be a string"),
        ({"enable_sleep_mode": True, "load_format": None}, "load_format must be a string"),
    ])
    def test_init_invalid_params(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params,
        invalid_param, error_match
    ):
        """Test initialization with invalid parameters."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        params = {**base_init_params}
        enable_sleep_mode = invalid_param.get("enable_sleep_mode", True)
        load_format = invalid_param.get("load_format", "megatron")
        with pytest.raises(ValueError, match=error_match):
            AsyncVLLMInferEngine(enable_sleep_mode=enable_sleep_mode, load_format=load_format, **params)


class TestAsyncVLLMInferEngineWorkerInit:
    """Test suite for AsyncVLLMInferEngine.init_worker method."""
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.get_local_rank")
    def test_init_worker_success(
        self, mock_get_rank, mock_weight_mgr_cls, mock_mem_mgr_cls,
        mock_cache_mgr_cls, mock_file_check, mock_vllm_environment,
        base_init_params, monkeypatch
    ):
        """Test successful worker initialization."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        mock_get_rank.return_value = 0
        
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        
        all_kwargs = [{"vllm_config": MagicMock()}]
        engine.init_worker(all_kwargs)
        
        assert engine.inference_engine is not None
        assert engine.hf_config is not None


class TestAsyncVLLMInferEngineLoadModel:
    """Test suite for AsyncVLLMInferEngine.load_model method."""
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_load_model_success(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test successful model loading."""
        mock_file_check.return_value = None
        mock_cache_mgr = MagicMock()
        mock_mem_mgr = MagicMock()
        mock_mem_mgr.create_cpu_model_copy.return_value = {}
        mock_cache_mgr_cls.return_value = mock_cache_mgr
        mock_mem_mgr_cls.return_value = mock_mem_mgr
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = MagicMock()
        
        engine.load_model()
        
        assert engine.model is not None
        assert engine.cache_manager.kv_cache_configs is None
        mock_mem_mgr.create_cpu_model_copy.assert_called_once()
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    @pytest.mark.parametrize("error_type,error_msg,expected_match", [
        (RuntimeError, "Model load failed", "Model loading failed"),
        (MemoryError, "Out of memory", "Model loading failed"),
        (OSError, "File not found", "Model loading failed"),
        (IOError, "IO error", "Model loading failed"),
    ])
    def test_load_model_inference_engine_errors(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params,
        error_type, error_msg, expected_match
    ):
        """Test load_model with inference engine loading failures."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = MagicMock()
        engine.inference_engine.load_model.side_effect = error_type(error_msg)
        
        with pytest.raises(RuntimeError, match=expected_match):
            engine.load_model()


class TestAsyncVLLMInferEngineExecuteMethod:
    """Test suite for AsyncVLLMInferEngine.execute_method."""
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    @pytest.mark.parametrize("method_name", [
        "init_worker",
        "load_model",
        "sleep",
        "wake_up",
    ])
    def test_execute_method_internal_dispatch(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params, method_name
    ):
        """Test execute_method dispatches to internal methods."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        
        # Mock the internal method
        with patch.object(engine, method_name) as mock_method:
            engine.execute_method(method_name, "arg1", kwarg1="value1")
            mock_method.assert_called_once_with("arg1", kwarg1="value1")
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_execute_method_bytes_method_name(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test execute_method with bytes method name."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        
        with patch.object(engine, "sleep") as mock_sleep:
            engine.execute_method(b"sleep")
            mock_sleep.assert_called_once()
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_execute_method_delegation_to_inference_engine(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test execute_method delegates unknown methods to inference_engine."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = MagicMock()
        engine.inference_engine.execute_method.return_value = "result"
        
        result = engine.execute_method("custom_method", "arg1")
        
        engine.inference_engine.execute_method.assert_called_once_with("custom_method", "arg1")
        assert result == "result"
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_execute_method_uninitialized_inference_engine(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test execute_method with uninitialized inference_engine."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = None
        
        with pytest.raises(RuntimeError, match="Inference engine is not initialized"):
            engine.execute_method("unknown_method")
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    @pytest.mark.parametrize("invalid_method,expected_match", [
        (123, "method must be a string or bytes"),
        (None, "method must be a string or bytes"),
        ([], "method must be a string or bytes"),
    ])
    def test_execute_method_invalid_params(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params,
        invalid_method, expected_match
    ):
        """Test execute_method with invalid method parameter."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        
        with pytest.raises(ValueError, match=expected_match):
            engine.execute_method(invalid_method)


class TestAsyncVLLMInferEngineCacheManagement:
    """Test suite for AsyncVLLMInferEngine cache management methods."""

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_init_cache_engine_success(
            self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
            mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test successful cache engine initialization."""
        mock_file_check.return_value = None
        mock_cache_mgr = MagicMock()
        mock_cache_mgr_cls.return_value = mock_cache_mgr
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()

        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = MagicMock()

        engine.init_cache_engine()

        mock_cache_mgr.init_cache_engine.assert_called_once_with(engine.inference_engine)

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_init_cache_engine_uninitialized_engine(
            self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
            mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test init_cache_engine with uninitialized inference_engine."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()

        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = None

        with pytest.raises(RuntimeError, match="Inference engine is not initialized"):
            engine.init_cache_engine()

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_free_cache_engine_success(
            self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
            mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test successful cache engine freeing."""
        mock_file_check.return_value = None
        mock_cache_mgr = MagicMock()
        mock_cache_mgr_cls.return_value = mock_cache_mgr
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()

        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = MagicMock()
        engine.model = MagicMock()

        engine.free_cache_engine()

        mock_cache_mgr.free_cache_engine.assert_called_once_with(engine.inference_engine, engine.model)

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_free_cache_engine_model_none(
            self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
            mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test free_cache_engine with model=None."""
        mock_file_check.return_value = None
        mock_cache_mgr = MagicMock()
        mock_cache_mgr_cls.return_value = mock_cache_mgr
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()

        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = MagicMock()
        engine.model = None

        # Should return early without error
        engine.free_cache_engine()

        mock_cache_mgr.free_cache_engine.assert_not_called()


class TestAsyncVLLMInferEngineWeightManagement:
    """Test suite for AsyncVLLMInferEngine weight management methods."""

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_offload_model_weights_success(
            self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
            mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test successful model weight offloading."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr = MagicMock()
        mock_mem_mgr.cpu_model = {}
        mock_mem_mgr_cls.return_value = mock_mem_mgr
        mock_weight_mgr_cls.return_value = MagicMock()

        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.model = MagicMock()

        engine.offload_model_weights()

        mock_mem_mgr.offload_model_weights.assert_called_once_with(engine.model, mock_mem_mgr.cpu_model)

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    @pytest.mark.parametrize("missing_attr,expected_match", [
        ("model", "Model is not initialized"),
        ("cpu_model", "CPU model is not initialized"),
    ])
    def test_offload_model_weights_not_initialized(
            self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
            mock_file_check, mock_vllm_environment, base_init_params,
            missing_attr, expected_match
    ):
        """Test offload_model_weights with uninitialized model or CPU model."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr = MagicMock()
        mock_mem_mgr.cpu_model = {}
        mock_mem_mgr_cls.return_value = mock_mem_mgr
        mock_weight_mgr_cls.return_value = MagicMock()

        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)

        if missing_attr == "model":
            engine.model = None
        else:
            engine.model = MagicMock()
            engine.memory_manager.cpu_model = None

        with pytest.raises(RuntimeError, match=expected_match):
            engine.offload_model_weights()

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_sync_model_weights_success(
            self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
            mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test successful model weight synchronization."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr = MagicMock()
        mock_weight_mgr_cls.return_value = mock_weight_mgr

        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.model = MagicMock()
        engine.hf_config = MagicMock()

        params = {"param1": "value1"}
        engine.sync_model_weights(params, load_format="megatron")

        mock_weight_mgr.load_megatron_weights.assert_called_once_with(
            params, engine.model, engine.hf_config
        )

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    @pytest.mark.parametrize("invalid_param,value,expected_match", [
        ("params", "not_a_dict", "params must be a dictionary"),
        ("params", None, "params must be a dictionary"),
        ("load_format", 123, "load_format must be a string"),
        ("load_format", None, "load_format must be a string"),
    ])
    def test_sync_model_weights_invalid_params(
            self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
            mock_file_check, mock_vllm_environment, base_init_params,
            invalid_param, value, expected_match
    ):
        """Test sync_model_weights with invalid parameters."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()

        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.model = MagicMock()
        engine.hf_config = MagicMock()

        kwargs = {"params": {"valid": "dict"}, "load_format": "megatron", invalid_param: value}

        with pytest.raises(ValueError, match=expected_match):
            engine.sync_model_weights(**kwargs)

    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    @pytest.mark.parametrize("missing_attr,expected_match", [
        ("model", "Model is not initialized"),
        ("hf_config", "HuggingFace config is not initialized"),
    ])
    def test_sync_model_weights_not_initialized(
            self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
            mock_file_check, mock_vllm_environment, base_init_params,
            missing_attr, expected_match
    ):
        """Test sync_model_weights with uninitialized model or hf_config."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()

        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)

        if missing_attr == "model":
            engine.model = None
            engine.hf_config = MagicMock()
        else:
            engine.model = MagicMock()
            engine.hf_config = None

        with pytest.raises(RuntimeError, match=expected_match):
            engine.sync_model_weights({"param": "value"})


class TestAsyncVLLMInferEngineSleepWake:
    """Test suite for AsyncVLLMInferEngine sleep and wake_up methods."""
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_sleep_with_sleep_mode_enabled(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test sleep with enable_sleep_mode=True."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr = MagicMock()
        mock_mem_mgr_cls.return_value = mock_mem_mgr
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = MagicMock()
        
        engine.sleep()
        
        engine.inference_engine.sleep.assert_called_once_with(level=2)
        mock_mem_mgr.clear_gpu_memory.assert_called_once()
        assert engine.is_sleep is True
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_sleep_with_sleep_mode_disabled(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test sleep with enable_sleep_mode=False."""
        mock_file_check.return_value = None
        mock_cache_mgr = MagicMock()
        mock_cache_mgr_cls.return_value = mock_cache_mgr
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=False, **base_init_params)
        engine.inference_engine = MagicMock()
        engine.model = MagicMock()
        
        engine.sleep()
        
        mock_cache_mgr.free_cache_engine.assert_called_once()
        assert engine.is_sleep is True
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_sleep_inference_engine_failure(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test sleep with inference engine sleep failure."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = MagicMock()
        engine.inference_engine.sleep.side_effect = RuntimeError("Sleep failed")
        
        with pytest.raises(RuntimeError, match="Inference engine sleep failed"):
            engine.sleep()
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_wake_up_with_sleep_mode_enabled(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test wake_up with enable_sleep_mode=True."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = MagicMock()
        
        engine.wake_up()
        
        engine.inference_engine.wake_up.assert_called_once_with(tags=["kv_cache"])
        assert engine.is_sleep is False
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_wake_up_first_time_sleep_mode_disabled(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test first wake_up with enable_sleep_mode=False."""
        mock_file_check.return_value = None
        mock_cache_mgr = MagicMock()
        mock_cache_mgr_cls.return_value = mock_cache_mgr
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=False, **base_init_params)
        engine.inference_engine = MagicMock()
        engine.model = MagicMock()
        engine.first_wake_up = True
        
        engine.wake_up()
        
        # First wake up should call free_cache_engine, initialize KV caches, and init_cache_engine
        mock_cache_mgr.free_cache_engine.assert_called_once()
        mock_cache_mgr.initialize_kv_caches.assert_called_once()
        mock_cache_mgr.init_cache_engine.assert_called_once()
        assert engine.first_wake_up is False
        assert engine.is_sleep is False
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_wake_up_subsequent_sleep_mode_disabled(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test subsequent wake_up with enable_sleep_mode=False."""
        mock_file_check.return_value = None
        mock_cache_mgr = MagicMock()
        mock_cache_mgr_cls.return_value = mock_cache_mgr
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=False, **base_init_params)
        engine.inference_engine = MagicMock()
        engine.model = MagicMock()
        engine.first_wake_up = False
        
        engine.wake_up()
        
        # Subsequent wake-ups should only call init_cache_engine
        mock_cache_mgr.initialize_kv_caches.assert_not_called()
        mock_cache_mgr.init_cache_engine.assert_called_once()
        assert engine.is_sleep is False


class TestAsyncVLLMInferEnginePrivateHelpers:
    """Test suite for AsyncVLLMInferEngine private helper methods."""
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_setup_worker_config_with_env_vars(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params, monkeypatch
    ):
        """Test _setup_worker_config with RANK and LOCAL_RANK environment variables."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("LOCAL_RANK", "1")
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        
        all_kwargs = [{"vllm_config": MagicMock()}]
        engine._setup_worker_config(all_kwargs)
        
        assert all_kwargs[0]["rank"] == 2
        assert all_kwargs[0]["local_rank"] == 1
        assert engine.vllm_config is not None
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_setup_worker_config_missing_vllm_config(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test _setup_worker_config with missing vllm_config."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        
        all_kwargs = [{}]
        
        with pytest.raises(RuntimeError, match="Missing or invalid vllm_config"):
            engine._setup_worker_config(all_kwargs)
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.get_local_rank")
    def test_setup_tokenizer_and_config_success(
        self, mock_get_rank, mock_weight_mgr_cls, mock_mem_mgr_cls,
        mock_cache_mgr_cls, mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test _setup_tokenizer_and_config successful execution."""
        mock_file_check.return_value = None
        mock_cache_mgr_cls.return_value = MagicMock()
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        mock_get_rank.return_value = 0
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        
        engine._setup_tokenizer_and_config()
        
        assert engine.hf_config is not None
        assert engine.local_rank == 0
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_initialize_kv_caches_success(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test _initialize_kv_caches successful execution."""
        mock_file_check.return_value = None
        mock_cache_mgr = MagicMock()
        mock_cache_mgr_cls.return_value = mock_cache_mgr
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = MagicMock()
        
        vllm_config = MagicMock()
        engine._initialize_kv_caches(vllm_config)
        
        mock_cache_mgr.initialize_kv_caches.assert_called_once_with(
            engine.inference_engine, vllm_config
        )
    
    @patch("agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.CacheManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.MemoryManager")
    @patch("agentic_rl.runner.infer_adapter.vllm.vllm_worker.WeightManager")
    def test_initialize_kv_caches_with_none_config(
        self, mock_weight_mgr_cls, mock_mem_mgr_cls, mock_cache_mgr_cls,
        mock_file_check, mock_vllm_environment, base_init_params
    ):
        """Test _initialize_kv_caches with None vllm_config."""
        mock_file_check.return_value = None
        mock_cache_mgr = MagicMock()
        mock_cache_mgr.initialize_kv_caches.side_effect = ValueError("vllm_config cannot be None")
        mock_cache_mgr_cls.return_value = mock_cache_mgr
        mock_mem_mgr_cls.return_value = MagicMock()
        mock_weight_mgr_cls.return_value = MagicMock()
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncVLLMInferEngine
        
        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = MagicMock()
        
        with pytest.raises(ValueError):
            engine._initialize_kv_caches(None)
