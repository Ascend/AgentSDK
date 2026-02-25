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

import importlib
import sys
from types import ModuleType
from unittest.mock import patch
import pytest
from agentic_rl.runner.infer_adapter.vllm.base_inference_engine import BaseInferEngine
 
 
class ConcreteInferEngine(BaseInferEngine):
    def init_cache_engine(self):
        pass
 
    def free_cache_engine(self):
        pass
 
    def offload_model_weights(self):
        pass
 
    def sync_model_weights(self, params, load_format='megatron'):
        pass
 
    def generate_sequences(self, prompts=None, sampling_params=None, prompt_token_ids=None, use_tqdm=None, **kwargs):
        pass


@pytest.fixture
def BaseInferEngine(monkeypatch):
    """Provide isolated BaseInferEngine by injecting minimal parent class."""
    # Build minimal parent chain
    parent_pkg = ModuleType('mindspeed_rl')
    models_pkg = ModuleType('mindspeed_rl.models')
    base_pkg = ModuleType('mindspeed_rl.models.base')
    base_module = ModuleType('mindspeed_rl.models.base.base_inference_engine')
    
    # Mark as packages
    parent_pkg.__path__ = []
    models_pkg.__path__ = []
    base_pkg.__path__ = []
    
    class _MockBaseInferEngine:
        def __init__(self, **kwargs):
            pass
    
    base_module.BaseInferEngine = _MockBaseInferEngine
    
    # Install parent packages
    monkeypatch.setitem(sys.modules, 'mindspeed_rl', parent_pkg)
    monkeypatch.setitem(sys.modules, 'mindspeed_rl.models', models_pkg)
    monkeypatch.setitem(sys.modules, 'mindspeed_rl.models.base', base_pkg)
    monkeypatch.setitem(sys.modules, 'mindspeed_rl.models.base.base_inference_engine', base_module)
    
    # Stub rollout packages
    rollout_pkg = ModuleType('mindspeed_rl.models.rollout')
    rollout_pkg.__path__ = []
    vllm_adapter_mod = ModuleType('mindspeed_rl.models.rollout.vllm_adapter')
    vllm_adapter_mod.__path__ = []
    vllm_parallel_state_mod = ModuleType('mindspeed_rl.models.rollout.vllm_adapter.vllm_parallel_state')
    vllm_parallel_state_mod.initialize_parallel_state = lambda **kwargs: None
    
    monkeypatch.setitem(sys.modules, 'mindspeed_rl.models.rollout', rollout_pkg)
    monkeypatch.setitem(sys.modules, 'mindspeed_rl.models.rollout.vllm_adapter', vllm_adapter_mod)
    monkeypatch.setitem(sys.modules, 'mindspeed_rl.models.rollout.vllm_adapter.vllm_parallel_state', vllm_parallel_state_mod)
    
    # Stub vllm_ascend packages
    vllm_ascend_pkg = ModuleType('vllm_ascend')
    vllm_ascend_pkg.__path__ = []
    patch_pkg = ModuleType('vllm_ascend.patch')
    patch_pkg.__path__ = []
    
    monkeypatch.setitem(sys.modules, 'vllm_ascend', vllm_ascend_pkg)
    monkeypatch.setitem(sys.modules, 'vllm_ascend.patch', patch_pkg)
    monkeypatch.setitem(sys.modules, 'vllm_ascend.patch.platform', ModuleType('vllm_ascend.patch.platform'))
    monkeypatch.setitem(sys.modules, 'vllm_ascend.patch.worker', ModuleType('vllm_ascend.patch.worker'))
    
    # Stub vllm.worker.worker_base
    vllm_pkg = ModuleType('vllm')
    vllm_pkg.__path__ = []
    worker_pkg = ModuleType('vllm.worker')
    worker_pkg.__path__ = []
    worker_base_mod = ModuleType('vllm.worker.worker_base')
    
    class _WorkerWrapperBase:
        def __init__(self, *args, **kwargs):
            self.worker = type('W', (), {
                'model_runner': type('MR', (), {'get_model': lambda self: object()})()
            })()
        
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
    
    worker_base_mod.WorkerWrapperBase = _WorkerWrapperBase
    
    monkeypatch.setitem(sys.modules, 'vllm', vllm_pkg)
    monkeypatch.setitem(sys.modules, 'vllm.worker', worker_pkg)
    monkeypatch.setitem(sys.modules, 'vllm.worker.worker_base', worker_base_mod)
    
    # Stub agentic_rl vllm submodules
    stub_worker = ModuleType('agentic_rl.runner.infer_adapter.vllm.vllm_worker')
    stub_worker.AsyncVLLMInferEngine = type('AsyncVLLMInferEngine', (), {})
    
    stub_async_server = ModuleType('agentic_rl.runner.infer_adapter.vllm.vllm_async_server')
    stub_async_server.AsyncVLLMServer = type('AsyncVLLMServer', (), {})
    
    monkeypatch.setitem(sys.modules, 'agentic_rl.runner.infer_adapter.vllm.vllm_worker', stub_worker)
    monkeypatch.setitem(sys.modules, 'agentic_rl.runner.infer_adapter.vllm.vllm_async_server', stub_async_server)
    
    # Clean import
    monkeypatch.delitem(sys.modules, 'agentic_rl.runner.infer_adapter.vllm.base_inference_engine', raising=False)
    module = importlib.import_module('agentic_rl.runner.infer_adapter.vllm.base_inference_engine')
    
    return module.BaseInferEngine


@pytest.fixture
def valid_params():
    """Fixture providing valid initialization parameters."""
    return {
        "tokenizer_name_or_path": "/valid/tokenizer/path",
        "train_tensor_parallel_size": 4,
        "train_pipeline_parallel_size": 2,
        "prompt_type": "test_prompt",
        "prompt_type_path": "/valid/prompt/path",
        "train_expert_parallel_size": 1,
        "train_context_parallel_size": 1,
        "infer_tensor_parallel_size": 8,
        "infer_pipeline_parallel_size": 1,
        "infer_expert_parallel_size": 1,
        "max_num_seqs": 1,
        "max_model_len": 2048,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.5,
        "trust_remote_code": True,
        "enable_expert_parallel": False,
        "infer_backend": "vllm"
    }


class TestBaseInferEngine:
    """Test suite for BaseInferEngine class."""

    @pytest.mark.parametrize("invalid_path,error_match", [
        ("tokenizer_name_or_path", "Tokenizer name or path.*is not valid"),
        ("prompt_type_path", "Prompt type path.*is not valid"),
    ])
    @patch('agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_data_path_is_valid')
    def test_invalid_paths(
        self, mock_file_check, valid_params, invalid_path, error_match
    ):
        """Test validation errors for invalid paths."""
        def side_effect(path):
            if path == valid_params[invalid_path]:
                raise ValueError(error_match)
        
        mock_file_check.side_effect = side_effect
        
        with pytest.raises(ValueError, match=error_match):
            ConcreteInferEngine(**valid_params)

    @pytest.mark.parametrize("param_name,invalid_value,error_match", [
        ("train_tensor_parallel_size", -1, "must be a positive integer"),
        ("train_tensor_parallel_size", 0, "must be a positive integer"),
        ("train_pipeline_parallel_size", -1, "must be a positive integer"),
        ("train_expert_parallel_size", 0, "must be a positive integer"),
        ("max_num_seqs", "10", "must be a positive integer"),
        ("max_model_len", 128 * 1024 + 1, "must be a positive integer and less than or equal to"),
        ("dtype", "float64", "must be a string and one of"),
        ("gpu_memory_utilization", 0.0, "must be a float"),
        ("gpu_memory_utilization", 1.1, "must be a float"),
        ("gpu_memory_utilization", 1, "must be a float"),
        ("infer_backend", "pytorch", "must be a string and one of"),
        ("trust_remote_code", "true", "must be a boolean"),
    ])
    @patch('agentic_rl.runner.infer_adapter.vllm.base_inference_engine.FileCheck.check_path_is_exist_and_valid')
    def test_parameter_validation(
        self, mock_file_check, valid_params, 
        param_name, invalid_value, error_match
    ):
        """Test validation errors for invalid parameter values."""
        mock_file_check.return_value = True
        valid_params[param_name] = invalid_value
        
        with pytest.raises(ValueError, match=error_match):
            ConcreteInferEngine(**valid_params)
