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
from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True, scope="function")
def mock_vllm_environment(monkeypatch):
    """Fixture to mock all vLLM and related dependencies."""
    # Mock vllm modules
    vllm_mod = ModuleType("vllm")
    vllm_mod.__path__ = []
    
    # Mock vllm.model_executor
    model_executor_pkg = ModuleType("vllm.model_executor")
    model_executor_pkg.__path__ = []
    
    # Mock vllm.model_executor.models
    models_pkg = ModuleType("vllm.model_executor.models")
    models_pkg.__path__ = []
    
    # Mock ModelRegistry
    class MockModelRegistry:
        @staticmethod
        def get_supported_archs():
            return ["Qwen2ForCausalLM", "LlamaForCausalLM", "GPT2LMHeadModel"]
    
    models_pkg.ModelRegistry = MockModelRegistry
    
    # Mock vllm.model_executor.layers
    layers_pkg = ModuleType("vllm.model_executor.layers")
    layers_pkg.__path__ = []
    
    # Mock vllm.model_executor.layers.linear
    linear_mod = ModuleType("vllm.model_executor.layers.linear")
    
    class MockColumnParallelLinear:
        weight_loader = None
    
    class MockMergedColumnParallelLinear:
        weight_loader = None
    
    class MockQKVParallelLinear:
        weight_loader = None
    
    class MockRowParallelLinear:
        weight_loader = None
    
    class MockReplicatedLinear:
        weight_loader = None
    
    linear_mod.ColumnParallelLinear = MockColumnParallelLinear
    linear_mod.MergedColumnParallelLinear = MockMergedColumnParallelLinear
    linear_mod.QKVParallelLinear = MockQKVParallelLinear
    linear_mod.RowParallelLinear = MockRowParallelLinear
    linear_mod.ReplicatedLinear = MockReplicatedLinear
    
    # Mock vllm.model_executor.layers.vocab_parallel_embedding
    vocab_embedding_mod = ModuleType("vllm.model_executor.layers.vocab_parallel_embedding")
    
    class MockParallelLMHead:
        weight_loader = None
    
    class MockVocabParallelEmbedding:
        weight_loader = None
    
    vocab_embedding_mod.ParallelLMHead = MockParallelLMHead
    vocab_embedding_mod.VocabParallelEmbedding = MockVocabParallelEmbedding
    
    # Mock vllm.model_executor.layers.fused_moe
    fused_moe_pkg = ModuleType("vllm.model_executor.layers.fused_moe")
    fused_moe_pkg.__path__ = []
    fused_moe_layer_mod = ModuleType("vllm.model_executor.layers.fused_moe.layer")
    
    class MockFusedMoE:
        weight_loader = None
    
    fused_moe_layer_mod.FusedMoE = MockFusedMoE
    
    # Set up module hierarchy
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor_pkg)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", models_pkg)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", layers_pkg)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", linear_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.vocab_parallel_embedding", vocab_embedding_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe", fused_moe_pkg)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe.layer", fused_moe_layer_mod)
    
    yield {
        "vllm": vllm_mod,
        "ModelRegistry": MockModelRegistry,
        "ColumnParallelLinear": MockColumnParallelLinear,
        "MergedColumnParallelLinear": MockMergedColumnParallelLinear,
        "QKVParallelLinear": MockQKVParallelLinear,
        "RowParallelLinear": MockRowParallelLinear,
        "ReplicatedLinear": MockReplicatedLinear,
        "ParallelLMHead": MockParallelLMHead,
        "VocabParallelEmbedding": MockVocabParallelEmbedding,
        "FusedMoE": MockFusedMoE,
    }


class TestVllmMegatronWeightLoaders:
    """Test suite for VllmMegatronWeightLoaders class."""
    
    def test_init_success(self, mock_vllm_environment):
        """Test successful initialization with model loader registration."""
        from agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders import VllmMegatronWeightLoaders
        from agentic_rl.base.weight_loaders.megatron_weight_loaders import BaseMegatronWeightLoader
        
        # Create instance
        loader = VllmMegatronWeightLoaders()
        
        # Verify model loaders are registered
        assert "Qwen2ForCausalLM" in loader.model_megatron_weight_loader_registry
        assert "CustomQwen2ForCausalLM" in loader.model_megatron_weight_loader_registry
        assert "Qwen3ForCausalLM" in loader.model_megatron_weight_loader_registry
        
        # Verify all loaders reference the same qwen_megatron_weight_loader
        assert loader.model_megatron_weight_loader_registry["Qwen2ForCausalLM"] == \
               BaseMegatronWeightLoader.qwen_megatron_weight_loader
        assert loader.model_megatron_weight_loader_registry["CustomQwen2ForCausalLM"] == \
               BaseMegatronWeightLoader.qwen_megatron_weight_loader
        assert loader.model_megatron_weight_loader_registry["Qwen3ForCausalLM"] == \
               BaseMegatronWeightLoader.qwen_megatron_weight_loader
    
    def test_init_attribute_error(self, mock_vllm_environment):
        """Test initialization fails gracefully on AttributeError during registration."""
        from agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders import VllmMegatronWeightLoaders
        
        with patch.object(VllmMegatronWeightLoaders, 'register_model_loader', 
                         side_effect=AttributeError("Mock attribute error")):
            with pytest.raises(RuntimeError, match="Failed to register model loaders"):
                VllmMegatronWeightLoaders()
    
    def test_init_type_error(self, mock_vllm_environment):
        """Test initialization fails gracefully on TypeError during registration."""
        from agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders import VllmMegatronWeightLoaders
        
        with patch.object(VllmMegatronWeightLoaders, 'register_model_loader',
                         side_effect=TypeError("Mock type error")):
            with pytest.raises(RuntimeError, match="Failed to register model loaders"):
                VllmMegatronWeightLoaders()
    
    def test_get_supported_architectures_success(self, mock_vllm_environment):
        """Test get_supported_architectures returns correct architecture list."""
        from agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders import VllmMegatronWeightLoaders
        
        loader = VllmMegatronWeightLoaders()
        architectures = loader.get_supported_architectures()
        
        # Verify it returns the mocked list
        assert isinstance(architectures, list)
        assert "Qwen2ForCausalLM" in architectures
        assert "LlamaForCausalLM" in architectures
        assert "GPT2LMHeadModel" in architectures
    
    def test_get_supported_architectures_import_error(self, mock_vllm_environment, monkeypatch):
        """Test get_supported_architectures handles ImportError correctly."""
        from agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders import VllmMegatronWeightLoaders
        
        loader = VllmMegatronWeightLoaders()
        
        # Remove the mocked module to trigger ImportError
        monkeypatch.delitem(sys.modules, "vllm.model_executor.models")
        
        with pytest.raises(ImportError, match="Failed to import vllm.model_executor.models.ModelRegistry"):
            loader.get_supported_architectures()
    
    def test_get_supported_architectures_attribute_error(self, mock_vllm_environment):
        """Test get_supported_architectures handles AttributeError correctly."""
        from agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders import VllmMegatronWeightLoaders
        
        loader = VllmMegatronWeightLoaders()
        
        # Mock ModelRegistry.get_supported_archs to raise AttributeError
        with patch("vllm.model_executor.models.ModelRegistry.get_supported_archs", 
                   side_effect=AttributeError("Method not found")):
            with pytest.raises(AttributeError, match="ModelRegistry.get_supported_archs\\(\\) is not available"):
                loader.get_supported_architectures()
    
    def test_update_megatron_weight_loader_success(self, mock_vllm_environment):
        """Test update_megatron_weight_loader sets weight_loader for all layer classes."""
        from agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders import VllmMegatronWeightLoaders
        from agentic_rl.base.weight_loaders.megatron_weight_loaders import BaseMegatronWeightLoader
        
        # Import layer classes before calling update
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear,
            RowParallelLinear, ReplicatedLinear
        )
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
        
        # Verify initial state (all None)
        assert ColumnParallelLinear.weight_loader is None
        
        loader = VllmMegatronWeightLoaders()
        loader.update_megatron_weight_loader()
        
        # Verify weight_loader is set for all layer classes
        assert ColumnParallelLinear.weight_loader == BaseMegatronWeightLoader.parallel_weight_loader
        assert MergedColumnParallelLinear.weight_loader == BaseMegatronWeightLoader.parallel_weight_loader
        assert QKVParallelLinear.weight_loader == BaseMegatronWeightLoader.parallel_weight_loader
        assert RowParallelLinear.weight_loader == BaseMegatronWeightLoader.parallel_weight_loader
        assert ReplicatedLinear.weight_loader == BaseMegatronWeightLoader.parallel_weight_loader
        assert VocabParallelEmbedding.weight_loader == BaseMegatronWeightLoader.parallel_weight_loader
        assert ParallelLMHead.weight_loader == BaseMegatronWeightLoader.parallel_weight_loader
    
    def test_update_megatron_weight_loader_import_error(self, mock_vllm_environment, monkeypatch):
        """Test update_megatron_weight_loader handles ImportError correctly."""
        from agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders import VllmMegatronWeightLoaders
        
        loader = VllmMegatronWeightLoaders()
        
        # Remove mocked linear module to trigger ImportError
        monkeypatch.delitem(sys.modules, "vllm.model_executor.layers.linear")
        
        with pytest.raises(ImportError, match="Failed to import vllm layer classes"):
            loader.update_megatron_weight_loader()
    
    def test_update_megatron_weight_loader_attribute_error(self, mock_vllm_environment, monkeypatch):
        """Test update_megatron_weight_loader handles AttributeError when setting weight_loader."""
        from agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders import VllmMegatronWeightLoaders
        
        loader = VllmMegatronWeightLoaders()
        
        # Create a metaclass that raises AttributeError when setting weight_loader
        class ErrorMeta(type):
            def __setattr__(cls, name, value):
                if name == "weight_loader":
                    raise AttributeError("Cannot set attribute")
                super().__setattr__(name, value)
        
        class MockColumnParallelLinearWithError(metaclass=ErrorMeta):
            weight_loader = None
        
        # Replace the mock class in sys.modules
        linear_mod = sys.modules["vllm.model_executor.layers.linear"]
        original_class = linear_mod.ColumnParallelLinear
        linear_mod.ColumnParallelLinear = MockColumnParallelLinearWithError
        
        try:
            with pytest.raises(RuntimeError, match="Failed to set weight_loader for"):
                loader.update_megatron_weight_loader()
        finally:
            # Restore original class
            linear_mod.ColumnParallelLinear = original_class
    
    def test_update_megatron_weight_loader_type_error(self, mock_vllm_environment, monkeypatch):
        """Test update_megatron_weight_loader handles TypeError when setting weight_loader."""
        from agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders import VllmMegatronWeightLoaders
        
        loader = VllmMegatronWeightLoaders()
        
        # Create a metaclass that raises TypeError when setting weight_loader
        class TypeErrorMeta(type):
            def __setattr__(cls, name, value):
                if name == "weight_loader":
                    raise TypeError("Invalid type for weight_loader")
                super().__setattr__(name, value)
        
        class MockRowParallelLinearWithError(metaclass=TypeErrorMeta):
            weight_loader = None
        
        # Replace the mock class in sys.modules
        linear_mod = sys.modules["vllm.model_executor.layers.linear"]
        original_class = linear_mod.RowParallelLinear
        linear_mod.RowParallelLinear = MockRowParallelLinearWithError
        
        try:
            with pytest.raises(RuntimeError, match="Failed to set weight_loader for"):
                loader.update_megatron_weight_loader()
        finally:
            # Restore original class
            linear_mod.RowParallelLinear = original_class
    
    def test_fused_moe_import_fallback(self, mock_vllm_environment, monkeypatch):
        """Test that FusedMoE fallback class is used when import fails."""
        # Remove FusedMoE from mocked modules
        monkeypatch.delitem(sys.modules, "vllm.model_executor.layers.fused_moe.layer")
        monkeypatch.delitem(sys.modules, "vllm.model_executor.layers.fused_moe")
        
        # Re-import the module to trigger the fallback
        import importlib
        if "agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders" in sys.modules:
            importlib.reload(sys.modules["agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders"])
        
        from agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders import FusedMoE
        
        # Verify fallback class exists and is a class
        assert FusedMoE is not None
        assert isinstance(FusedMoE, type)
