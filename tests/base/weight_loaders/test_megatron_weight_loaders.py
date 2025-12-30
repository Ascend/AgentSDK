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

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from transformers.configuration_utils import PretrainedConfig

from agentic_rl.base.weight_loaders.megatron_weight_loaders import (
    BaseMegatronWeightLoader,
    InferParallelConfig,
)


# Concrete implementation for testing abstract base class
class ConcreteMegatronWeightLoader(BaseMegatronWeightLoader):
    """Concrete implementation of BaseMegatronWeightLoader for testing."""

    def get_supported_architectures(self):
        return ["TestModel", "AnotherModel"]

    def update_megatron_weight_loader(self):
        pass


# Fixtures
@pytest.fixture
def concrete_loader():
    """Fixture providing a concrete loader instance."""
    return ConcreteMegatronWeightLoader()


@pytest.fixture
def mock_infer_parallel_config():
    """Fixture providing a mock InferParallelConfig."""
    return InferParallelConfig(
        infer_tensor_parallel_size=2,
        infer_pipeline_parallel_size=1,
        infer_expert_parallel_size=1,
    )


@pytest.fixture
def mock_hf_config():
    """Fixture providing a mock HuggingFace config."""
    config = Mock(spec=PretrainedConfig)
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    return config


@pytest.fixture
def mock_model():
    """Fixture providing a mock nn.Module."""
    model = Mock(spec=nn.Module)
    model.__class__.__name__ = "TestModel"
    return model


# Test Classes

class TestInferParallelConfig:
    """Test cases for InferParallelConfig."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        config = InferParallelConfig(
            infer_tensor_parallel_size=4,
            infer_pipeline_parallel_size=2,
            infer_expert_parallel_size=1,
        )
        assert config.infer_tensor_parallel_size == 4
        assert config.infer_pipeline_parallel_size == 2
        assert config.infer_expert_parallel_size == 1

    def test_init_invalid_tensor_parallel_size(self):
        """Test initialization with invalid tensor parallel size."""
        with pytest.raises(ValueError, match="infer_tensor_parallel_size must be a positive integer"):
            InferParallelConfig(
                infer_tensor_parallel_size=0,
                infer_pipeline_parallel_size=1,
                infer_expert_parallel_size=1,
            )

    def test_init_invalid_pipeline_parallel_size(self):
        """Test initialization with invalid pipeline parallel size."""
        with pytest.raises(ValueError, match="infer_pipeline_parallel_size must be a positive integer"):
            InferParallelConfig(
                infer_tensor_parallel_size=1,
                infer_pipeline_parallel_size=-1,
                infer_expert_parallel_size=1,
            )

    def test_init_invalid_expert_parallel_size(self):
        """Test initialization with invalid expert parallel size."""
        with pytest.raises(ValueError, match="infer_expert_parallel_size must be a positive integer"):
            InferParallelConfig(
                infer_tensor_parallel_size=1,
                infer_pipeline_parallel_size=1,
                infer_expert_parallel_size=0,
            )

    def test_init_non_integer_types(self):
        """Test initialization with non-integer types."""
        with pytest.raises(ValueError, match="infer_tensor_parallel_size must be a positive integer"):
            InferParallelConfig(
                infer_tensor_parallel_size="2",
                infer_pipeline_parallel_size=1,
                infer_expert_parallel_size=1,
            )


class TestBaseMegatronWeightLoaderInit:
    """Test cases for BaseMegatronWeightLoader initialization."""

    def test_init_success(self, concrete_loader):
        """Test successful initialization."""
        assert isinstance(concrete_loader.model_megatron_weight_loader_registry, dict)
        assert len(concrete_loader.model_megatron_weight_loader_registry) == 0

    def test_init_creates_empty_registry(self):
        """Test that initialization creates an empty registry."""
        loader = ConcreteMegatronWeightLoader()
        assert not loader.model_megatron_weight_loader_registry


class TestRegisterModelLoader:
    """Test cases for register_model_loader method."""

    def test_register_model_loader_success(self, concrete_loader):
        """Test successful registration of a model loader."""
        def dummy_loader(weights, model, config, hf_config):
            return model

        concrete_loader.register_model_loader("TestModel", dummy_loader)
        assert "TestModel" in concrete_loader.model_megatron_weight_loader_registry
        assert concrete_loader.model_megatron_weight_loader_registry["TestModel"] == dummy_loader

    def test_register_model_loader_override_warning(self, concrete_loader):
        """Test that overriding an existing loader issues a warning."""
        def loader1(weights, model, config, hf_config):
            return model

        def loader2(weights, model, config, hf_config):
            return model

        concrete_loader.register_model_loader("TestModel", loader1)
        
        with patch('agentic_rl.base.weight_loaders.megatron_weight_loaders.logger') as mock_logger:
            concrete_loader.register_model_loader("TestModel", loader2)
            mock_logger.warning.assert_called_once()
            assert "Overriding existing loader" in str(mock_logger.warning.call_args)

    def test_register_model_loader_invalid_model_key(self, concrete_loader):
        """Test registration with invalid model_key type."""
        with pytest.raises(ValueError, match="model_key must be a string"):
            concrete_loader.register_model_loader(123, lambda: None)

    def test_register_model_loader_invalid_loader_func(self, concrete_loader):
        """Test registration with non-callable loader_func."""
        with pytest.raises(ValueError, match="loader_func must be a callable"):
            concrete_loader.register_model_loader("TestModel", "not_callable")

    def test_register_model_loader_exception_handling(self, concrete_loader):
        """Test exception handling during registration."""
        def faulty_loader():
            pass

        # Patch the registry to raise an exception
        mock_dict = MagicMock()
        mock_dict.__setitem__.side_effect = Exception("Test error")
        mock_dict.__contains__.return_value = False
        
        with patch.object(concrete_loader, 'model_megatron_weight_loader_registry', mock_dict):
            with pytest.raises(Exception, match="Test error"):
                concrete_loader.register_model_loader("TestModel", faulty_loader)


class TestGetModelWeightLoader:
    """Test cases for _get_model_weight_loader method."""

    def test_get_model_weight_loader_success(self, concrete_loader):
        """Test successful retrieval of registered loader."""
        def dummy_loader(weights, model, config, hf_config):
            return model

        concrete_loader.register_model_loader("TestModel", dummy_loader)
        loader = concrete_loader._get_model_weight_loader("TestModel")
        assert loader == dummy_loader

    def test_get_model_weight_loader_not_found(self, concrete_loader):
        """Test retrieval of non-existent loader raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported model arch: UnknownModel"):
            concrete_loader._get_model_weight_loader("UnknownModel")

    def test_get_model_weight_loader_shows_supported_architectures(self, concrete_loader):
        """Test that error message includes supported architectures."""
        with pytest.raises(ValueError, match="Supported architectures:.*TestModel.*AnotherModel"):
            concrete_loader._get_model_weight_loader("UnsupportedModel")

    def test_get_model_weight_loader_invalid_arch_type(self, concrete_loader):
        """Test retrieval with invalid arch type."""
        with pytest.raises(ValueError, match="arch must be a string"):
            concrete_loader._get_model_weight_loader(123)


class TestLoadMegatronWeights:
    """Test cases for load_megatron_weights method."""

    def test_load_megatron_weights_success(self, concrete_loader, mock_model, 
                                          mock_infer_parallel_config, mock_hf_config):
        """Test successful weight loading."""
        def dummy_loader(weights, model, config, hf_config):
            return model

        concrete_loader.register_model_loader("TestModel", dummy_loader)
        actor_weights = {"weight1": torch.randn(10, 10)}

        with patch.object(concrete_loader, 'finalize_loading', return_value=mock_model):
            result = concrete_loader.load_megatron_weights(
                actor_weights, mock_model, mock_infer_parallel_config, mock_hf_config
            )
            assert result == mock_model

    def test_load_megatron_weights_invalid_actor_weights(self, concrete_loader, mock_model,
                                                         mock_infer_parallel_config, mock_hf_config):
        """Test with invalid actor_weights type."""
        with pytest.raises(ValueError, match="actor_weights must be a dictionary"):
            concrete_loader.load_megatron_weights(
                "not_a_dict", mock_model, mock_infer_parallel_config, mock_hf_config
            )

    def test_load_megatron_weights_invalid_model(self, concrete_loader, mock_infer_parallel_config,
                                                 mock_hf_config):
        """Test with invalid model type."""
        with pytest.raises(ValueError, match="model must be a nn.Module instance"):
            concrete_loader.load_megatron_weights(
                {}, "not_a_model", mock_infer_parallel_config, mock_hf_config
            )

    def test_load_megatron_weights_invalid_infer_config(self, concrete_loader, mock_model, 
                                                        mock_hf_config):
        """Test with invalid infer_parallel_config type."""
        with pytest.raises(ValueError, match="infer_parallel_config must be an instance of InferParallelConfig"):
            concrete_loader.load_megatron_weights(
                {}, mock_model, "not_a_config", mock_hf_config
            )

    def test_load_megatron_weights_invalid_hf_config(self, concrete_loader, mock_model,
                                                     mock_infer_parallel_config):
        """Test with invalid hf_config type."""
        with pytest.raises(ValueError, match="hf_config must be an instance of PretrainedConfig"):
            concrete_loader.load_megatron_weights(
                {}, mock_model, mock_infer_parallel_config, "not_a_config"
            )

    def test_load_megatron_weights_loader_not_found(self, concrete_loader, mock_model,
                                                    mock_infer_parallel_config, mock_hf_config):
        """Test with unregistered model architecture."""
        mock_model.__class__.__name__ = "UnknownModel"
        
        with pytest.raises(RuntimeError, match="Failed to load Megatron weights"):
            concrete_loader.load_megatron_weights(
                {}, mock_model, mock_infer_parallel_config, mock_hf_config
            )

    def test_load_megatron_weights_loader_exception(self, concrete_loader, mock_model,
                                                    mock_infer_parallel_config, mock_hf_config):
        """Test handling of exception during weight loading."""
        def faulty_loader(weights, model, config, hf_config):
            raise Exception("Loader error")

        concrete_loader.register_model_loader("TestModel", faulty_loader)

        with pytest.raises(RuntimeError, match="Unexpected error occurred during loading Megatron weights"):
            concrete_loader.load_megatron_weights(
                {}, mock_model, mock_infer_parallel_config, mock_hf_config
            )


class TestQwenMegatronWeightLoader:
    """Test cases for qwen_megatron_weight_loader static method."""

    def test_qwen_loader_success_no_qkv(self, mock_model, mock_infer_parallel_config, 
                                        mock_hf_config):
        """Test Qwen loader with simple weights (no QKV)."""
        param1 = nn.Parameter(torch.randn(10, 10))
        param2 = nn.Parameter(torch.randn(5, 5))
        
        mock_model.named_parameters.return_value = [
            ("layer1.weight", param1),
            ("layer2.weight", param2),
        ]
        
        actor_weights = {
            "layer1.weight": torch.randn(10, 10),
            "layer2.weight": torch.randn(5, 5),
        }

        result = BaseMegatronWeightLoader.qwen_megatron_weight_loader(
            actor_weights, mock_model, mock_infer_parallel_config, mock_hf_config
        )
        assert result == mock_model

    def test_qwen_loader_skips_missing_weights(self, mock_model, mock_infer_parallel_config,
                                               mock_hf_config):
        """Test that loader skips weights not in model parameters."""
        param1 = nn.Parameter(torch.randn(10, 10))
        
        mock_model.named_parameters.return_value = [("layer1.weight", param1)]
        
        actor_weights = {
            "layer1.weight": torch.randn(10, 10),
            "layer2.weight": torch.randn(5, 5),  # Not in model
        }

        result = BaseMegatronWeightLoader.qwen_megatron_weight_loader(
            actor_weights, mock_model, mock_infer_parallel_config, mock_hf_config
        )
        assert result == mock_model

    def test_qwen_loader_invalid_actor_weights(self, mock_model, mock_infer_parallel_config,
                                               mock_hf_config):
        """Test with invalid actor_weights type."""
        with pytest.raises(ValueError, match="actor_weights must be a dictionary"):
            BaseMegatronWeightLoader.qwen_megatron_weight_loader(
                "not_a_dict", mock_model, mock_infer_parallel_config, mock_hf_config
            )

    def test_qwen_loader_invalid_model(self, mock_infer_parallel_config, mock_hf_config):
        """Test with invalid model type."""
        with pytest.raises(ValueError, match="model must be a nn.Module instance"):
            BaseMegatronWeightLoader.qwen_megatron_weight_loader(
                {}, "not_a_model", mock_infer_parallel_config, mock_hf_config
            )

    def test_qwen_loader_named_parameters_fails(self, mock_model, mock_infer_parallel_config,
                                                mock_hf_config):
        """Test handling of named_parameters exception."""
        mock_model.named_parameters.side_effect = Exception("Parameters error")

        with pytest.raises(Exception, match="Parameters error"):
            BaseMegatronWeightLoader.qwen_megatron_weight_loader(
                {}, mock_model, mock_infer_parallel_config, mock_hf_config
            )

    def test_qwen_loader_qkv_weight_processing(self, mock_model, mock_infer_parallel_config,
                                               mock_hf_config):
        """Test QKV weight processing in Qwen loader."""
        # Create a parameter with appropriate size for QKV splitting
        qkv_size = 384  # 32 heads * 8 / 2 tensor parallel * (repeats + 2)
        param1 = nn.Parameter(torch.randn(qkv_size, 512))
        
        mock_model.named_parameters.return_value = [
            ("attention.qkv.weight", param1),
        ]
        
        actor_weights = {
            "attention.qkv.weight": torch.randn(qkv_size, 512),
        }

        with patch.object(BaseMegatronWeightLoader, 'qkv_split_weight') as mock_split:
            q = torch.randn(256, 512)
            k = torch.randn(64, 512)
            v = torch.randn(64, 512)
            mock_split.return_value = (q, k, v)
            
            with patch.object(BaseMegatronWeightLoader, 'load_single_weight'):
                result = BaseMegatronWeightLoader.qwen_megatron_weight_loader(
                    actor_weights, mock_model, mock_infer_parallel_config, mock_hf_config
                )
                assert result == mock_model
                mock_split.assert_called_once()

    def test_qwen_loader_qkv_bias_processing(self, mock_model, mock_infer_parallel_config,
                                             mock_hf_config):
        """Test QKV bias processing in Qwen loader."""
        qkv_size = 384
        param1 = nn.Parameter(torch.randn(qkv_size))
        
        mock_model.named_parameters.return_value = [
            ("attention.qkv.bias", param1),
        ]
        
        actor_weights = {
            "attention.qkv.bias": torch.randn(qkv_size),
        }

        with patch.object(BaseMegatronWeightLoader, 'qkv_split_bias') as mock_split:
            q = torch.randn(256)
            k = torch.randn(64)
            v = torch.randn(64)
            mock_split.return_value = (q, k, v)
            
            with patch.object(BaseMegatronWeightLoader, 'load_single_weight'):
                result = BaseMegatronWeightLoader.qwen_megatron_weight_loader(
                    actor_weights, mock_model, mock_infer_parallel_config, mock_hf_config
                )
                assert result == mock_model
                mock_split.assert_called_once()


class TestQkvSplitWeight:
    """Test cases for qkv_split_weight static method."""

    def test_qkv_split_weight_success(self, mock_infer_parallel_config, mock_hf_config):
        """Test successful QKV weight splitting."""
        qkv_tensor = torch.randn(4, 6, 64, 512)  # (ng=4, repeats+2=6, 64, 512)
        qkv_tensor = qkv_tensor.reshape(-1, 512)  # Flatten to (1536, 512)

        q, k, v = BaseMegatronWeightLoader.qkv_split_weight(
            qkv_tensor, mock_infer_parallel_config, mock_hf_config
        )

        assert q.shape == (1024, 512)  # 4 * 4 * 64 = 1024
        assert k.shape == (256, 512)   # 4 * 1 * 64 = 256
        assert v.shape == (256, 512)   # 4 * 1 * 64 = 256

    def test_qkv_split_weight_invalid_tensor(self, mock_infer_parallel_config, mock_hf_config):
        """Test with invalid tensor type."""
        with pytest.raises(ValueError, match="query_key_value must be a torch.Tensor"):
            BaseMegatronWeightLoader.qkv_split_weight(
                "not_a_tensor", mock_infer_parallel_config, mock_hf_config
            )

    def test_qkv_split_weight_invalid_config(self, mock_hf_config):
        """Test with invalid infer_parallel_config type."""
        with pytest.raises(ValueError, match="infer_parallel_config must be an instance of InferParallelConfig"):
            BaseMegatronWeightLoader.qkv_split_weight(
                torch.randn(10, 10), "not_a_config", mock_hf_config
            )

    def test_qkv_split_weight_invalid_hf_config(self, mock_infer_parallel_config):
        """Test with invalid hf_config type."""
        with pytest.raises(ValueError, match="hf_config must be an instance of PretrainedConfig"):
            BaseMegatronWeightLoader.qkv_split_weight(
                torch.randn(10, 10), mock_infer_parallel_config, "not_a_config"
            )

    def test_qkv_split_weight_missing_num_attention_heads(self, mock_infer_parallel_config):
        """Test with hf_config missing num_attention_heads."""
        bad_config = Mock(spec=PretrainedConfig)
        bad_config.num_key_value_heads = 8
        
        with pytest.raises(RuntimeError, match="Failed to split QKV weight due to tensor operation"):
            BaseMegatronWeightLoader.qkv_split_weight(
                torch.randn(10, 10), mock_infer_parallel_config, bad_config
            )

    def test_qkv_split_weight_missing_num_key_value_heads(self, mock_infer_parallel_config):
        """Test with hf_config missing num_key_value_heads."""
        bad_config = Mock(spec=PretrainedConfig)
        bad_config.num_attention_heads = 32
        
        with pytest.raises(RuntimeError, match="Failed to split QKV weight due to tensor operation"):
            BaseMegatronWeightLoader.qkv_split_weight(
                torch.randn(10, 10), mock_infer_parallel_config, bad_config
            )

    def test_qkv_split_weight_incompatible_attention_heads(self, mock_infer_parallel_config):
        """Test with attention heads not divisible by tensor parallel size."""
        bad_config = Mock(spec=PretrainedConfig)
        bad_config.num_attention_heads = 31  # Not divisible by 2
        bad_config.num_key_value_heads = 8

        with pytest.raises(RuntimeError, match="Failed to split QKV weight due to tensor operation"):
            BaseMegatronWeightLoader.qkv_split_weight(
                torch.randn(10, 10), mock_infer_parallel_config, bad_config
            )

    def test_qkv_split_weight_incompatible_kv_heads(self, mock_infer_parallel_config):
        """Test with kv heads not divisible by tensor parallel size."""
        bad_config = Mock(spec=PretrainedConfig)
        bad_config.num_attention_heads = 32
        bad_config.num_key_value_heads = 7  # Not divisible by 2

        with pytest.raises(RuntimeError, match="Failed to split QKV weight due to tensor operation"):
            BaseMegatronWeightLoader.qkv_split_weight(
                torch.randn(10, 10), mock_infer_parallel_config, bad_config
            )

    def test_qkv_split_weight_zero_heads_per_gpu(self, mock_hf_config):
        """Test with configuration resulting in zero heads per GPU."""
        large_parallel_config = InferParallelConfig(
            infer_tensor_parallel_size=16,  # Larger than num_key_value_heads
            infer_pipeline_parallel_size=1,
            infer_expert_parallel_size=1,
        )

        with pytest.raises(RuntimeError, match="Failed to split QKV weight due to tensor operation"):
            BaseMegatronWeightLoader.qkv_split_weight(
                torch.randn(10, 10), large_parallel_config, mock_hf_config
            )


class TestQkvSplitBias:
    """Test cases for qkv_split_bias static method."""

    def test_qkv_split_bias_success(self, mock_infer_parallel_config, mock_hf_config):
        """Test successful QKV bias splitting."""
        qkv_bias = torch.randn(4, 6, 64)  # (ng=4, repeats+2=6, 64)
        qkv_bias = qkv_bias.reshape(-1)  # Flatten to (1536,)

        q, k, v = BaseMegatronWeightLoader.qkv_split_bias(
            qkv_bias, mock_infer_parallel_config, mock_hf_config
        )

        assert q.shape == (1024,)  # 4 * 4 * 64 = 1024
        assert k.shape == (256,)   # 4 * 1 * 64 = 256
        assert v.shape == (256,)   # 4 * 1 * 64 = 256

    def test_qkv_split_bias_invalid_tensor(self, mock_infer_parallel_config, mock_hf_config):
        """Test with invalid tensor type."""
        with pytest.raises(ValueError, match="query_key_value must be a torch.Tensor"):
            BaseMegatronWeightLoader.qkv_split_bias(
                "not_a_tensor", mock_infer_parallel_config, mock_hf_config
            )

    def test_qkv_split_bias_invalid_config(self, mock_hf_config):
        """Test with invalid infer_parallel_config type."""
        with pytest.raises(ValueError, match="infer_parallel_config must be an instance of InferParallelConfig"):
            BaseMegatronWeightLoader.qkv_split_bias(
                torch.randn(10), "not_a_config", mock_hf_config
            )

    def test_qkv_split_bias_invalid_hf_config(self, mock_infer_parallel_config):
        """Test with invalid hf_config type."""
        with pytest.raises(ValueError, match="hf_config must be an instance of PretrainedConfig"):
            BaseMegatronWeightLoader.qkv_split_bias(
                torch.randn(10), mock_infer_parallel_config, "not_a_config"
            )

    def test_qkv_split_bias_missing_attributes(self, mock_infer_parallel_config):
        """Test with hf_config missing required attributes."""
        bad_config = Mock(spec=PretrainedConfig)
        
        with pytest.raises(RuntimeError, match="Failed to split QKV bias due to tensor operation"):
            BaseMegatronWeightLoader.qkv_split_bias(
                torch.randn(10), mock_infer_parallel_config, bad_config
            )

    def test_qkv_split_bias_incompatible_attention_heads(self, mock_infer_parallel_config):
        """Test with incompatible attention heads."""
        bad_config = Mock(spec=PretrainedConfig)
        bad_config.num_attention_heads = 31
        bad_config.num_key_value_heads = 8

        with pytest.raises(RuntimeError, match="Failed to split QKV bias due to tensor operation"):
            BaseMegatronWeightLoader.qkv_split_bias(
                torch.randn(10), mock_infer_parallel_config, bad_config
            )

    def test_qkv_split_bias_zero_heads_per_gpu(self, mock_hf_config):
        """Test with configuration resulting in zero heads per GPU."""
        large_parallel_config = InferParallelConfig(
            infer_tensor_parallel_size=16,
            infer_pipeline_parallel_size=1,
            infer_expert_parallel_size=1,
        )

        with pytest.raises(RuntimeError, match="Failed to split QKV bias due to tensor operation"):
            BaseMegatronWeightLoader.qkv_split_bias(
                torch.randn(10), large_parallel_config, mock_hf_config
            )


class TestLoadSingleWeight:
    """Test cases for load_single_weight static method."""

    def test_load_single_weight_success(self):
        """Test successful loading of a single weight."""
        param = nn.Parameter(torch.randn(10, 10))
        loaded_weight = torch.randn(10, 10)
        params_dict = {"layer.weight": param}

        BaseMegatronWeightLoader.load_single_weight(params_dict, "layer.weight", loaded_weight)
        # Should complete without error

    def test_load_single_weight_with_custom_loader(self):
        """Test loading with custom weight_loader attribute."""
        param = nn.Parameter(torch.randn(10, 10))
        loaded_weight = torch.randn(10, 10)
        
        custom_loader_called = []

        def custom_loader(p, w):
            custom_loader_called.append(True)
            p.data = w.data
        
        param.weight_loader = custom_loader
        params_dict = {"layer.weight": param}

        BaseMegatronWeightLoader.load_single_weight(params_dict, "layer.weight", loaded_weight)
        assert len(custom_loader_called) == 1

    def test_load_single_weight_not_in_params_dict(self):
        """Test loading weight not in params_dict (should log warning and return)."""
        params_dict = {}
        loaded_weight = torch.randn(10, 10)

        with patch('agentic_rl.base.weight_loaders.megatron_weight_loaders.logger') as mock_logger:
            BaseMegatronWeightLoader.load_single_weight(params_dict, "missing.weight", loaded_weight)
            mock_logger.warning.assert_called_once()

    def test_load_single_weight_invalid_params_dict(self):
        """Test with invalid params_dict type."""
        with pytest.raises(ValueError, match="params_dict must be a dictionary"):
            BaseMegatronWeightLoader.load_single_weight("not_a_dict", "name", torch.randn(10))

    def test_load_single_weight_invalid_name(self):
        """Test with invalid name type."""
        with pytest.raises(ValueError, match="name must be a string"):
            BaseMegatronWeightLoader.load_single_weight({}, 123, torch.randn(10))

    def test_load_single_weight_invalid_loaded_weight(self):
        """Test with invalid loaded_weight type."""
        with pytest.raises(ValueError, match="loaded_weight must be a torch.Tensor"):
            BaseMegatronWeightLoader.load_single_weight({}, "name", "not_a_tensor")

    def test_load_single_weight_loader_exception(self):
        """Test handling of exception during weight loading."""
        param = nn.Parameter(torch.randn(10, 10))
        loaded_weight = torch.randn(10, 10)
        
        def faulty_loader(p, w):
            raise Exception("Loader error")
        
        param.weight_loader = faulty_loader
        params_dict = {"layer.weight": param}

        with pytest.raises(RuntimeError, match="Unexpected error occurred during loading single weight"):
            BaseMegatronWeightLoader.load_single_weight(params_dict, "layer.weight", loaded_weight)


class TestDefaultWeightLoader:
    """Test cases for default_weight_loader static method."""

    def test_default_weight_loader_success(self):
        """Test successful weight loading with matching sizes and types."""
        param = torch.randn(10, 10)
        loaded_weight = torch.randn(10, 10)

        BaseMegatronWeightLoader.default_weight_loader(param, loaded_weight)
        assert torch.equal(param.data, loaded_weight.data)

    def test_default_weight_loader_size_mismatch(self):
        """Test with mismatched tensor sizes."""
        param = torch.randn(10, 10)
        loaded_weight = torch.randn(5, 5)

        with pytest.raises(RuntimeError, match="Failed to load weight"):
            BaseMegatronWeightLoader.default_weight_loader(param, loaded_weight)

    def test_default_weight_loader_dtype_mismatch(self):
        """Test with mismatched data types."""
        param = torch.randn(10, 10, dtype=torch.float32)
        loaded_weight = torch.randn(10, 10, dtype=torch.float16)

        with pytest.raises(RuntimeError, match="Failed to load weight"):
            BaseMegatronWeightLoader.default_weight_loader(param, loaded_weight)

    def test_default_weight_loader_invalid_param(self):
        """Test with invalid param type."""
        with pytest.raises(ValueError, match="param must be a torch.Tensor"):
            BaseMegatronWeightLoader.default_weight_loader("not_a_tensor", torch.randn(10))

    def test_default_weight_loader_invalid_loaded_weight(self):
        """Test with invalid loaded_weight type."""
        with pytest.raises(ValueError, match="loaded_weight must be a torch.Tensor"):
            BaseMegatronWeightLoader.default_weight_loader(torch.randn(10), "not_a_tensor")


class TestParallelWeightLoader:
    """Test cases for parallel_weight_loader method."""

    def test_parallel_weight_loader_success(self, concrete_loader):
        """Test successful parallel weight loading."""
        param = torch.randn(10, 10)
        loaded_weight = torch.randn(10, 10)

        concrete_loader.parallel_weight_loader(param, loaded_weight)
        assert torch.equal(param.data, loaded_weight.data)

    def test_parallel_weight_loader_size_mismatch(self, concrete_loader):
        """Test with mismatched tensor sizes."""
        param = torch.randn(10, 10)
        loaded_weight = torch.randn(5, 5)

        with pytest.raises(RuntimeError, match="Failed to load parallel weight due to tensor operation"):
            concrete_loader.parallel_weight_loader(param, loaded_weight)

    def test_parallel_weight_loader_dtype_mismatch(self, concrete_loader):
        """Test with mismatched data types."""
        param = torch.randn(10, 10, dtype=torch.float32)
        loaded_weight = torch.randn(10, 10, dtype=torch.float16)

        with pytest.raises(RuntimeError, match="Failed to load parallel weight due to tensor operation"):
            concrete_loader.parallel_weight_loader(param, loaded_weight)

    def test_parallel_weight_loader_invalid_param(self, concrete_loader):
        """Test with invalid param type."""
        with pytest.raises(ValueError, match="param must be a torch.Tensor"):
            concrete_loader.parallel_weight_loader("not_a_tensor", torch.randn(10))

    def test_parallel_weight_loader_invalid_loaded_weight(self, concrete_loader):
        """Test with invalid loaded_weight type."""
        with pytest.raises(ValueError, match="loaded_weight must be a torch.Tensor"):
            concrete_loader.parallel_weight_loader(torch.randn(10), "not_a_tensor")


class TestFinalizeLoading:
    """Test cases for finalize_loading method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_finalize_loading_success_with_cuda(self, concrete_loader, mock_model):
        """Test successful model finalization with CUDA available."""
        mock_model.cuda.return_value = mock_model

        result = concrete_loader.finalize_loading(mock_model)
        assert result == mock_model
        mock_model.cuda.assert_called_once()

    def test_finalize_loading_no_cuda(self, concrete_loader, mock_model):
        """Test finalization when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                concrete_loader.finalize_loading(mock_model)

    def test_finalize_loading_invalid_model(self, concrete_loader):
        """Test with invalid model type."""
        with pytest.raises(ValueError, match="model must be a nn.Module instance"):
            concrete_loader.finalize_loading("not_a_model")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_finalize_loading_cuda_exception(self, concrete_loader, mock_model):
        """Test handling of exception during CUDA operation."""
        mock_model.cuda.side_effect = Exception("CUDA error")

        with pytest.raises(RuntimeError, match="Failed to finalize model loading"):
            concrete_loader.finalize_loading(mock_model)
