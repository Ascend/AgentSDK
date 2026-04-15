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
#        http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import sys
import importlib.util
from unittest.mock import MagicMock

# Mock safetensors before importing the module under test
mock_safetensors = MagicMock()
mock_safetensors.__spec__ = importlib.util.spec_from_loader('safetensors', loader=None)
sys.modules['safetensors'] = mock_safetensors

mock_safetensors_torch = MagicMock()
mock_safetensors_torch.__spec__ = importlib.util.spec_from_loader('safetensors.torch', loader=None)
sys.modules['safetensors.torch'] = mock_safetensors_torch

import pytest
import torch
from unittest.mock import patch
from torch import nn

from agentic_rl.base.weight_loaders.megatron_weight_loaders import (
    InferParallelConfig,
    BaseMegatronWeightLoader,
)


class TestInferParallelConfig:
    """Test cases for InferParallelConfig class."""

    def test_init_with_valid_values(self):
        """Test initialization with valid parallel sizes."""
        config = InferParallelConfig(
            infer_tensor_parallel_size=2,
            infer_pipeline_parallel_size=1,
            infer_expert_parallel_size=4
        )
        assert config.infer_tensor_parallel_size == 2
        assert config.infer_pipeline_parallel_size == 1
        assert config.infer_expert_parallel_size == 4

    def test_init_with_single_parallel(self):
        """Test initialization with single parallel size."""
        config = InferParallelConfig(
            infer_tensor_parallel_size=1,
            infer_pipeline_parallel_size=1,
            infer_expert_parallel_size=1
        )
        assert config.infer_tensor_parallel_size == 1
        assert config.infer_pipeline_parallel_size == 1
        assert config.infer_expert_parallel_size == 1


class ConcreteMegatronWeightLoader(BaseMegatronWeightLoader):
    """Concrete implementation for testing abstract class."""

    def get_supported_architectures(self):
        return ["TestModel", "AnotherModel"]

    def update_megatron_weight_loader(self):
        pass


class TestBaseMegatronWeightLoader:
    """Test cases for BaseMegatronWeightLoader class."""

    def test_init_creates_empty_registry(self):
        """Test that initialization creates empty registry."""
        loader = ConcreteMegatronWeightLoader()
        assert loader.model_megatron_weight_loader_registry == {}

    def test_register_model_loader(self):
        """Test registering a model loader."""
        loader = ConcreteMegatronWeightLoader()
        mock_loader = MagicMock()
        loader.register_model_loader("TestModel", mock_loader)
        assert "TestModel" in loader.model_megatron_weight_loader_registry
        assert loader.model_megatron_weight_loader_registry["TestModel"] == mock_loader

    def test_register_multiple_loaders(self):
        """Test registering multiple model loaders."""
        loader = ConcreteMegatronWeightLoader()
        mock_loader1 = MagicMock()
        mock_loader2 = MagicMock()
        loader.register_model_loader("Model1", mock_loader1)
        loader.register_model_loader("Model2", mock_loader2)
        assert len(loader.model_megatron_weight_loader_registry) == 2

    def test_get_model_weight_loader_registered(self):
        """Test getting a registered model weight loader."""
        loader = ConcreteMegatronWeightLoader()
        mock_loader = MagicMock()
        loader.register_model_loader("TestModel", mock_loader)
        result = loader._get_model_weight_loader("TestModel")
        assert result == mock_loader

    def test_get_model_weight_loader_unregistered(self):
        """Test getting an unregistered model weight loader raises error."""
        loader = ConcreteMegatronWeightLoader()
        with pytest.raises(ValueError, match="Model architectures UnknownModel are not supported"):
            loader._get_model_weight_loader("UnknownModel")

    def test_get_supported_architectures(self):
        """Test getting supported architectures."""
        loader = ConcreteMegatronWeightLoader()
        archs = loader.get_supported_architectures()
        assert archs == ["TestModel", "AnotherModel"]


class TestQkvSplitWeight:
    """Test cases for qkv_split_weight static method."""

    def test_qkv_split_weight_basic(self):
        """Test basic QKV weight splitting."""
        infer_config = InferParallelConfig(
            infer_tensor_parallel_size=1,
            infer_pipeline_parallel_size=1,
            infer_expert_parallel_size=1
        )
        hf_config = MagicMock()
        hf_config.num_attention_heads = 8
        hf_config.num_key_value_heads = 2

        query_key_value = torch.randn(96, 64)
        q, k, v = BaseMegatronWeightLoader.qkv_split_weight(
            query_key_value, infer_config, hf_config
        )

        assert q.shape[0] + k.shape[0] + v.shape[0] == 96
        assert q.shape[1] == 64
        assert k.shape[1] == 64
        assert v.shape[1] == 64

    def test_qkv_split_weight_with_tensor_parallel(self):
        """Test QKV weight splitting with tensor parallelism."""
        infer_config = InferParallelConfig(
            infer_tensor_parallel_size=2,
            infer_pipeline_parallel_size=1,
            infer_expert_parallel_size=1
        )
        hf_config = MagicMock()
        hf_config.num_attention_heads = 8
        hf_config.num_key_value_heads = 2

        query_key_value = torch.randn(96, 64)
        q, k, v = BaseMegatronWeightLoader.qkv_split_weight(
            query_key_value, infer_config, hf_config
        )

        assert isinstance(q, torch.Tensor)
        assert isinstance(k, torch.Tensor)
        assert isinstance(v, torch.Tensor)


class TestQkvSplitBias:
    """Test cases for qkv_split_bias static method."""

    def test_qkv_split_bias_basic(self):
        """Test basic QKV bias splitting."""
        infer_config = InferParallelConfig(
            infer_tensor_parallel_size=1,
            infer_pipeline_parallel_size=1,
            infer_expert_parallel_size=1
        )
        hf_config = MagicMock()
        hf_config.num_attention_heads = 8
        hf_config.num_key_value_heads = 2

        query_key_value = torch.randn(96)
        q, k, v = BaseMegatronWeightLoader.qkv_split_bias(
            query_key_value, infer_config, hf_config
        )

        assert q.dim() == 1
        assert k.dim() == 1
        assert v.dim() == 1
        assert q.shape[0] + k.shape[0] + v.shape[0] == 96

    def test_qkv_split_bias_with_tensor_parallel(self):
        """Test QKV bias splitting with tensor parallelism."""
        infer_config = InferParallelConfig(
            infer_tensor_parallel_size=2,
            infer_pipeline_parallel_size=1,
            infer_expert_parallel_size=1
        )
        hf_config = MagicMock()
        hf_config.num_attention_heads = 8
        hf_config.num_key_value_heads = 2

        query_key_value = torch.randn(96)
        q, k, v = BaseMegatronWeightLoader.qkv_split_bias(
            query_key_value, infer_config, hf_config
        )

        assert isinstance(q, torch.Tensor)
        assert isinstance(k, torch.Tensor)
        assert isinstance(v, torch.Tensor)


class TestDefaultWeightLoader:
    """Test cases for default_weight_loader static method."""

    def test_default_weight_loader_size_mismatch(self):
        """Test weight loading with size mismatch raises error."""
        param = torch.randn(10, 20)
        loaded_weight = torch.randn(10, 30)

        with pytest.raises(ValueError, match="parameter size does not match"):
            BaseMegatronWeightLoader.default_weight_loader(param, loaded_weight)

    def test_default_weight_loader_dtype_mismatch(self):
        """Test weight loading with dtype mismatch raises error."""
        param = torch.randn(10, 20, dtype=torch.float32)
        loaded_weight = torch.randn(10, 20, dtype=torch.float64)

        with pytest.raises(ValueError, match="data type should also be the same"):
            BaseMegatronWeightLoader.default_weight_loader(param, loaded_weight)


class TestParallelWeightLoader:
    """Test cases for parallel_weight_loader method."""

    def test_parallel_weight_loader_success(self):
        """Test successful parallel weight loading."""
        loader = ConcreteMegatronWeightLoader()
        param = torch.randn(10, 20)
        loaded_weight = torch.randn(10, 20)

        loader.parallel_weight_loader(param, loaded_weight)

        assert torch.equal(param.data, loaded_weight.data)

    def test_parallel_weight_loader_size_mismatch(self):
        """Test parallel weight loading with size mismatch raises error."""
        loader = ConcreteMegatronWeightLoader()
        param = torch.randn(10, 20)
        loaded_weight = torch.randn(10, 30)

        with pytest.raises(ValueError, match="parameter size is not align"):
            loader.parallel_weight_loader(param, loaded_weight)

    def test_parallel_weight_loader_dtype_mismatch(self):
        """Test parallel weight loading with dtype mismatch raises error."""
        loader = ConcreteMegatronWeightLoader()
        param = torch.randn(10, 20, dtype=torch.float32)
        loaded_weight = torch.randn(10, 20, dtype=torch.float64)

        with pytest.raises(ValueError, match="data type should also be the same"):
            loader.parallel_weight_loader(param, loaded_weight)


class TestLoadSingleWeight:
    """Test cases for load_single_weight static method."""

    def test_load_single_weight_default_loader(self):
        """Test loading single weight with default loader."""
        param = torch.randn(10, 20)
        params_dict = {"test_param": param}
        loaded_weight = torch.randn(10, 20)

        BaseMegatronWeightLoader.load_single_weight(params_dict, "test_param", loaded_weight)

        assert torch.equal(param.data, loaded_weight.data)

    def test_load_single_weight_custom_loader(self):
        """Test loading single weight with custom loader."""
        param = torch.randn(10, 20)
        custom_loader = MagicMock()
        param.weight_loader = custom_loader
        params_dict = {"test_param": param}
        loaded_weight = torch.randn(10, 20)

        BaseMegatronWeightLoader.load_single_weight(params_dict, "test_param", loaded_weight)

        custom_loader.assert_called_once_with(param, loaded_weight)


class TestFinalizeLoading:
    """Test cases for finalize_loading method."""

    def test_finalize_loading(self):
        """Test finalize loading moves model to cuda."""
        loader = ConcreteMegatronWeightLoader()
        model = MagicMock()

        with patch.object(model, 'cuda', return_value=model) as mock_cuda:
            result = loader.finalize_loading(model)
            mock_cuda.assert_called_once()
            assert result == model


class TestLoadMegatronWeights:
    """Test cases for load_megatron_weights method."""

    def test_load_megatron_weights_success(self):
        """Test successful megatron weight loading."""
        loader = ConcreteMegatronWeightLoader()
        mock_model_loader = MagicMock()
        mock_model = MagicMock(spec=nn.Module)
        mock_model.__class__.__name__ = "TestModel"
        mock_model_loader.return_value = mock_model

        loader.register_model_loader("TestModel", mock_model_loader)

        actor_weights = {"layer1.weight": torch.randn(10, 20)}
        infer_config = InferParallelConfig(1, 1, 1)
        hf_config = MagicMock()

        with patch.object(loader, 'finalize_loading', return_value=mock_model) as mock_finalize:
            result = loader.load_megatron_weights(
                actor_weights, mock_model, infer_config, hf_config
            )

            mock_model_loader.assert_called_once_with(
                actor_weights, mock_model, infer_config, hf_config
            )
            mock_finalize.assert_called_once_with(mock_model)
            assert result == mock_model

    def test_load_megatron_weights_unsupported_arch(self):
        """Test loading with unsupported architecture raises error."""
        loader = ConcreteMegatronWeightLoader()
        mock_model = MagicMock(spec=nn.Module)
        mock_model.__class__.__name__ = "UnsupportedModel"

        actor_weights = {}
        infer_config = InferParallelConfig(1, 1, 1)
        hf_config = MagicMock()

        with pytest.raises(ValueError, match="Model architectures UnsupportedModel are not supported"):
            loader.load_megatron_weights(
                actor_weights, mock_model, infer_config, hf_config
            )


class TestLlamaMegatronCoreWeightLoader:
    """Test cases for llama_megatron_core_weight_loader static method."""

    def test_llama_loader_basic(self):
        """Test basic llama weight loading."""
        model = MagicMock(spec=nn.Module)
        model.named_parameters.return_value = [
            ("layer1.weight", torch.randn(10, 20)),
            ("layer2.weight", torch.randn(10, 20)),
        ]

        actor_weights = {
            "layer1.weight": torch.randn(10, 20),
            "layer2.weight": torch.randn(10, 20),
        }

        infer_config = InferParallelConfig(1, 1, 1)
        hf_config = MagicMock()
        hf_config.num_attention_heads = 8
        hf_config.num_key_value_heads = 2

        result = BaseMegatronWeightLoader.llama_megatron_core_weight_loader(
            actor_weights, model, infer_config, hf_config
        )

        assert result == model

    def test_llama_loader_skips_bias_not_in_params(self):
        """Test llama loader skips bias not in params."""
        model = MagicMock(spec=nn.Module)
        model.named_parameters.return_value = [
            ("layer1.weight", torch.randn(10, 20)),
        ]

        actor_weights = {
            "layer1.weight": torch.randn(10, 20),
            "layer1.bias": torch.randn(10),
        }

        infer_config = InferParallelConfig(1, 1, 1)
        hf_config = MagicMock()

        result = BaseMegatronWeightLoader.llama_megatron_core_weight_loader(
            actor_weights, model, infer_config, hf_config
        )

        assert result == model

    def test_llama_loader_skips_rotary_emb(self):
        """Test llama loader skips rotary embedding inv_freq."""
        model = MagicMock(spec=nn.Module)
        model.named_parameters.return_value = [
            ("layer1.weight", torch.randn(10, 20)),
        ]

        actor_weights = {
            "layer1.weight": torch.randn(10, 20),
            "rotary_emb.inv_freq": torch.randn(10),
        }

        infer_config = InferParallelConfig(1, 1, 1)
        hf_config = MagicMock()

        result = BaseMegatronWeightLoader.llama_megatron_core_weight_loader(
            actor_weights, model, infer_config, hf_config
        )

        assert result == model

    def test_llama_loader_skips_lm_head(self):
        """Test llama loader skips lm_head."""
        model = MagicMock(spec=nn.Module)
        model.named_parameters.return_value = [
            ("layer1.weight", torch.randn(10, 20)),
            ("lm_head.weight", torch.randn(10, 20)),
        ]

        actor_weights = {
            "layer1.weight": torch.randn(10, 20),
            "lm_head.weight": torch.randn(10, 20),
        }

        infer_config = InferParallelConfig(1, 1, 1)
        hf_config = MagicMock()

        result = BaseMegatronWeightLoader.llama_megatron_core_weight_loader(
            actor_weights, model, infer_config, hf_config
        )

        assert result == model


class TestQwenMegatronWeightLoader:
    """Test cases for qwen_megatron_weight_loader static method."""

    def test_qwen_loader_basic(self):
        """Test basic qwen weight loading."""
        model = MagicMock(spec=nn.Module)
        model.named_parameters.return_value = [
            ("layer1.weight", torch.randn(10, 20)),
        ]

        actor_weights = {
            "layer1.weight": torch.randn(10, 20),
        }

        infer_config = InferParallelConfig(1, 1, 1)
        hf_config = MagicMock()
        hf_config.num_attention_heads = 8
        hf_config.num_key_value_heads = 2

        result = BaseMegatronWeightLoader.qwen_megatron_weight_loader(
            actor_weights, model, infer_config, hf_config
        )

        assert result == model

    def test_qwen_loader_with_qkv_weight(self):
        """Test qwen loader with qkv weight."""
        model = MagicMock(spec=nn.Module)
        model.named_parameters.return_value = [
            ("layer1.qkv.weight", torch.randn(96, 64)),
        ]

        actor_weights = {
            "layer1.qkv.weight": torch.randn(96, 64),
        }

        infer_config = InferParallelConfig(1, 1, 1)
        hf_config = MagicMock()
        hf_config.num_attention_heads = 8
        hf_config.num_key_value_heads = 2

        result = BaseMegatronWeightLoader.qwen_megatron_weight_loader(
            actor_weights, model, infer_config, hf_config
        )

        assert result == model

    def test_qwen_loader_with_qkv_bias(self):
        """Test qwen loader with qkv bias."""
        model = MagicMock(spec=nn.Module)
        model.named_parameters.return_value = [
            ("layer1.qkv.bias", torch.randn(96)),
        ]

        actor_weights = {
            "layer1.qkv.bias": torch.randn(96),
        }

        infer_config = InferParallelConfig(1, 1, 1)
        hf_config = MagicMock()
        hf_config.num_attention_heads = 8
        hf_config.num_key_value_heads = 2

        result = BaseMegatronWeightLoader.qwen_megatron_weight_loader(
            actor_weights, model, infer_config, hf_config
        )

        assert result == model


class TestDeepseekMegatronWeightLoader:
    """Test cases for deepseek_megatron_weight_loader static method."""

    def test_deepseek_loader_basic(self):
        """Test basic deepseek weight loading."""
        model = MagicMock(spec=nn.Module)
        model.named_parameters.return_value = [
            ("layer1.weight", torch.randn(10, 20)),
        ]

        actor_weights = {
            "layer1.weight": torch.randn(10, 20),
        }

        infer_config = InferParallelConfig(1, 1, 1)
        hf_config = MagicMock()
        hf_config.q_lora_rank = None
        hf_config.qk_nope_head_dim = 16
        hf_config.qk_rope_head_dim = 8
        hf_config.num_attention_heads = 8

        result = BaseMegatronWeightLoader.deepseek_megatron_weight_loader(
            actor_weights, model, infer_config, hf_config
        )

        assert result == model

    def test_deepseek_loader_with_qkv_proj(self):
        """Test deepseek loader with qkv_proj."""
        model = MagicMock(spec=nn.Module)
        q_a_proj = torch.randn(64, 128)
        kv_a_proj = torch.randn(32, 128)
        model.named_parameters.return_value = [
            ("layer1.q_a_proj", q_a_proj),
            ("layer1.kv_a_proj_with_mqa", kv_a_proj),
        ]

        actor_weights = {
            "layer1.qkv_proj": torch.randn(96, 128),
        }

        infer_config = InferParallelConfig(1, 1, 1)
        hf_config = MagicMock()
        hf_config.q_lora_rank = 64
        hf_config.num_attention_heads = 8

        result = BaseMegatronWeightLoader.deepseek_megatron_weight_loader(
            actor_weights, model, infer_config, hf_config
        )

        assert result == model

    def test_deepseek_loader_unexpected_key(self):
        """Test deepseek loader raises error for unexpected key."""
        model = MagicMock(spec=nn.Module)
        model.named_parameters.return_value = []

        actor_weights = {
            "unexpected_key": torch.randn(10, 20),
        }

        infer_config = InferParallelConfig(1, 1, 1)
        hf_config = MagicMock()
        hf_config.q_lora_rank = None
        hf_config.qk_nope_head_dim = 16
        hf_config.qk_rope_head_dim = 8
        hf_config.num_attention_heads = 8

        with pytest.raises(ValueError, match="unexpected key"):
            BaseMegatronWeightLoader.deepseek_megatron_weight_loader(
                actor_weights, model, infer_config, hf_config
            )


@pytest.fixture(scope="module", autouse=True)
def cleanup_registry():
    """Cleanup mock modules after all tests in this module."""
    yield
    modules_to_clean = ['safetensors', 'safetensors.torch']
    for mod in modules_to_clean:
        if mod in sys.modules:
            del sys.modules[mod]
