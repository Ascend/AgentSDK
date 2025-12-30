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

from unittest.mock import MagicMock, patch
import pytest
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig


class TestWeightManager:
    """Test suite for WeightManager class."""

    @pytest.fixture
    def mock_vllm_weight_loaders(self):
        """Fixture providing a mock VllmMegatronWeightLoaders instance."""
        mock_loader = MagicMock()
        mock_loader.update_megatron_weight_loader = MagicMock()
        mock_loader.load_megatron_weights = MagicMock()
        return mock_loader

    @pytest.fixture
    def mock_model(self):
        """Fixture providing a mock nn.Module instance."""
        model = MagicMock(spec=nn.Module)
        model.__class__.__name__ = "Module"
        # Set up the model structure for MLA testing
        model.model = MagicMock()
        model.model.start_layer = 0
        model.model.end_layer = 2
        model.model.layers = []
        return model

    @pytest.fixture
    def mock_hf_config(self):
        """Fixture providing a mock PretrainedConfig instance."""
        config = MagicMock(spec=PretrainedConfig)
        return config

    @pytest.fixture
    def weight_manager(self, mock_vllm_weight_loaders):
        """Fixture providing a WeightManager instance with mocked dependencies."""
        with patch(
            "agentic_rl.runner.infer_adapter.vllm.weight_manager.VllmMegatronWeightLoaders",
            return_value=mock_vllm_weight_loaders,
        ):
            from agentic_rl.runner.infer_adapter.vllm.weight_manager import WeightManager
            return WeightManager(
                infer_tensor_parallel_size=2,
                infer_pipeline_parallel_size=2,
                infer_expert_parallel_size=1,
                load_format="megatron",
            )

    # Tests for __init__
    def test_init_success(self, mock_vllm_weight_loaders):
        """Test successful WeightManager initialization with valid parameters."""
        with patch(
            "agentic_rl.runner.infer_adapter.vllm.weight_manager.VllmMegatronWeightLoaders",
            return_value=mock_vllm_weight_loaders,
        ):
            from agentic_rl.runner.infer_adapter.vllm.weight_manager import WeightManager
            manager = WeightManager(
                infer_tensor_parallel_size=4,
                infer_pipeline_parallel_size=2,
                infer_expert_parallel_size=1,
                load_format="megatron",
            )
            assert manager.infer_tensor_parallel_size == 4
            assert manager.infer_pipeline_parallel_size == 2
            assert manager.infer_expert_parallel_size == 1
            assert manager.load_format == "megatron"
            assert manager.vllm_megatron_weight_loaders is not None

    def test_init_invalid_tensor_parallel_size_negative(self):
        """Test initialization fails with negative tensor parallel size."""
        from agentic_rl.runner.infer_adapter.vllm.weight_manager import WeightManager
        with pytest.raises(ValueError, match="infer_tensor_parallel_size must be a positive integer"):
            WeightManager(
                infer_tensor_parallel_size=-1,
                infer_pipeline_parallel_size=2,
                infer_expert_parallel_size=1,
                load_format="megatron",
            )

    def test_init_invalid_tensor_parallel_size_zero(self):
        """Test initialization fails with zero tensor parallel size."""
        from agentic_rl.runner.infer_adapter.vllm.weight_manager import WeightManager
        with pytest.raises(ValueError, match="infer_tensor_parallel_size must be a positive integer"):
            WeightManager(
                infer_tensor_parallel_size=0,
                infer_pipeline_parallel_size=2,
                infer_expert_parallel_size=1,
                load_format="megatron",
            )

    def test_init_invalid_tensor_parallel_size_non_int(self):
        """Test initialization fails with non-integer tensor parallel size."""
        from agentic_rl.runner.infer_adapter.vllm.weight_manager import WeightManager
        with pytest.raises(ValueError, match="infer_tensor_parallel_size must be a positive integer"):
            WeightManager(
                infer_tensor_parallel_size="2",
                infer_pipeline_parallel_size=2,
                infer_expert_parallel_size=1,
                load_format="megatron",
            )

    def test_init_invalid_pipeline_parallel_size_negative(self):
        """Test initialization fails with negative pipeline parallel size."""
        from agentic_rl.runner.infer_adapter.vllm.weight_manager import WeightManager
        with pytest.raises(ValueError, match="infer_pipeline_parallel_size must be a positive integer"):
            WeightManager(
                infer_tensor_parallel_size=2,
                infer_pipeline_parallel_size=-1,
                infer_expert_parallel_size=1,
                load_format="megatron",
            )

    def test_init_invalid_pipeline_parallel_size_zero(self):
        """Test initialization fails with zero pipeline parallel size."""
        from agentic_rl.runner.infer_adapter.vllm.weight_manager import WeightManager
        with pytest.raises(ValueError, match="infer_pipeline_parallel_size must be a positive integer"):
            WeightManager(
                infer_tensor_parallel_size=2,
                infer_pipeline_parallel_size=0,
                infer_expert_parallel_size=1,
                load_format="megatron",
            )

    def test_init_invalid_expert_parallel_size_negative(self):
        """Test initialization fails with negative expert parallel size."""
        from agentic_rl.runner.infer_adapter.vllm.weight_manager import WeightManager
        with pytest.raises(ValueError, match="infer_expert_parallel_size must be a positive integer"):
            WeightManager(
                infer_tensor_parallel_size=2,
                infer_pipeline_parallel_size=2,
                infer_expert_parallel_size=-1,
                load_format="megatron",
            )

    def test_init_invalid_expert_parallel_size_zero(self):
        """Test initialization fails with zero expert parallel size."""
        from agentic_rl.runner.infer_adapter.vllm.weight_manager import WeightManager
        with pytest.raises(ValueError, match="infer_expert_parallel_size must be a positive integer"):
            WeightManager(
                infer_tensor_parallel_size=2,
                infer_pipeline_parallel_size=2,
                infer_expert_parallel_size=0,
                load_format="megatron",
            )

    def test_init_invalid_load_format_empty_string(self):
        """Test initialization fails with empty string load format."""
        from agentic_rl.runner.infer_adapter.vllm.weight_manager import WeightManager
        with pytest.raises(ValueError, match="load_format must be a non-empty string"):
            WeightManager(
                infer_tensor_parallel_size=2,
                infer_pipeline_parallel_size=2,
                infer_expert_parallel_size=1,
                load_format="",
            )

    def test_init_invalid_load_format_non_string(self):
        """Test initialization fails with non-string load format."""
        from agentic_rl.runner.infer_adapter.vllm.weight_manager import WeightManager
        with pytest.raises(ValueError, match="load_format must be a non-empty string"):
            WeightManager(
                infer_tensor_parallel_size=2,
                infer_pipeline_parallel_size=2,
                infer_expert_parallel_size=1,
                load_format=123,
            )

    def test_init_vllm_loader_initialization_failure(self):
        """Test initialization raises RuntimeError when VllmMegatronWeightLoaders fails."""
        with patch(
            "agentic_rl.runner.infer_adapter.vllm.weight_manager.VllmMegatronWeightLoaders",
            side_effect=Exception("Loader init failed"),
        ):
            from agentic_rl.runner.infer_adapter.vllm.weight_manager import WeightManager
            with pytest.raises(RuntimeError, match="WeightManager initialization failed"):
                WeightManager(
                    infer_tensor_parallel_size=2,
                    infer_pipeline_parallel_size=2,
                    infer_expert_parallel_size=1,
                    load_format="megatron",
                )

    # Tests for initialize_weight_loader()
    def test_initialize_weight_loader_megatron(self, weight_manager):
        """Test successful weight loader initialization with megatron format."""
        weight_manager.initialize_weight_loader()
        weight_manager.vllm_megatron_weight_loaders.update_megatron_weight_loader.assert_called_once()

    def test_initialize_weight_loader_unsupported_format(self, mock_vllm_weight_loaders):
        """Test weight loader initialization logs warning for unsupported format."""
        with patch(
            "agentic_rl.runner.infer_adapter.vllm.weight_manager.VllmMegatronWeightLoaders",
            return_value=mock_vllm_weight_loaders,
        ):
            from agentic_rl.runner.infer_adapter.vllm.weight_manager import WeightManager
            manager = WeightManager(
                infer_tensor_parallel_size=2,
                infer_pipeline_parallel_size=2,
                infer_expert_parallel_size=1,
                load_format="unsupported_format",
            )
            with patch("agentic_rl.runner.infer_adapter.vllm.weight_manager.logger") as mock_logger:
                manager.initialize_weight_loader()
                mock_logger.warning.assert_called_once()
                assert "Unsupported load format" in str(mock_logger.warning.call_args)

    def test_initialize_weight_loader_attribute_error(self, weight_manager):
        """Test weight loader initialization raises RuntimeError on AttributeError."""
        weight_manager.vllm_megatron_weight_loaders.update_megatron_weight_loader.side_effect = AttributeError(
            "Missing attribute"
        )
        with pytest.raises(RuntimeError, match="Failed to initialize weight loader due to missing attribute"):
            weight_manager.initialize_weight_loader()

    def test_initialize_weight_loader_general_exception(self, weight_manager):
        """Test weight loader initialization raises RuntimeError on general exception."""
        weight_manager.vllm_megatron_weight_loaders.update_megatron_weight_loader.side_effect = Exception(
            "Unexpected error"
        )
        with pytest.raises(RuntimeError, match="Unexpected error during weight loader initialization"):
            weight_manager.initialize_weight_loader()

    def test_load_megatron_weights_invalid_params(self, weight_manager, mock_model, mock_hf_config):
        """Test load_megatron_weights raises ValueError for non-dict params."""
        with pytest.raises(ValueError, match="params must be a dict"):
            weight_manager.load_megatron_weights("not_a_dict", mock_model, mock_hf_config)

    def test_load_megatron_weights_invalid_model(self, weight_manager, mock_hf_config):
        """Test load_megatron_weights raises ValueError for non-Module model."""
        params = {"param1": "value1"}
        with pytest.raises(ValueError, match="model must be a nn.Module instance"):
            weight_manager.load_megatron_weights(params, "not_a_model", mock_hf_config)

    def test_load_megatron_weights_general_exception(self, weight_manager, mock_model, mock_hf_config):
        """Test load_megatron_weights raises RuntimeError on unexpected error."""
        params = {"param1": "value1"}
        with patch(
            "agentic_rl.runner.infer_adapter.vllm.weight_manager.InferParallelConfig"
        ) as mock_config_class:
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config
            weight_manager.vllm_megatron_weight_loaders.load_megatron_weights.side_effect = RuntimeError(
                "Unexpected error"
            )
            
            with pytest.raises(RuntimeError, match="Weight loading failed"):
                weight_manager.load_megatron_weights(params, mock_model, mock_hf_config)

    def test_load_megatron_weights_with_mla_processing(self, weight_manager, mock_hf_config):
        """Test load_megatron_weights processes MLA weights when present."""
        params = {"param1": "value1"}
        
        # Create a model with MLA structure
        mock_model = MagicMock(spec=nn.Module)
        mock_model.model = MagicMock()
        mock_model.model.start_layer = 0
        mock_model.model.end_layer = 2
        
        # Create mock layers with MLA
        layer0 = MagicMock()
        layer0.self_attn.mla_attn.impl = MagicMock()
        layer0.self_attn.mla_attn.impl.w_kc = MagicMock()
        layer0.self_attn.mla_attn.impl.w_vc = MagicMock()
        layer0.self_attn.mla_attn.impl.W_UV = MagicMock()
        layer0.self_attn.mla_attn.impl.W_UK_T = MagicMock()
        layer0.self_attn.mla_attn.impl.process_weights_after_loading = MagicMock()
        
        layer1 = MagicMock()
        layer1.self_attn.mla_attn.impl = MagicMock()
        layer1.self_attn.mla_attn.impl.w_kc = MagicMock()
        layer1.self_attn.mla_attn.impl.w_vc = MagicMock()
        layer1.self_attn.mla_attn.impl.W_UV = MagicMock()
        layer1.self_attn.mla_attn.impl.W_UK_T = MagicMock()
        layer1.self_attn.mla_attn.impl.process_weights_after_loading = MagicMock()
        
        mock_model.model.layers = [layer0, layer1]
        
        with patch(
            "agentic_rl.runner.infer_adapter.vllm.weight_manager.InferParallelConfig"
        ) as mock_config_class:
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config
            
            weight_manager.load_megatron_weights(params, mock_model, mock_hf_config)
            
            # Verify MLA processing was called
            assert layer0.self_attn.mla_attn.impl.w_kc is None
            assert layer0.self_attn.mla_attn.impl.w_vc is None
            assert layer1.self_attn.mla_attn.impl.w_kc is None
            assert layer1.self_attn.mla_attn.impl.w_vc is None
            layer0.self_attn.mla_attn.impl.process_weights_after_loading.assert_called_once_with(None)
            layer1.self_attn.mla_attn.impl.process_weights_after_loading.assert_called_once_with(None)

    def test_load_megatron_weights_without_mla(self, weight_manager, mock_hf_config):
        """Test load_megatron_weights skips MLA processing when not present."""
        params = {"param1": "value1"}
        
        # Create a model without MLA structure
        mock_model = MagicMock(spec=nn.Module)
        mock_model.model = MagicMock()
        mock_model.model.start_layer = 0
        mock_model.model.end_layer = 2
        
        # Create mock layers without MLA
        layer0 = MagicMock()
        del layer0.self_attn.mla_attn  # Remove mla_attn attribute
        layer1 = MagicMock()
        del layer1.self_attn.mla_attn
        
        mock_model.model.layers = [layer0, layer1]
        
        with patch(
            "agentic_rl.runner.infer_adapter.vllm.weight_manager.InferParallelConfig"
        ) as mock_config_class:
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config
            
            # Should not raise an exception, just skip MLA processing
            weight_manager.load_megatron_weights(params, mock_model, mock_hf_config)
            
            weight_manager.vllm_megatron_weight_loaders.load_megatron_weights.assert_called_once()

