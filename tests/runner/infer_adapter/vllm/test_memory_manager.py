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


from agentic_rl.runner.infer_adapter.vllm.memory_manager import MemoryManager


class TestMemoryManager:
    """Test suite for MemoryManager class."""

    @pytest.fixture
    def memory_manager(self):
        """Fixture providing a MemoryManager instance."""
        return MemoryManager()

    @pytest.fixture
    def mock_model(self):
        """Fixture providing a mock model with named_parameters."""
        model = MagicMock()
        param1 = MagicMock()
        param1.data = MagicMock()
        param2 = MagicMock()
        param2.data = MagicMock()
        model.named_parameters.return_value = [
            ("layer1.weight", param1),
            ("layer2.bias", param2),
        ]
        return model

    @pytest.fixture
    def mock_cpu_model(self):
        """Fixture providing a mock CPU model dictionary."""
        return {
            "layer1.weight": MagicMock(),
            "layer2.bias": MagicMock(),
        }

    def test_init(self, memory_manager):
        """Test MemoryManager initialization."""
        assert memory_manager.cpu_model == {}
        assert isinstance(memory_manager.cpu_model, dict)

    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.gc")
    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.torch")
    def test_clear_gpu_memory_success(self, mock_torch, mock_gc):
        """Test successful GPU memory clearing and garbage collection."""
        MemoryManager.clear_gpu_memory()
        
        mock_torch.cuda.empty_cache.assert_called_once()
        mock_gc.collect.assert_called_once()

    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.logger")
    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.gc")
    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.torch")
    def test_clear_gpu_memory_cuda_runtime_error(self, mock_torch, mock_gc, mock_logger):
        """Test clear_gpu_memory when CUDA is not available (RuntimeError)."""
        mock_torch.cuda.empty_cache.side_effect = RuntimeError("CUDA not available")
        
        MemoryManager.clear_gpu_memory()
        
        mock_logger.warning.assert_called()
        mock_gc.collect.assert_called_once()

    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.logger")
    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.gc")
    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.torch")
    def test_clear_gpu_memory_cuda_unexpected_error(self, mock_torch, mock_gc, mock_logger):
        """Test clear_gpu_memory when unexpected error occurs clearing CUDA cache."""
        mock_torch.cuda.empty_cache.side_effect = Exception("Unexpected error")
        
        MemoryManager.clear_gpu_memory()
        
        mock_logger.warning.assert_called()
        mock_gc.collect.assert_called_once()

    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.logger")
    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.gc")
    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.torch")
    def test_clear_gpu_memory_gc_fails(self, mock_torch, mock_gc, mock_logger):
        """Test clear_gpu_memory when garbage collection fails."""
        mock_gc.collect.side_effect = Exception("GC failed")
        
        MemoryManager.clear_gpu_memory()
        
        mock_torch.cuda.empty_cache.assert_called_once()
        mock_logger.warning.assert_called()

    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.torch")
    def test_create_cpu_model_copy_success(self, mock_torch, memory_manager, mock_model):
        """Test successful CPU model copy creation."""
        mock_tensor = MagicMock()
        mock_torch.empty_like.return_value = mock_tensor
        
        result = memory_manager.create_cpu_model_copy(mock_model)
        
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "layer1.weight" in result
        assert "layer2.bias" in result
        assert mock_torch.empty_like.call_count == 2

    def test_create_cpu_model_copy_model_none(self, memory_manager):
        """Test create_cpu_model_copy with None model."""
        with pytest.raises((ValueError, TypeError)):
            memory_manager.create_cpu_model_copy(None)

    def test_create_cpu_model_copy_model_no_named_parameters(self, memory_manager):
        """Test create_cpu_model_copy when model lacks named_parameters."""
        model = MagicMock(spec=[])
        
        with pytest.raises((ValueError, TypeError)):
            memory_manager.create_cpu_model_copy(model)

    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.logger")
    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.torch")
    def test_create_cpu_model_copy_allocation_fails(self, mock_torch, mock_logger, memory_manager, mock_model):
        """Test create_cpu_model_copy when CPU allocation fails."""
        mock_torch.empty_like.side_effect = RuntimeError("Out of memory")
        
        with pytest.raises(RuntimeError, match="Failed to allocate CPU memory"):
            memory_manager.create_cpu_model_copy(mock_model)

    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.logger")
    def test_create_cpu_model_copy_iteration_fails(self, mock_logger, memory_manager):
        """Test create_cpu_model_copy when parameter iteration fails."""
        model = MagicMock()
        model.named_parameters.side_effect = Exception("Iteration failed")
        
        with pytest.raises(RuntimeError, match="Unexpected error during creating CPU model copy"):
            memory_manager.create_cpu_model_copy(model)

    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.logger")
    def test_offload_model_weights_success(self, mock_logger, memory_manager, mock_model, mock_cpu_model):
        """Test successful weight offloading."""
        with patch.dict("os.environ", {"RANK": "0"}):
            memory_manager.offload_model_weights(mock_model, mock_cpu_model)
        
        # Verify parameters were updated
        for name, param in mock_model.named_parameters():
            assert param.data == mock_cpu_model[name]

    def test_offload_model_weights_model_none(self, memory_manager, mock_cpu_model):
        """Test offload_model_weights with None model."""
        with pytest.raises((ValueError, TypeError)):
            memory_manager.offload_model_weights(None, mock_cpu_model)

    def test_offload_model_weights_model_no_named_parameters(self, memory_manager, mock_cpu_model):
        """Test offload_model_weights when model lacks named_parameters."""
        model = MagicMock(spec=[])
        
        with pytest.raises((ValueError, TypeError)):
            memory_manager.offload_model_weights(model, mock_cpu_model)

    def test_offload_model_weights_cpu_model_not_dict(self, memory_manager, mock_model):
        """Test offload_model_weights when cpu_model is not a dict."""
        with pytest.raises((ValueError, TypeError)):
            memory_manager.offload_model_weights(mock_model, "not_a_dict")

    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.logger")
    def test_offload_model_weights_rank_not_set(self, mock_logger, memory_manager, mock_model, mock_cpu_model):
        """Test offload_model_weights when RANK environment variable is not set."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="Failed to retrieve rank"):
                memory_manager.offload_model_weights(mock_model, mock_cpu_model)

    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.logger")
    def test_offload_model_weights_rank_invalid_value(self, mock_logger, memory_manager, mock_model, mock_cpu_model):
        """Test offload_model_weights when RANK has invalid value."""
        with patch.dict("os.environ", {"RANK": "invalid"}):
            with pytest.raises(RuntimeError, match="Failed to retrieve rank"):
                memory_manager.offload_model_weights(mock_model, mock_cpu_model)

    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.logger")
    def test_offload_model_weights_missing_parameter(self, mock_logger, memory_manager, mock_model):
        """Test offload_model_weights when parameter is missing in cpu_model."""
        incomplete_cpu_model = {"layer1.weight": MagicMock()}  # Missing layer2.bias
        
        with patch.dict("os.environ", {"RANK": "0"}):
            with pytest.raises(RuntimeError, match="Cannot offload parameter"):
                memory_manager.offload_model_weights(mock_model, incomplete_cpu_model)

    @patch("agentic_rl.runner.infer_adapter.vllm.memory_manager.logger")
    def test_offload_model_weights_parameter_assignment_fails(self, mock_logger, memory_manager, mock_cpu_model):
        """Test offload_model_weights when parameter assignment fails."""
        model = MagicMock()
        param = MagicMock()
        # Make data assignment raise an exception
        type(param).data = property(lambda self: None, lambda self, val: (_ for _ in ()).throw(Exception("Assignment failed")))
        model.named_parameters.return_value = [("layer1.weight", param)]
        
        with patch.dict("os.environ", {"RANK": "0"}):
            with pytest.raises(RuntimeError, match="Unexpected error: failed to offload parameter"):
                memory_manager.offload_model_weights(model, mock_cpu_model)

