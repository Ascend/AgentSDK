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
from unittest.mock import MagicMock, patch
import pytest


class TestCacheManager:
    """Test suite for CacheManager class."""

    @pytest.fixture
    def mock_vllm_environment(self, monkeypatch):
        """Fixture to mock all vLLM dependencies in isolated scope."""
        # Create mock base classes
        class MockWorkerWrapperBase:
            def get_kv_cache_spec(self):
                pass
            
            def determine_available_memory(self):
                pass
            
            def initialize_from_config(self, configs):
                pass

        class MockVllmConfig:
            pass

        # Mock vLLM modules
        mock_vllm = MagicMock()
        mock_vllm_config = MagicMock()
        mock_vllm_config.VllmConfig = MockVllmConfig
        
        mock_worker_base = MagicMock()
        mock_worker_base.WorkerWrapperBase = MockWorkerWrapperBase
        
        # Install mocks using monkeypatch (isolated to this test scope)
        monkeypatch.setitem(sys.modules, "vllm", mock_vllm)
        monkeypatch.setitem(sys.modules, "vllm.config", mock_vllm_config)
        monkeypatch.setitem(sys.modules, "vllm.worker.worker_base", mock_worker_base)
        monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", MagicMock())
        monkeypatch.setitem(sys.modules, "vllm.attention", MagicMock())
        monkeypatch.setitem(sys.modules, "vllm_ascend", MagicMock())
        monkeypatch.setitem(sys.modules, "vllm_ascend.platform", MagicMock())
        
        # Clean import of CacheManager to pick up mocked dependencies
        monkeypatch.delitem(sys.modules, "agentic_rl.runner.infer_adapter.vllm.cache_manager", raising=False)
        
        from agentic_rl.runner.infer_adapter.vllm.cache_manager import CacheManager
        
        return {
            "CacheManager": CacheManager,
            "MockWorkerWrapperBase": MockWorkerWrapperBase,
            "MockVllmConfig": MockVllmConfig,
        }

    @pytest.fixture
    def cache_manager(self, mock_vllm_environment):
        """Fixture providing a CacheManager instance."""
        CacheManager = mock_vllm_environment["CacheManager"]
        return CacheManager()

    @pytest.fixture
    def mock_inference_engine(self, mock_vllm_environment):
        """Fixture providing a mock inference engine."""
        MockWorkerWrapperBase = mock_vllm_environment["MockWorkerWrapperBase"]
        
        engine = MagicMock(spec=MockWorkerWrapperBase)
        engine.__class__ = MockWorkerWrapperBase
        
        worker = MagicMock()
        worker.model_runner.kv_caches = []
        worker.vllm_config = MagicMock()
        worker.vllm_config.parallel_config.pipeline_parallel_size = 2
        
        engine.worker = worker
        engine.get_kv_cache_spec.return_value = MagicMock()
        engine.determine_available_memory.return_value = 1024 * 1024 * 1024
        
        return engine

    @pytest.fixture
    def mock_vllm_config(self, mock_vllm_environment):
        """Fixture providing a mock VllmConfig."""
        MockVllmConfig = mock_vllm_environment["MockVllmConfig"]
        
        config = MagicMock(spec=MockVllmConfig)
        config.__class__ = MockVllmConfig
        
        return config

    def test_init(self, cache_manager):
        """Test CacheManager initialization."""
        assert cache_manager.kv_cache_configs is None
        assert cache_manager.freed_bytes == 0.0
        assert cache_manager.used_bytes == 0.0

    @pytest.mark.parametrize("invalid_engine,invalid_config,expected_error", [
        ("invalid_engine", None, (ValueError, TypeError)),
        (None, "invalid_config", (ValueError, TypeError)),
    ])
    def test_initialize_kv_caches_invalid_params(
        self, cache_manager, mock_inference_engine, mock_vllm_config,
        invalid_engine, invalid_config, expected_error
    ):
        """Test initialize_kv_caches with invalid parameters."""
        engine = invalid_engine if invalid_engine else mock_inference_engine
        config = invalid_config if invalid_config else mock_vllm_config
        
        with pytest.raises(expected_error):
            cache_manager.initialize_kv_caches(engine, config)

    @pytest.mark.parametrize("error_method,error_type,error_msg,expected_match", [
        ("get_kv_cache_spec", AttributeError, "Mock error", "KV cache specification retrieval failed"),
        ("determine_available_memory", RuntimeError, "Mock memory error", "Available memory determination failed"),
    ])
    @patch("agentic_rl.runner.infer_adapter.vllm.cache_manager.logger")
    def test_initialize_kv_caches_errors(
        self, mock_logger, cache_manager, mock_inference_engine, mock_vllm_config,
        error_method, error_type, error_msg, expected_match
    ):
        """Test initialize_kv_caches error handling."""
        getattr(mock_inference_engine, error_method).side_effect = error_type(error_msg)
        
        with pytest.raises(RuntimeError, match=expected_match):
            cache_manager.initialize_kv_caches(mock_inference_engine, mock_vllm_config)

    @patch("agentic_rl.runner.infer_adapter.vllm.cache_manager.logger")
    def test_initialize_kv_caches_inconsistent_num_blocks(
        self, mock_logger, cache_manager, mock_inference_engine, mock_vllm_config
    ):
        """Test initialize_kv_caches with inconsistent num_blocks across workers."""
        mock_kv_cache_config1 = MagicMock()
        mock_kv_cache_config1.num_blocks = 100
        mock_kv_cache_config2 = MagicMock()
        mock_kv_cache_config2.num_blocks = 200

        cache_manager.kv_cache_configs = [mock_kv_cache_config1, mock_kv_cache_config2]

        with pytest.raises(RuntimeError, match="KV cache configs are not consistent"):
            if not all(cfg.num_blocks == cache_manager.kv_cache_configs[0].num_blocks 
                      for cfg in cache_manager.kv_cache_configs):
                raise RuntimeError("KV cache configs are not consistent across all workers")

    @pytest.mark.parametrize("kv_caches_empty,should_initialize", [
        (True, True),
        (False, False),
    ])
    def test_init_cache_engine(
        self, cache_manager, mock_inference_engine, kv_caches_empty, should_initialize
    ):
        """Test cache engine initialization based on kv_caches state."""
        mock_inference_engine.worker.model_runner.kv_caches = [] if kv_caches_empty else [MagicMock()]
        mock_kv_cache_config = MagicMock()
        cache_manager.kv_cache_configs = [mock_kv_cache_config]

        cache_manager.init_cache_engine(mock_inference_engine)

        if should_initialize:
            mock_inference_engine.initialize_from_config.assert_called_once_with([mock_kv_cache_config])
        else:
            mock_inference_engine.initialize_from_config.assert_not_called()

    def test_init_cache_engine_invalid_inference_engine(self, cache_manager):
        """Test init_cache_engine with invalid inference_engine parameter."""
        with pytest.raises((ValueError, TypeError)):
            cache_manager.init_cache_engine("invalid_engine")

    @patch("agentic_rl.runner.infer_adapter.vllm.cache_manager.logger")
    def test_init_cache_engine_attribute_error(
        self, mock_logger, cache_manager, mock_inference_engine
    ):
        """Test init_cache_engine when AttributeError is raised."""
        mock_inference_engine.worker.model_runner.kv_caches = []
        mock_inference_engine.initialize_from_config.side_effect = AttributeError("Mock attribute error")
        cache_manager.kv_cache_configs = [MagicMock()]

        with pytest.raises(RuntimeError, match="Cache engine initialization failed"):
            cache_manager.init_cache_engine(mock_inference_engine)

    @pytest.mark.parametrize("invalid_param,param_value,expected_error", [
        ("inference_engine", "invalid_engine", (ValueError, TypeError)),
        ("model", None, (ValueError, TypeError)),
    ])
    def test_free_cache_engine_invalid_params(
        self, cache_manager, mock_inference_engine, invalid_param, param_value, expected_error
    ):
        """Test free_cache_engine with invalid parameters."""
        engine = param_value if invalid_param == "inference_engine" else mock_inference_engine
        model = param_value if invalid_param == "model" else MagicMock()
        
        with pytest.raises(expected_error):
            cache_manager.free_cache_engine(engine, model)

    @patch("agentic_rl.runner.infer_adapter.vllm.cache_manager.logger")
    @patch("agentic_rl.runner.infer_adapter.vllm.cache_manager.torch")
    @patch("agentic_rl.runner.infer_adapter.vllm.cache_manager.gc")
    def test_free_cache_engine_cleanup_on_error(
        self, mock_gc, mock_torch, mock_logger, cache_manager, mock_inference_engine
    ):
        """Test that cleanup happens even when _free_cache_engine raises an error."""
        with patch("vllm_ascend.platform.NPUPlatform") as mock_npu_platform:
            mock_npu_platform.mem_get_info.return_value = (1000000000, 2000000000)
            mock_inference_engine.worker.model_runner.vllm_config.compilation_config.static_forward_context = None

            with pytest.raises(RuntimeError):
                cache_manager.free_cache_engine(mock_inference_engine, MagicMock())

            # Verify _current_worker was cleaned up in finally block
            assert cache_manager._current_worker is None
