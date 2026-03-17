#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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
from typing import Any, Dict, List, Union
import sys
import pytest
from agentic_rl.base.log.loggers import Loggers  # 保留真实 logger


class MockAsyncBaseVLLMInferEngine:
    def __init__(self, *args, **kwargs):
        self.enable_sleep_mode = True

    def _setup_worker_config(self, all_kwargs: List[Dict[str, Any]]):
        pass

    def _setup_compilation():
        pass

    def _create_inference_engine(self, all_kwargs: List[Dict[str, Any]]):
        pass


@pytest.fixture(autouse=True, scope="function")
def mock_vllm_module():
    """Fixture to mock only the necessary vLLM and related dependencies."""
    # Mock only the necessary parts
    with patch.dict(
        sys.modules,
        {
            "vllm": MagicMock(),
            "vllm_worker": MagicMock(),
            "vllm_ascend": MagicMock(),
            "vllm_ascend.platform": MagicMock(),
            "vllm_ascend.patch": MagicMock(),
            "agentic_rl.runner.infer_adapter.vllm.vllm_worker": MagicMock(),
        },
    ):
        yield


@pytest.fixture(autouse=True, scope="function")
def mock_vllm_environment(mock_vllm_module):
    """Fixture to mock only the necessary vLLM and related dependencies."""
    # Mock only the necessary parts
    with (
        patch("agentic_rl.runner.infer_adapter.vllm.patch.apply_patch"),
        patch("vllm_ascend.platform.NPUPlatform") as mock_npu_platform,
        patch(
            "agentic_rl.runner.infer_adapter.vllm.vllm_worker.AsyncBaseVLLMInferEngine",
            MockAsyncBaseVLLMInferEngine,
        ) as mock_base_engine,
        patch("torch.cuda.memory_allocated", return_value=512 * 1024 * 1024),
        patch("torch.cuda.memory_reserved", return_value=768 * 1024 * 1024),
        patch("torch.cuda.empty_cache"),
        patch("torch.jit.script"),
        patch("torch.compile"),
    ):
        mock_npu_platform.mem_get_info.return_value = (
            1024 * 1024 * 1024,
            2 * 1024 * 1024 * 1024,
        )
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


class TestAsyncVLLMInferEngineInitialization:
    """Test suite for AsyncVLLMInferEngine initialization."""

    def test_init_default_params(self, mock_vllm_environment, base_init_params):
        """Test initialization with default parameters."""
        from agentic_rl.trainer.train_adapter.verl.vllm_infer_engine import (
            AsyncVLLMInferEngine,
        )

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)

        assert engine.enable_sleep_mode is True


class TestAsyncVLLMInferEngineWorkerInit:
    """Test suite for AsyncVLLMInferEngine.init_worker method."""

    def test_init_worker_success(self, mock_vllm_environment, base_init_params):
        """Test successful worker initialization."""
        from agentic_rl.trainer.train_adapter.verl.vllm_infer_engine import (
            AsyncVLLMInferEngine,
        )

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)

        all_kwargs = [{"vllm_config": MagicMock()}]
        engine._setup_worker_config = MagicMock()
        engine._setup_compilation = MagicMock()

        def mock_create_inference_engine(all_kwargs):
            mock_engine = MagicMock()
            return mock_engine

        engine._create_inference_engine = MagicMock(
            side_effect=mock_create_inference_engine
        )

        engine.init_worker(all_kwargs)

        engine._setup_worker_config.assert_called_once_with(all_kwargs)
        engine._setup_compilation.assert_called_once()
        engine._create_inference_engine.assert_called_once_with(all_kwargs)

    def test_init_worker_invalid_params(self, mock_vllm_environment, base_init_params):
        """Test init_worker with invalid parameters."""
        from agentic_rl.trainer.train_adapter.verl.vllm_infer_engine import (
            AsyncVLLMInferEngine,
        )

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)

        # Test with invalid all_kwargs
        invalid_kwargs = ["not a dict"]
        with pytest.raises(
            ValueError, match="all_kwargs must be a list of dictionaries"
        ):
            engine.init_worker(invalid_kwargs)


class TestAsyncVLLMInferEngineLoadModel:
    """Test suite for AsyncVLLMInferEngine.load_model method."""

    def test_load_model_success(self, mock_vllm_environment, base_init_params):
        """Test successful model loading."""
        from agentic_rl.trainer.train_adapter.verl.vllm_infer_engine import (
            AsyncVLLMInferEngine,
        )

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = MagicMock()
        engine.sharding_manager = MagicMock()

        engine.load_model()

        engine.inference_engine.load_model.assert_called_once()
        assert engine.sharding_manager.inference_engine == engine.inference_engine
        assert (
            engine.sharding_manager.model_runner
            == engine.inference_engine.worker.model_runner
        )

    def test_load_model_inference_engine_errors(
        self, mock_vllm_environment, base_init_params
    ):
        """Test load_model with inference engine loading failures."""
        from agentic_rl.trainer.train_adapter.verl.vllm_infer_engine import (
            AsyncVLLMInferEngine,
        )

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.inference_engine = MagicMock()
        engine.inference_engine.load_model.side_effect = RuntimeError(
            "Model load failed"
        )

        with pytest.raises(RuntimeError, match="Model loading failed"):
            engine.load_model()


class TestAsyncVLLMInferEngineSleepWake:
    """Test suite for AsyncVLLMInferEngine sleep and wake_up methods."""

    def test_sleep(self, mock_vllm_environment, base_init_params):
        """Test sleep method."""
        from agentic_rl.trainer.train_adapter.verl.vllm_infer_engine import (
            AsyncVLLMInferEngine,
        )

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.sharding_manager = MagicMock()
        engine.is_sleep = False

        engine.sleep()

        engine.sharding_manager.__exit__.assert_called_once_with(None, None, None)
        assert engine.is_sleep is True

    def test_sleep_already_sleeping(self, mock_vllm_environment, base_init_params):
        """Test sleep method when already sleeping."""
        from agentic_rl.trainer.train_adapter.verl.vllm_infer_engine import (
            AsyncVLLMInferEngine,
        )

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.sharding_manager = MagicMock()
        engine.is_sleep = True

        engine.sleep()

        engine.sharding_manager.__exit__.assert_not_called()

    def test_wake_up(self, mock_vllm_environment, base_init_params):
        """Test wake_up method."""
        from agentic_rl.trainer.train_adapter.verl.vllm_infer_engine import (
            AsyncVLLMInferEngine,
        )

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.sharding_manager = MagicMock()
        engine.is_sleep = True

        engine.wake_up()

        engine.sharding_manager.__enter__.assert_called_once()
        assert engine.is_sleep is False

    def test_wake_up_already_awake(self, mock_vllm_environment, base_init_params):
        """Test wake_up method when already awake."""
        from agentic_rl.trainer.train_adapter.verl.vllm_infer_engine import (
            AsyncVLLMInferEngine,
        )

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.sharding_manager = MagicMock()
        engine.is_sleep = False

        engine.wake_up()

        engine.sharding_manager.__enter__.assert_not_called()

    def test_wake_up_no_sharding_manager(self, mock_vllm_environment, base_init_params):
        """Test wake_up method with no sharding_manager."""
        from agentic_rl.trainer.train_adapter.verl.vllm_infer_engine import (
            AsyncVLLMInferEngine,
        )

        engine = AsyncVLLMInferEngine(enable_sleep_mode=True, **base_init_params)
        engine.sharding_manager = None
        engine.is_sleep = True

        with pytest.raises(RuntimeError, match="sharding_manager is not initialized"):
            engine.wake_up()
