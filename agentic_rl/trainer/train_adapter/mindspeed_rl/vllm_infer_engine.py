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

import os
from typing import Any, Dict, List
from transformers import AutoConfig
import torch
import torch.distributed
from mindspeed_rl.models.rollout.vllm_adapter.vllm_parallel_state import initialize_parallel_state
# [do not delete!!!] vLLM Ascend must be patched in advance
from vllm_ascend.patch import platform, worker  # noqa: F401

from agentic_rl.runner.infer_adapter.vllm.patch import apply_patch
from agentic_rl.runner.infer_adapter.vllm.cache_manager import CacheManager
from agentic_rl.runner.infer_adapter.vllm.memory_manager import MemoryManager
from agentic_rl.runner.infer_adapter.vllm.weight_manager import WeightManager
from agentic_rl.base.utils.get_local_rank import get_local_rank

apply_patch()

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.checker import validate_params
from agentic_rl.runner.infer_adapter.vllm.vllm_worker import AsyncBaseVLLMInferEngine


def dummy_compile(*compile_args, **compile_kwargs):
    def decorate(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorate


torch.jit.script = dummy_compile
torch.compile = dummy_compile

logger = Loggers(__name__)

# Constants
DEFAULT_LOAD_FORMAT = "megatron"
SLEEP_LEVEL = 2
KV_CACHE_TAG = ["kv_cache"]
CUDA_TIMER_STREAM_KAFKA_ENABLE = '0'
MEGATRON_IMPORT_TIMERS = '0'


class AsyncVLLMInferEngine(AsyncBaseVLLMInferEngine):
    """
    Asynchronous VLLM inference engine with sleep/wake functionality.

    This class manages VLLM inference with support for model sleeping (offloading weights)
    and waking (reloading weights) to optimize memory usage.
    """

    def __init__(self, enable_sleep_mode: bool, load_format: str = DEFAULT_LOAD_FORMAT, *args, **kwargs):
        super().__init__(enable_sleep_mode, load_format, *args, **kwargs)

        # Core managers
        self.cache_manager = CacheManager()
        self.memory_manager = MemoryManager()

        try:
            self.weight_manager = WeightManager(
                infer_tensor_parallel_size=self.infer_tensor_parallel_size,
                infer_pipeline_parallel_size=self.infer_pipeline_parallel_size,
                infer_expert_parallel_size=self.infer_expert_parallel_size,
                load_format=load_format or DEFAULT_LOAD_FORMAT
            )
        except RuntimeError as e:
            logger.error(f"Failed to initialize WeightManager with load_format={load_format}: {e}")
            raise RuntimeError(f"WeightManager initialization failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error initializing WeightManager: {e}")
            raise RuntimeError(f"Unexpected error during WeightManager initialization: {e}") from e

    @staticmethod
    def _setup_compilation():
        """Set up torch compilation settings."""
        torch.compile = dummy_compile

    @validate_params(
        all_kwargs=dict(
            validator=lambda x: x and isinstance(x, list) and all(isinstance(item, dict) for item in x),
            message="all_kwargs must be a list of dictionaries",
        ),
    )
    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """
        Initialize the worker engine with configuration and dependencies.

        Args:
            all_kwargs: List containing configuration dictionaries for initialization
        """
        self._setup_worker_config(all_kwargs)
        self._setup_compilation()
        self._setup_tokenizer_and_config()
        self._setup_parallel_state()
        self._setup_weight_loader()
        self._create_inference_engine(all_kwargs)

        logger.info("Worker initialization completed successfully")

    def init_cache_engine(self):
        """Initialize the cache engine for the worker."""
        try:
            if self.inference_engine is None:
                raise RuntimeError("Inference engine is not initialized")
            self.cache_manager.init_cache_engine(self.inference_engine)
        except RuntimeError as e:
            logger.error(f"Failed to initialize cache engine in AsyncBaseVLLMInferEngine: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing cache engine: {e}")
            raise RuntimeError(f"Unexpected error during cache engine initialization: {e}") from e

    def free_cache_engine(self):
        """
        Free cache engine resources and track memory statistics.
        """
        try:
            if self.inference_engine is None:
                raise RuntimeError("Inference engine is not initialized")
            if self.model is None:
                logger.warning("Model is None, skipping cache engine free")
            else:
                self.cache_manager.free_cache_engine(self.inference_engine, self.model)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to free cache engine: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error freeing cache engine: {e}") from e

    def offload_model_weights(self):
        """
        Offload model weights to CPU memory.
        """
        try:
            if self.model is None:
                raise RuntimeError("Model is not initialized")
            if self.memory_manager.cpu_model is None:
                raise RuntimeError("CPU model is not initialized")
            logger.info("Starting model weight offload to CPU")
            self.memory_manager.offload_model_weights(self.model, self.memory_manager.cpu_model)
            logger.info("Model weight offload completed successfully")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to offload model weights: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error offloading model weights: {e}") from e
    
    @validate_params(
        params=dict(validator=lambda x: isinstance(x, dict), message="params must be a dictionary"),
        load_format=dict(validator=lambda x: isinstance(x, str), message="load_format must be a string"),
    )
    def sync_model_weights(self, params, load_format='megatron'):
        """Synchronize model weights with the given parameters."""
        try:
            if self.model is None:
                raise RuntimeError("Model is not initialized")
            if self.hf_config is None:
                raise RuntimeError("HuggingFace config is not initialized")
            if not params:
                raise ValueError("params dictionary is empty")
            logger.info(f"Starting model weight synchronization with format={load_format}")
            self.weight_manager.load_megatron_weights(params, self.model, self.hf_config)
            logger.info("Model weight synchronization completed successfully")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Weight synchronization failed due to invalid parameters: {e}") from e
        except RuntimeError as e:
            raise RuntimeError(f"Failed to synchronize model weights: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error synchronizing model weights: {e}") from e

    def _setup_tokenizer_and_config(self):
        try:
            self.hf_config = AutoConfig.from_pretrained(
                self.tokenizer_name_or_path,
                trust_remote_code=self.trust_remote_code,
                local_files_only=True,
                weights_only=True
            )
        except (OSError, IOError) as e:
            raise RuntimeError(f"HuggingFace config loading failed: {e}") from e
        except RuntimeError as e:
            raise RuntimeError(f"HuggingFace config loading failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading HuggingFace config: {e}") from e

        self.local_rank = get_local_rank(logger_name="vllm_engine")

    def _setup_weight_loader(self):
        try:
            self.weight_manager.initialize_weight_loader()
        except (RuntimeError, ValueError, AttributeError) as e:
            raise RuntimeError(f"Weight loader initialization failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during weight loader initialization: {e}") from e

    def load_model(self, *args, **kwargs):
        """Load the model into the inference engine."""
        try:
            self.inference_engine.load_model(*args, **kwargs)
        except (RuntimeError, MemoryError, OSError, IOError) as e:
            raise RuntimeError(f"Model loading failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during model loading: {e}") from e

        try:
            self.model = self.inference_engine.worker.model_runner.get_model()
        except AttributeError as e:
            raise RuntimeError(f"Model access failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during model access: {e}") from e

        try:
            self.memory_manager.cpu_model = self.memory_manager.create_cpu_model_copy(self.model)
        except (RuntimeError, MemoryError) as e:
            raise RuntimeError(f"CPU model copy creation failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during CPU model copy creation: {e}") from e

        self.cache_manager.kv_cache_configs = None
        logger.info("Model loaded successfully")

    def sleep(self, *args, **kwargs):
        """Put the inference engine to sleep by offloading weights or freeing cache."""
        if self.enable_sleep_mode:
            logger.info("Entering inference engine sleep mode")
            try:
                if self.inference_engine is None:
                    raise RuntimeError("Inference engine is not initialized")
                self.inference_engine.sleep(level=SLEEP_LEVEL)
            except (RuntimeError, AttributeError) as e:
                logger.error(f"Failed to put inference engine to sleep: {e}")
                raise RuntimeError(f"Inference engine sleep failed: {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error during inference engine sleep: {e}")
                raise RuntimeError(f"Unexpected error during sleep operation: {e}") from e

            self.memory_manager.clear_gpu_memory()
            self.is_sleep = True
            logger.info("Inference engine sleep mode completed successfully")
        else:
            logger.info("Entering cache engine free mode")
            try:
                self.free_cache_engine()
                self.is_sleep = True
                logger.info("Cache engine free mode completed successfully")
            except RuntimeError as e:
                raise RuntimeError(f"Failed to free cache engine during sleep: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error during cache free operation: {e}") from e

    def wake_up(self, *args, **kwargs):
        """Wake up the inference engine by loading weights and building KV cache."""
        if self.enable_sleep_mode:
            logger.info("Entering inference engine wake up mode")
            try:
                if self.inference_engine is None:
                    raise RuntimeError("Inference engine is not initialized")
                self.inference_engine.wake_up(tags=KV_CACHE_TAG)
                self.is_sleep = False
                logger.info("Inference engine wake up completed successfully")
            except (RuntimeError, AttributeError) as e:
                raise RuntimeError(f"Failed to wake up inference engine: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error during inference engine wake up: {e}") from e
        else:
            if self.first_wake_up:
                logger.info("First wake up - initializing KV caches")
                self.free_cache_engine()

                if self.inference_engine is None or self.inference_engine.worker is None:
                    raise RuntimeError("Inference engine or worker is not initialized")
                try:
                    vllm_config = self.inference_engine.worker.vllm_config
                    self._initialize_kv_caches(vllm_config)
                except (RuntimeError, AttributeError) as e:
                    raise RuntimeError(f"Failed to initialize KV caches during first wake up: {e}") from e
                except Exception as e:
                    raise RuntimeError(f"Unexpected error during first wake up KV cache initialization: {e}") from e
                self.first_wake_up = False

            logger.info("Initializing cache engine")
            self.init_cache_engine()
            self.is_sleep = False
            logger.info("Cache engine initialization completed successfully")

    def _setup_parallel_state(self):
        self._log_parallel_config()

        if self.train_tensor_parallel_size is not None:
            self._configure_parallel_environment()

            try:
                initialize_parallel_state(
                    infer_tensor_model_parallel_size=self.infer_tensor_parallel_size,
                    train_tensor_model_parallel_size=self.train_tensor_parallel_size,
                    infer_pipeline_model_parallel_size=self.infer_pipeline_parallel_size,
                    train_pipeline_model_parallel_size=self.train_pipeline_parallel_size,
                    train_expert_model_parallel_size=self.train_expert_parallel_size,
                    infer_expert_model_parallel_size=self.infer_expert_parallel_size,
                    train_context_model_parallel_size=self.train_context_parallel_size,
                )
            except (RuntimeError, ValueError) as e:
                logger.error(
                    f"Failed to initialize parallel state with config "
                    f"(infer_tp={self.infer_tensor_parallel_size}, train_tp={self.train_tensor_parallel_size}): {e}")
                raise RuntimeError(f"Parallel state initialization failed: {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error initializing parallel state: {e}")
                raise RuntimeError(f"Unexpected error during parallel state initialization: {e}") from e
    
    def _log_parallel_config(self):
        logger.info(f"self.infer_tensor_parallel_size={self.infer_tensor_parallel_size}, "
                    f"self.train_tensor_parallel_size={self.train_tensor_parallel_size}, "
                    f"self.infer_pipeline_parallel_size={self.infer_pipeline_parallel_size}, "
                    f"self.train_pipeline_parallel_size={self.train_pipeline_parallel_size}, "
                    f"self.train_expert_parallel_size={self.train_expert_parallel_size}, "
                    f"self.infer_expert_parallel_size={self.infer_expert_parallel_size}, "
                    f"self.train_context_parallel_size={self.train_context_parallel_size}")

    def _configure_parallel_environment(self):
        os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = CUDA_TIMER_STREAM_KAFKA_ENABLE
        os.environ['MEGATRON_IMPORT_TIMERS'] = MEGATRON_IMPORT_TIMERS

    def _initialize_kv_caches(self, vllm_config):
        """
        Initialize KV caches for the model.

        Args:
            vllm_config: VLLM configuration object
        """
        if vllm_config is None:
            raise ValueError("vllm_config cannot be None")
        if self.inference_engine is None:
            raise RuntimeError("Inference engine is not initialized")
        try:
            self.cache_manager.initialize_kv_caches(self.inference_engine, vllm_config)
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Failed to initialize KV caches: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error initializing KV caches: {e}") from e