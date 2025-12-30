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

import gc
import torch
from vllm.config import VllmConfig
from vllm.worker.worker_base import WorkerWrapperBase

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.checker import validate_params


logger = Loggers(__name__)


class CacheManager:
    """Manages KV cache operations for VLLM inference engine."""

    GIB_BYTES = 1 << 30  # 1 GiB in bytes

    def __init__(self):
        self.kv_cache_configs = None
        self.freed_bytes = 0.0
        self.used_bytes = 0.0
        self._current_worker = None

    @staticmethod
    def _clear_standard_model_kv_cache(model):
        """Clear KV cache for standard model architecture."""
        for i in range(model.model.start_layer, model.model.end_layer):
            attn_impl = model.model.layers[i].self_attn.attn.impl
            if hasattr(attn_impl, "key_cache"):
                attn_impl.key_cache = None
                attn_impl.value_cache = None

    @staticmethod
    def _clear_language_model_kv_cache(model):
        """Clear KV cache for language model architecture."""
        for i in range(model.language_model.model.start_layer, model.language_model.model.end_layer):
            attn_impl = model.language_model.model.layers[i].self_attn.attn.impl
            if hasattr(attn_impl, "key_cache"):
                attn_impl.key_cache = None
                attn_impl.value_cache = None

    @validate_params(
        inference_engine=dict(
            validator=lambda x: isinstance(x, WorkerWrapperBase),
            message="inference_engine must be a WorkerWrapperBase instance",
        ),
        vllm_config=dict(
            validator=lambda x: isinstance(x, VllmConfig), message="vllm_config must be a VllmConfig instance"
        ),
    )
    def initialize_kv_caches(self, inference_engine, vllm_config: VllmConfig):
        """
        Initialize KV caches for the model.

        Args:
            inference_engine: The VLLM inference engine instance
            vllm_config: VLLM configuration object
        """
        from vllm.v1.core.kv_cache_utils import get_kv_cache_config, unify_kv_cache_configs

        # Get all kv cache needed by the model
        try:
            kv_cache_specs = inference_engine.get_kv_cache_spec()
        except (AttributeError, RuntimeError, MemoryError) as e:
            logger.error(f"Failed to get KV cache specifications from inference engine: {e}")
            raise RuntimeError(f"KV cache specification retrieval failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting KV cache specifications: {e}")
            raise RuntimeError(f"Unexpected error during KV cache specification retrieval: {e}") from e

        # Profiles the peak memory usage of the model to determine how much
        # memory can be allocated for kv cache.
        try:
            available_gpu_memory = inference_engine.determine_available_memory()
        except (AttributeError, RuntimeError, MemoryError) as e:
            logger.error(f"Failed to determine available GPU memory from inference engine: {e}")
            raise RuntimeError(f"Available memory determination failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error determining available GPU memory: {e}")
            raise RuntimeError(f"Unexpected error during memory determination: {e}") from e

        # Get the kv cache tensor size
        if self.kv_cache_configs is None:
            self.kv_cache_configs = []

        try:
            for kv_cache_spec_one_worker, available_gpu_memory_one_worker in zip(
                [kv_cache_specs], [available_gpu_memory]
            ):
                kv_cache_config = get_kv_cache_config(
                    vllm_config, kv_cache_spec_one_worker, available_gpu_memory_one_worker
                )
                self.kv_cache_configs.append(kv_cache_config)
        except (ValueError, RuntimeError, MemoryError) as e:
            logger.error(f"Failed to generate KV cache configuration: {e}")
            raise RuntimeError(f"KV cache configuration generation failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error generating KV cache configuration: {e}")
            raise RuntimeError(f"Unexpected error during KV cache configuration: {e}") from e

        # Since we use a shared centralized controller, we need the
        # `kv_cache_config` to be consistent across all workers to make sure
        # all the memory operators can be applied to all workers.
        try:
            unify_kv_cache_configs(self.kv_cache_configs)
        except (ValueError, RuntimeError) as e:
            logger.error(f"Failed to unify KV cache configurations: {e}")
            raise RuntimeError(f"KV cache configuration unification failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error unifying KV cache configurations: {e}")
            raise RuntimeError(f"Unexpected error during KV cache unification: {e}") from e

        # All workers have the same kv_cache_config except layer names, so use
        # an arbitrary one to initialize the scheduler.
        if not all([cfg.num_blocks == self.kv_cache_configs[0].num_blocks for cfg in self.kv_cache_configs]):
            raise RuntimeError("KV cache configs are not consistent across all workers")

    @validate_params(
        inference_engine=dict(
            validator=lambda x: isinstance(x, WorkerWrapperBase),
            message="inference_engine must be a WorkerWrapperBase instance",
        ),
    )
    def init_cache_engine(self, inference_engine):
        """
        Initialize the cache engine for the worker.

        Args:
            inference_engine: The VLLM inference engine instance
        """
        try:
            worker = inference_engine.worker
            if not worker.model_runner.kv_caches:
                # v1 uses explicit initialization method
                inference_engine.initialize_from_config(self.kv_cache_configs)
        except AttributeError as e:
            logger.error(f"Failed to access worker or model runner attributes: {e}")
            raise RuntimeError(f"Cache engine initialization failed due to missing attributes: {e}") from e
        except (RuntimeError, MemoryError) as e:
            logger.error(f"Failed to initialize cache engine: {e}")
            raise RuntimeError(f"Cache engine initialization failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during cache engine initialization: {e}")
            raise RuntimeError(f"Unexpected error during cache engine initialization: {e}") from e

    @validate_params(
        inference_engine=dict(
            validator=lambda x: isinstance(x, WorkerWrapperBase),
            message="inference_engine must be a WorkerWrapperBase instance",
        ),
        model=dict(validator=lambda x: x is not None, message="model must be a BaseModel instance"),
    )
    def free_cache_engine(self, inference_engine, model):
        """
        Free cache engine resources and track memory statistics.

        Args:
            inference_engine: The VLLM inference engine instance
            model: The model instance
        """
        from vllm_ascend.platform import NPUPlatform

        try:
            free_bytes_before_free = NPUPlatform.mem_get_info()[0]
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to get memory info before freeing cache: {e}")
            free_bytes_before_free = 0

        # Set worker context for _free_cache_engine
        self._current_worker = inference_engine.worker
        try:
            self._free_cache_engine(model)
        except RuntimeError as e:
            raise RuntimeError(f"Cache engine cleanup failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error: cache engine cleanup failed: {e}") from e
        finally:
            self._current_worker = None

        try:
            free_bytes_after_free, total = NPUPlatform.mem_get_info()
            self.freed_bytes = free_bytes_after_free - free_bytes_before_free
            self.used_bytes = total - free_bytes_after_free
            self._log_cache_statistics()
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to get memory info after freeing cache: {e}")
            # Set default values if memory info retrieval fails
            self.freed_bytes = 0.0
            self.used_bytes = 0.0

    def _free_cache_engine(self, model):
        """
        Internal method to free the cache engine resources.

        Args:
            model: The model instance
        """
        from vllm.attention import AttentionType

        # Get worker from inference engine (passed through context)
        worker = getattr(self, "_current_worker", None)
        if not worker:
            raise RuntimeError("Worker not set in cache manager context")

        ctx = worker.model_runner.vllm_config.compilation_config.static_forward_context
        layer_need_kv_cache = []
        for layer_name in ctx:
            if ctx[layer_name].attn_type in (AttentionType.DECODER, AttentionType.ENCODER_DECODER):
                layer_need_kv_cache.append(layer_name)

        pipeline_parallel_size = worker.vllm_config.parallel_config.pipeline_parallel_size
        for layer_name in layer_need_kv_cache:
            kv_cache = []
            for _ in range(pipeline_parallel_size):
                kv_cache.append(torch.tensor([]))
            ctx[layer_name].kv_cache = kv_cache

        # Clear kv cache
        worker.model_runner.kv_caches = []

        self._clear_model_kv_cache(model)
        torch.cuda.empty_cache()
        gc.collect()

    def _clear_model_kv_cache(self, model):
        """
        Clear KV cache from model layers.

        Args:
            model: The model instance
        """
        # Handle different model architectures
        if hasattr(model, "model") and hasattr(model.model.layers[0].self_attn, "attn"):
            self._clear_standard_model_kv_cache(model)
        elif hasattr(model, "language_model") and hasattr(model.language_model.model.layers[0].self_attn, "attn"):
            self._clear_language_model_kv_cache(model)

    def _log_cache_statistics(self):
        """Log cache memory statistics."""
        freed_bytes_gib = self.freed_bytes / self.GIB_BYTES
        used_bytes_gib = self.used_bytes / self.GIB_BYTES
        logger.info(
            f"===Free cache, freed {freed_bytes_gib:,.2f} GiB memory, "
            f"{used_bytes_gib:,.2f} GiB memory is still in use==="
        )
