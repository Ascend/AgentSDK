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
from typing import Any, Dict, List

import torch
import torch.distributed
from vllm_ascend.platform import NPUPlatform
# [do not delete!!!] vLLM Ascend must be patched in advance
from vllm_ascend.patch import platform, worker  # noqa: F401
from agentic_rl.runner.infer_adapter.vllm.patch import apply_patch
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

DEFAULT_LOAD_FORMAT = "megatron"


def get_device_memory():
    """
    Get the current device memory usage.
    
    This function retrieves the memory usage information from the NPU platform
    and returns the used memory in gigabytes.
    
    Returns:
        float: Used memory in gigabytes (GB)
    """
    free_bytes_after_free, total = NPUPlatform.mem_get_info()
    used_bytes = total - free_bytes_after_free
    gib_bytes = 1 << 30
    return used_bytes / gib_bytes


def print_memory(content, condition=True):
    """
    Print memory usage information with a custom message.
    
    Args:
        content (str): Custom message to include in the log
        condition (bool): Whether to print the memory information (default: True)
    """
    if condition:
        logger.info(
            f'{content} '
            f'torch allocated {torch.cuda.memory_allocated() / 1e9:.4f} GB, '
            f'reserved {torch.cuda.memory_reserved() / 1e9:.4f} device-memory-used: {get_device_memory()}GB')


class AsyncVLLMInferEngine(AsyncBaseVLLMInferEngine):
    """
    Asynchronous VLLM inference engine with sleep/wake functionality.

    This class manages VLLM inference with support for model sleeping (offloading weights)
    and waking (reloading weights) to optimize memory usage.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_sleep = False

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
        self._create_inference_engine(all_kwargs)

        logger.info("Worker initialization completed successfully")

    def load_model(self, *args, **kwargs):
        """Load the model into the inference engine."""
        print_memory("before load_model")
        try:
            self.inference_engine.load_model(*args, **kwargs)
        except (RuntimeError, MemoryError, OSError, IOError) as e:
            raise RuntimeError(f"Model loading failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during model loading: {e}") from e
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner
        logger.info("Model loaded successfully")
        print_memory("after load_model")

    def sleep(self, *args, **kwargs):
        """
        Put the inference engine to sleep by offloading model weights.
        
        """
        if self.is_sleep:
            return
        print_memory("before sleep")
        self.sharding_manager.__exit__(None, None, None)
        torch.cuda.empty_cache()
        gc.collect()
        self.is_sleep = True
        print_memory(f"after sleep")

    def wake_up(self, *args, **kwargs):
        """
        Wake up the inference engine by loading model weights and building KV cache.
        
        Raises:
            RuntimeError: If the sharding_manager is not initialized
        """
        if not self.is_sleep:
            return
        print_memory("before wake_up")
        if self.sharding_manager is None:
            raise RuntimeError("sharding_manager is not initialized")
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False
        print_memory(f"after wake_up")
        