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
from typing import Any, Dict, List, Union
from abc import abstractmethod
import torch
import torch.distributed
from vllm.worker.worker_base import WorkerWrapperBase

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.checker import validate_params
from agentic_rl.runner.infer_adapter.vllm.base_inference_engine import BaseInferEngine


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

_DEFAULT_FACTOR_FOR_QWEN2_5 = 4.0
_DEFAULT_MAX_POSITION_EMB_FOR_QWEN2_5 = 32768


class AsyncBaseVLLMInferEngine(BaseInferEngine):
    """
    Asynchronous VLLM inference engine with sleep/wake functionality.

    This class manages VLLM inference with support for model sleeping (offloading weights)
    and waking (reloading weights) to optimize memory usage.
    """

    @validate_params(
        enable_sleep_mode=dict(validator=lambda x: isinstance(x, bool), message="enable_sleep_mode must be a boolean"),
        load_format=dict(validator=lambda x: isinstance(x, str), message="load_format must be a string"),
    )
    def __init__(self, enable_sleep_mode: bool, load_format: str = DEFAULT_LOAD_FORMAT, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Model and inference state
        self.model = None
        self.inference_engine: WorkerWrapperBase = None
        self.hf_config = None
        self.vllm_config = None
        self.local_rank = None

        # Sleep/wake state management
        self.enable_sleep_mode = enable_sleep_mode
        self.is_sleep = True
        self.first_wake_up = True
        self.sharding_manager = None

    @staticmethod
    def _setup_compilation():
        """Set up torch compilation settings."""
        torch.compile = dummy_compile

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """
        Perform some necessary initialize procedure for agent engine.
        """
        pass

    @validate_params(
        method=dict(
            validator=lambda x: isinstance(x, str) or isinstance(x, bytes),
            message="method must be a string or bytes",
        ),
    )
    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        """Execute a method by name with dispatch to appropriate handler."""
        try:
            method_name = method.decode() if isinstance(method, bytes) else method
        except (AttributeError, UnicodeDecodeError) as e:
            logger.error(f"Failed to decode method name from bytes: {e}")
            raise RuntimeError(f"Method name decoding failed: {e}") from e

        method_dispatch = {"init_worker": self.init_worker, "load_model": self.load_model,
                           "sleep": self.sleep, "wake_up": self.wake_up}
        if method_name in method_dispatch:
            return method_dispatch[method_name](*args, **kwargs)
        else:
            try:
                if self.inference_engine is None:
                    raise RuntimeError("Inference engine is not initialized")
                return self.inference_engine.execute_method(method, *args, **kwargs)
            except AttributeError as e:
                logger.error(f"Inference engine missing execute_method or method '{method_name}' not found: {e}")
                raise RuntimeError(f"Failed to delegate method '{method_name}' to inference engine: {e}") from e
            except Exception as e:
                logger.error(f"Failed to execute delegated method '{method_name}': {e}")
                raise RuntimeError(f"Delegated method '{method_name}' execution failed: {e}") from e


    @validate_params(
        prompts=dict(validator=lambda x: isinstance(x, list), message="prompts must be a list"),
        sampling_params=dict(validator=lambda x: isinstance(x, dict), message="sampling_params must be a dictionary"),
        prompt_token_ids=dict(validator=lambda x: isinstance(x, list), message="prompt_token_ids must be a list"),
        use_tqdm=dict(validator=lambda x: isinstance(x, bool), message="use_tqdm must be a boolean"),
    )
    def generate_sequences(self, prompts=None, sampling_params=None, prompt_token_ids=None, use_tqdm=None, **kwargs):
        """Generate sequences using the inference engine."""
        pass

    def init_cache_engine(self):
        pass

    def free_cache_engine(self):
        pass

    def offload_model_weights(self):
        pass

    def sync_model_weights(self, params, load_format='megatron'):
        pass

    def _setup_worker_config(self, all_kwargs: List[Dict[str, Any]]):
        if "RANK" in os.environ:
            try:
                rank_value = int(os.environ["RANK"])
                all_kwargs[0]["rank"] = rank_value
            except (ValueError, IndexError, TypeError) as e:
                raise ValueError(f"RANK environment variable must be an integer: {e}") from e

            if all_kwargs[0]["rank"] < 0 or all_kwargs[0]["rank"] >= 8:
                raise ValueError("RANK environment variable must be in [0, 8)")

        if "LOCAL_RANK" in os.environ:
            try:
                local_rank_value = int(os.environ["LOCAL_RANK"])
                all_kwargs[0]["local_rank"] = local_rank_value
            except (ValueError, IndexError, TypeError) as e:
                raise ValueError(f"LOCAL_RANK environment variable must be an integer: {e}") from e

            if all_kwargs[0]["local_rank"] < 0 or all_kwargs[0]["local_rank"] >= 8:
                raise ValueError("RANK environment variable must be in [0, 8)")

        try:
            self.vllm_config = all_kwargs[0]["vllm_config"]
        except (IndexError, KeyError) as e:
            raise RuntimeError(f"Missing or invalid vllm_config in all_kwargs: {e}") from e

    def _create_inference_engine(self, all_kwargs: List[Dict[str, Any]]):
        try:
            self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        except (RuntimeError, MemoryError, TypeError) as e:
            raise RuntimeError(f"WorkerWrapperBase instantiation failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during WorkerWrapperBase creation: {e}") from e

        try:
            self.inference_engine.init_worker(all_kwargs)
        except (RuntimeError, MemoryError) as e:
            raise RuntimeError(f"Inference engine worker initialization failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during inference engine worker initialization: {e}") from e

    @abstractmethod
    def load_model(self, *args, **kwargs):
        """
        Perform some necessary initialize procedure for agent engine.
        """
        pass
 
    @abstractmethod
    def sleep(self, *args, **kwargs):
        """
        Perform some necessary initialize procedure for agent engine.
        """
        pass

    @abstractmethod
    def wake_up(self, *args, **kwargs):
        """
        Perform some necessary initialize procedure for agent engine.
        """
        pass
