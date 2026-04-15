#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------


# Standard library imports
import gc
import os
from typing import Any, Dict, List, Union

# Third-party library imports
import torch
import torch.distributed
from transformers import AutoConfig

# vLLM imports
from vllm.worker.worker_base import WorkerWrapperBase, set_current_vllm_config
from vllm.v1.core.kv_cache_utils import get_kv_cache_config, unify_kv_cache_configs

# Internal imports
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.get_local_rank import get_local_rank
from agentic_rl.base.utils.tokenizer import get_tokenizer
from agentic_rl.base.weight_loaders.megatron_weight_loaders import InferParallelConfig
from agentic_rl.runner.infer_adapter.vllm.base_inference_engine import BaseInferEngine
from agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders import VllmMegatronWeightLoaders
from agentic_rl.runner.infer_adapter.vllm.vllm_parallel_state import initialize_parallel_state

logger = Loggers(__name__).get_logger()


class AgentWorkerWrapperBase(WorkerWrapperBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_cache_configs = None

    def initialize_from_config(self, kv_cache_configs) -> None:
        if self.kv_cache_configs is None:
            self.kv_cache_configs = kv_cache_configs
        super().initialize_from_config(self.kv_cache_configs)


def get_device_memory():
    from vllm_ascend.platform import NPUPlatform
    free_bytes_after_free, total = NPUPlatform.mem_get_info()
    used_bytes = total - free_bytes_after_free
    gib_bytes = 1 << 30
    return used_bytes / gib_bytes


def print_memory(content, condition=True):
    if condition:
        # torch.cuda.empty_cache()
        logger.info(
            f'{content} '
            f'torch allocated {torch.cuda.memory_allocated() / 1e9:.4f} GB, '
            f'reserved {torch.cuda.memory_reserved() / 1e9:.4f} device-memory-used: {get_device_memory()}GB')

def swap_tensor_to_host(device_tensor):
    cpu_tensor = torch.empty_like(device_tensor, device='cpu')
    if device_tensor.storage().size() != 0:
        with torch.no_grad():
            cpu_tensor.copy_(device_tensor, non_blocking=False)
        torch.cuda.empty_cache()
        device_tensor.storage().resize_(0)
    return cpu_tensor

class AsyncVLLMInferEngine(BaseInferEngine):
    def __init__(self, enable_sleep_mode: bool, load_format: str = "megatron", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_cache_configs = None
        self.cpu_model = None
        self.model = None
        self.local_rank = None
        self.pad_token_id = None
        self.tokenizer = None
        self.hf_config = None
        self.vllm_config = None
        self.load_format = load_format
        self.inference_engine: WorkerWrapperBase = None
        self.is_sleep = True
        self.first_wake_up = True
        self.freed_bytes: float = 0
        self.used_bytes: float = 0
        self.enable_sleep_mode = enable_sleep_mode
        self.vllm_megatron_weight_loaders = VllmMegatronWeightLoaders()

        # [do not delete!!!] vLLM Ascend must be patched in advance
        from vllm_ascend.patch import platform
        from vllm_ascend.patch import worker

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = int(os.environ["LOCAL_RANK"])
        self.vllm_config = all_kwargs[0]["vllm_config"]

        logger.debug(f"rank={int(os.environ['RANK'])}")

        # torch.compile = dummy_compile

        self.hf_config = AutoConfig.from_pretrained(
            self.tokenizer_name_or_path,
            trust_remote_code=self.trust_remote_code
        )

        self.tokenizer = get_tokenizer(self.tokenizer_name_or_path,
                                       prompt_type=self.prompt_type, prompt_type_path=self.prompt_type_path)
        self.pad_token_id = (
            self.tokenizer.tokenizer.pad_token_id if self.tokenizer.tokenizer.pad_token_id is not None
            else self.tokenizer.tokenizer.eos_token_id)

        # Set up local rank using the helper function
        self.local_rank = get_local_rank(logger_name="vllm_engine")

        logger.info(f"self.infer_tensor_parallel_size={self.infer_tensor_parallel_size}, "
                    f"self.train_tensor_parallel_size={self.train_tensor_parallel_size}, "
                    f"self.infer_pipeline_parallel_size={self.infer_pipeline_parallel_size}, "
                    f"self.train_pipeline_parallel_size={self.train_pipeline_parallel_size}, "
                    f"self.train_expert_parallel_size={self.train_expert_parallel_size}, "
                    f"self.infer_expert_parallel_size={self.infer_expert_parallel_size}, "
                    f"self.train_context_parallel_size={self.train_context_parallel_size}")

        # Initialize parallel state if tensor parallel size is specified
        if self.train_tensor_parallel_size is not None:
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            with set_current_vllm_config(self.vllm_config):
                initialize_parallel_state(
                    infer_tensor_model_parallel_size=self.infer_tensor_parallel_size,
                    train_tensor_model_parallel_size=self.train_tensor_parallel_size,
                    infer_pipeline_model_parallel_size=self.infer_pipeline_parallel_size,
                    train_pipeline_model_parallel_size=self.train_pipeline_parallel_size,
                    train_expert_model_parallel_size=self.train_expert_parallel_size,
                    infer_expert_model_parallel_size=self.infer_expert_parallel_size,
                    train_context_model_parallel_size=self.train_context_parallel_size
                )

        if self.load_format == "megatron":
            self.vllm_megatron_weight_loaders.update_megatron_weight_loader()

        self.inference_engine = AgentWorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        self.model = self.inference_engine.worker.model_runner.get_model()
        self.cpu_model = {}
        for name, params in self.model.named_parameters():
            self.cpu_model[name] = torch.empty_like(params, device="cpu")

        self.kv_cache_configs = None

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):

        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            result = self.inference_engine.execute_method(method, *args, **kwargs)
            return result

    def init_cache_engine(self):
        self.inference_engine.initialize_from_config(None)

    from vllm.config import VllmConfig

    def _initialize_kv_caches(self, vllm_config: VllmConfig):

        # Get all kv cache needed by the model
        kv_cache_specs = self.inference_engine.get_kv_cache_spec()

        # Profiles the peak memory usage of the model to determine how much
        # memory can be allocated for kv cache.
        available_gpu_memory = self.inference_engine.determine_available_memory()

        # Get the kv cache tensor size
        self.kv_cache_configs = [
            get_kv_cache_config(vllm_config, kv_cache_spec_one_worker,
                                available_gpu_memory_one_worker)
            for kv_cache_spec_one_worker, available_gpu_memory_one_worker in
            zip([kv_cache_specs], [available_gpu_memory])
        ]

        # Since we use a shared centralized controller, we need the
        # `kv_cache_config` to be consistent across all workers to make sure
        # all the memory operators can be applied to all workers.
        unify_kv_cache_configs(self.kv_cache_configs)

        # All workers have the same kv_cache_config except layer names, so use
        # an arbitrary one to initialize the scheduler.
        if not all([
            cfg.num_blocks == self.kv_cache_configs[0].num_blocks
            for cfg in self.kv_cache_configs
        ]):
            raise ValueError("All kv_cache_configs must have the same num_blocks")

    def free_cache_statistic(self):
        gib_bytes = 1 << 30
        freed_bytes_gib = (self.freed_bytes / gib_bytes)
        used_bytes_gib = (self.used_bytes / gib_bytes)
        logger.info(f"Free cache, freed {freed_bytes_gib:,.2f} GiB memory, "
                    f"{used_bytes_gib:,.2f} GiB memory is still in use===")

    def _free_cache_engine(self):
        worker = self.inference_engine.worker
        ctx = worker.model_runner.vllm_config.compilation_config.static_forward_context
        from vllm.attention import AttentionType
        layer_need_kv_cache = []
        for layer_name in ctx:
            if  (hasattr(ctx[layer_name], 'attn_type') and ctx[layer_name].attn_type in
                    (AttentionType.DECODER, AttentionType.ENCODER_DECODER)):
                layer_need_kv_cache.append(layer_name)

        pipeline_parallel_size = worker.vllm_config.parallel_config.pipeline_parallel_size
        for layer_name in layer_need_kv_cache:
            kv_cache = []
            for _ in range(pipeline_parallel_size):
                kv_cache.append(torch.tensor([]))
            ctx[layer_name].kv_cache = kv_cache
        # clear kv cache
        worker.model_runner.kv_caches = []

        if hasattr(self.model, 'model') and hasattr(self.model.model.layers[0].self_attn, "attn"):
            for i in range(self.model.model.start_layer, self.model.model.end_layer):
                attn_impl = self.model.model.layers[i].self_attn.attn.impl
                if hasattr(attn_impl, "key_cache"):
                    attn_impl.key_cache = None
                    attn_impl.value_cache = None
        # clear kv cache
        elif hasattr(self.model, 'language_model') and hasattr(self.model.language_model.model.layers[0].self_attn,
                                                               "attn"):
            for i in range(self.model.language_model.model.start_layer, self.model.language_model.model.end_layer):
                attn_impl = self.model.language_model.model.layers[i].self_attn.attn.impl
                if hasattr(attn_impl, "key_cache"):
                    attn_impl.key_cache = None
                    attn_impl.value_cache = None
        torch.cuda.empty_cache()
        gc.collect()

    def free_cache_engine(self):
        from vllm_ascend.platform import NPUPlatform
        free_bytes_before_free = NPUPlatform.mem_get_info()[0]
        self._free_cache_engine()
        free_bytes_after_free, total = NPUPlatform.mem_get_info()
        self.freed_bytes = free_bytes_after_free - free_bytes_before_free
        self.used_bytes = total - free_bytes_after_free
        self.free_cache_statistic()

    def offload_model_weights(self):
        logger.info(f"offload model weights, rank={int(os.environ['RANK'])}")
        if self.enable_sleep_mode:
            self.inference_engine.sleep(level=1)
            return
        for name, params in self.model.named_parameters():
            params.data = self.cpu_model[name]
        if hasattr(self.model, 'model') and hasattr(self.model.model.layers[-1].self_attn, "mla_attn"):
            for i in range(self.model.model.start_layer, self.model.model.end_layer):
                mla = self.model.model.layers[i].self_attn.mla_attn.impl
                if hasattr(mla, "w_kc"):
                    mla.w_kc = None
                    mla.w_vc = None
                if hasattr(mla, "W_UV"):
                    mla.W_UV = None
                    mla.W_UK_T = None
        pass
    
    def sync_model_weights(self, params, load_format='megatron'):
        logger.info(f"sync model weights")
        infer_parallel_config = InferParallelConfig(self.infer_tensor_parallel_size, self.infer_pipeline_parallel_size,
                                                    self.infer_expert_parallel_size * self.infer_tensor_parallel_size)
        if self.enable_sleep_mode:
            print_memory("before sync_model_weights")

            backup_tensors = {}
            for name, pms in self.model.named_parameters():
                backup_tensors[name] = pms.data
                pms.data = self.cpu_model[name]
            self.vllm_megatron_weight_loaders.load_megatron_weights(params,
                              self.model,
                              infer_parallel_config,
                              self.hf_config)
            # synchronized model load
            torch.cuda.empty_cache()
            print_memory("sync_model_weights after load_megatron_weights")

            # wakeup, to keep the address of weight tensors (support aclgraph)
            # todo this will double the weights buffer useage
            self.inference_engine.wake_up(tags=["weights"])
            print_memory("sync_model_weights after wake_up")

            for name, pms in self.model.named_parameters():
                if name in backup_tensors:
                    backup_tensors[name].copy_(pms.data)
                    pms.data = backup_tensors[name]
                    torch.cuda.empty_cache()
            print_memory(f"sync_model_weights after weights copy")

            if hasattr(self.model, 'model') and hasattr(self.model.model.layers[0].self_attn, "mla_attn"):
                self._process_mla()
            return

        self.vllm_megatron_weight_loaders.load_megatron_weights(params,
                                                                self.model,
                                                                infer_parallel_config,
                                                                self.hf_config)
        if hasattr(self.model, 'model') and hasattr(self.model.model.layers[0].self_attn, "mla_attn"):
            self._process_mla()

    def _process_mla(self):
        for i in range(self.model.model.start_layer, self.model.model.end_layer):
            mla = self.model.model.layers[i].self_attn.mla_attn.impl
            if hasattr(mla, "w_kc"):
                mla.w_kc = None
                mla.w_vc = None
            if hasattr(mla, "W_UV"):
                mla.W_UV = None
                mla.W_UK_T = None
            mla.process_weights_after_loading(None)

    def generate_sequences(self, prompts=None, sampling_params=None, prompt_token_ids=None, use_tqdm=None, **kwargs):
        pass

    def sleep(self, *args, **kwargs):
        print_memory("before sleep")
        """Offload model weights and discard kv cache."""
        if self.enable_sleep_mode:
            logger.info(f"into inference engine sleep")
            self.inference_engine.sleep(level=2)
            torch.cuda.empty_cache()
            gc.collect()
        else:
            logger.info(f"into free_cache_engine")
            self.free_cache_engine()
        print_memory("after sleep")

    def wake_up(self, *args, **kwargs):
        print_memory("before wake_up")
        """Load model weights and build kv cache."""
        if self.enable_sleep_mode:
            logger.info(f"into inference engine wake_up")
            self.inference_engine.wake_up(tags=["kv_cache"])
        else:
            if self.first_wake_up:
                logger.info(f"into first wake up, init kv caches")
                self.free_cache_engine()
                self._initialize_kv_caches(self.inference_engine.worker.vllm_config)
                self.first_wake_up = False
            logger.info(f"into init_cache_engine")
            self.init_cache_engine()
        print_memory("after wake_up")