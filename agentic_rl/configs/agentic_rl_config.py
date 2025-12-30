#!/usr/bin/env python3
# coding=utf-8
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

from dataclasses import dataclass, field
from agentic_rl.base.utils.checker import Checker
from agentic_rl.base.utils.file_utils import FileCheck


@dataclass
class AgenticRLConfig:
    namespace: str = "agentic_raygroup"
    max_steps: int = 5
    max_tool_length: int = 8192
    simplify_think_content: bool = False
    train_backend: str = "mindspeed_rl"
    enable_sleep_mode: bool = True
    load_format: str = "megatron"
    infer_backend: str = "vllm"

    agent_name: str = "default_agent"
    agent_engine_wrapper_path: str = "rllm"


@dataclass
class SamplingConfig:
    logprobs: int = 1
    max_tokens: int = 128
    top_p: float = 1.0
    top_k: int = 50
    min_p: float = 0.0
    temperature: float = 0.2
    detokenize: bool = False
    seed: int = None

    def __post_init__(self):
        Checker.validate_param("logprobs", int, self.logprobs)
        Checker.validate_param("max_tokens", int, self.max_tokens, 0)
        Checker.validate_param("top_p", float, self.top_p, 0)
        Checker.validate_param("top_k", int, self.top_k, 0)
        Checker.validate_param("min_p", float, self.min_p)
        Checker.validate_param("temperature", float, self.temperature)
        Checker.validate_param("detokenize", bool, self.detokenize)
        Checker.validate_param("seed", int, self.seed)


@dataclass
class GenConfig:
    limit_mm_image_per_prompt: int = 1
    limit_mm_video_per_prompt: int = 0
    data_parallel_size: int = None
    tokenizer_name_or_path: str = "./"
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    infer_tensor_parallel_size: int = 8
    infer_pipeline_parallel_size: int = 1
    infer_expert_parallel_size: int = 1
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 2048
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.5
    offload_train_optimizer: bool = False
    offload_train_grad: bool = False
    offload_train_param: bool = False
    enable_prefix_caching: bool = False
    num_scheduler_steps: int = 1
    enforce_eager: bool = True
    torchair_graph: bool = False
    enable_expert_parallel: bool = False
    ascend_scheduler_config_enabled: bool = True
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    def __post_init__(self):
        Checker.validate_param("limit_mm_image_per_prompt", int, self.limit_mm_image_per_prompt)
        Checker.validate_param("limit_mm_video_per_prompt", int, self.limit_mm_video_per_prompt)
        Checker.validate_param("data_parallel_size", int, self.data_parallel_size)
        if self.tokenizer_name_or_path:
            FileCheck.check_data_path_is_valid(self.tokenizer_name_or_path)
        Checker.validate_param("trust_remote_code", bool, self.trust_remote_code)
        Checker.validate_param("dtype", str, self.dtype)
        Checker.validate_param("infer_tensor_parallel_size", int, self.infer_tensor_parallel_size)
        Checker.validate_param("infer_pipeline_parallel_size", int, self.infer_pipeline_parallel_size)
        Checker.validate_param("infer_expert_parallel_size", int, self.infer_expert_parallel_size)
        Checker.validate_param("max_num_seqs", int, self.max_num_seqs)
        Checker.validate_param("max_num_batched_tokens", int, self.max_num_batched_tokens)
        Checker.validate_param("max_model_len", int, self.max_model_len)
        Checker.validate_param("gpu_memory_utilization", float, self.gpu_memory_utilization)
        Checker.validate_param("offload_train_optimizer", bool, self.offload_train_optimizer)
        Checker.validate_param("offload_train_grad", bool, self.offload_train_grad)
        Checker.validate_param("offload_train_param", bool, self.offload_train_param)
        Checker.validate_param("enable_prefix_caching", bool, self.enable_prefix_caching)
        Checker.validate_param("num_scheduler_steps", int, self.num_scheduler_steps)
        Checker.validate_param("enforce_eager", bool, self.enforce_eager)
        Checker.validate_param("torchair_graph", bool, self.torchair_graph)
        Checker.validate_param("enable_expert_parallel", bool, self.enable_expert_parallel)
        Checker.validate_param("ascend_scheduler_config_enabled", bool, self.ascend_scheduler_config_enabled)
