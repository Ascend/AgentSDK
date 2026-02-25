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

from abc import ABC, abstractmethod
from agentic_rl.base.utils.checker import validate_params
from agentic_rl.base.utils.file_utils import FileCheck


class _BaseInferEngine(ABC):
    """
    This is the base class for the inference engine.
    It initializes the necessary parameters for the inference process,
    including tokenizer information, parallel sizes during training and inference,
    model length limits, data types, GPU memory utilization, and trust settings for remote code.
    """
    def __init__(
            self,
            tokenizer_name_or_path: str,
            train_tensor_parallel_size: int,
            train_pipeline_parallel_size: int,
            prompt_type: str = None,
            prompt_type_path: str = None,
            train_expert_parallel_size: int = 1,
            train_context_parallel_size: int = 1,
            infer_tensor_parallel_size: int = 8,
            infer_pipeline_parallel_size: int = 1,
            infer_expert_parallel_size: int = 1,
            max_num_seqs: int = 1,  # Default value set to 1
            max_model_len: int = 2048,  # Default value set to 2048
            dtype: str = "bfloat16",  # Default value set to "bfloat16"
            gpu_memory_utilization: float = 0.5,  # Default value set to 0.5
            trust_remote_code: bool = False,
            enable_expert_parallel: bool = False,
    ):
        """
        Initialize the base inference engine.

        Args:
            tokenizer_name_or_path (str): Path or name of the tokenizer.
            train_tensor_parallel_size (int): Tensor parallel size during training.
            train_pipeline_parallel_size (int): Pipeline parallel size during training.
            train_expert_parallel_size (int): Expert parallel size during training.
            train_context_parallel_size (int): Context parallel size during training.
            infer_tensor_parallel_size (int): Tensor parallel size during inference.
            infer_pipeline_parallel_size (int): Pipeline parallel size during inference.
            infer_expert_parallel_size (int): Expert parallel size during inference.
            max_num_seqs (int): Maximum number of sequences to process simultaneously. Default is 1.
            max_model_len (int): Maximum model length (in tokens). Default is 2048.
            dtype (str): Data type for model weights. Default is "bfloat16".
            gpu_memory_utilization (float): GPU memory utilization factor. Default is 0.5.
            trust_remote_code (bool): Whether to trust remote code (e.g., for custom tokenizers).
            enable_expert_parallel (bool): Whether to enable expert parallel.
        """
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.prompt_type = prompt_type
        self.prompt_type_path = prompt_type_path
        self.train_tensor_parallel_size = train_tensor_parallel_size
        self.train_pipeline_parallel_size = train_pipeline_parallel_size
        self.train_expert_parallel_size = train_expert_parallel_size
        self.train_context_parallel_size = train_context_parallel_size
        self.infer_tensor_parallel_size = infer_tensor_parallel_size
        self.infer_pipeline_parallel_size = infer_pipeline_parallel_size
        self.infer_expert_parallel_size = infer_expert_parallel_size
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.enable_expert_parallel = enable_expert_parallel


    @abstractmethod
    def init_cache_engine(self):
        pass

    @abstractmethod
    def free_cache_engine(self):
        pass

    @abstractmethod
    def offload_model_weights(self):
        pass

    @abstractmethod
    def sync_model_weights(self, params, load_format='megatron'):
        pass

    @abstractmethod
    def generate_sequences(self,
                           prompts=None,
                           sampling_params=None,
                           prompt_token_ids=None,
                           use_tqdm=None,
                           **kwargs):
        pass


class BaseInferEngine(_BaseInferEngine):
    """
    This is the base class for the inference engine.
    It initializes the necessary parameters for the inference process,
    including tokenizer information, parallel sizes during training and inference,
    model length limits, data types, GPU memory utilization, and trust settings for remote code.
    """

    @validate_params(
        train_tensor_parallel_size=dict(
            validator=lambda x: isinstance(x, int) and x > 0,
            message="train_tensor_parallel_size must be a positive integer",
        ),
        train_pipeline_parallel_size=dict(
            validator=lambda x: isinstance(x, int) and x > 0,
            message="train_pipeline_parallel_size must be a positive integer",
        ),
        train_expert_parallel_size=dict(
            validator=lambda x: isinstance(x, int) and x > 0,
            message="train_expert_parallel_size must be a positive integer",
        ),
        train_context_parallel_size=dict(
            validator=lambda x: isinstance(x, int) and x > 0,
            message="train_context_parallel_size must be a positive integer",
        ),
        infer_tensor_parallel_size=dict(
            validator=lambda x: isinstance(x, int) and x > 0,
            message="infer_tensor_parallel_size must be a positive integer",
        ),
        infer_pipeline_parallel_size=dict(
            validator=lambda x: isinstance(x, int) and x > 0,
            message="infer_pipeline_parallel_size must be a positive integer",
        ),
        infer_expert_parallel_size=dict(
            validator=lambda x: isinstance(x, int) and x > 0,
            message="infer_expert_parallel_size must be a positive integer",
        ),
        max_num_seqs=dict(
            validator=lambda x: isinstance(x, int) and x > 0, message="max_num_seqs must be a positive integer"
        ),
        max_model_len=dict(
            validator=lambda x: isinstance(x, int) and 0 < x <= 128 * 1024,
            message="max_model_len must be a positive integer and less than or equal to 128 * 1024",
        ),
        dtype=dict(
            validator=lambda x: isinstance(x, str) and x in ["bfloat16", "float16", "float32"],
            message="dtype must be a string and one of ['bfloat16', 'float16', 'float32']",
        ),
        gpu_memory_utilization=dict(
            validator=lambda x: isinstance(x, float) and 0 < x <= 1,
            message="gpu_memory_utilization must be a float and greater than 0 and less than or equal to 1",
        ),
        trust_remote_code=dict(
            validator=lambda x: isinstance(x, bool),
            message="trust_remote_code must be a boolean"
        ),
        enable_expert_parallel=dict(
            validator=lambda x: isinstance(x, bool), message="enable_expert_parallel must be a boolean"
        ),
        infer_backend=dict(
            validator=lambda x: isinstance(x, str) and x in ["vllm"],
            message="infer_backend must be a string and one of ['vllm']"
        ),
        prompt_type=dict(
            validator=lambda x: x is None or isinstance(x, str), message="prompt_type must be a string or None"
        ),
    )
    def __init__(
        self,
        tokenizer_name_or_path: str,
        train_tensor_parallel_size: int,
        train_pipeline_parallel_size: int,
        prompt_type: str = None,
        prompt_type_path: str = None,
        train_expert_parallel_size: int = 1,
        train_context_parallel_size: int = 1,
        infer_tensor_parallel_size: int = 8,
        infer_pipeline_parallel_size: int = 1,
        infer_expert_parallel_size: int = 1,
        max_num_seqs: int = 1,
        max_model_len: int = 2048,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.5,
        trust_remote_code: bool = False,
        enable_expert_parallel: bool = False,
        infer_backend: str = "vllm"
    ):
        """
        Initialize the base inference engine.

        Args:
            tokenizer_name_or_path (str): Path or name of the tokenizer.
            train_tensor_parallel_size (int): Tensor parallel size during training.
            train_pipeline_parallel_size (int): Pipeline parallel size during training.
            train_expert_parallel_size (int): Expert parallel size during training.
            train_context_parallel_size (int): Context parallel size during training.
            infer_tensor_parallel_size (int): Tensor parallel size during inference.
            infer_pipeline_parallel_size (int): Pipeline parallel size during inference.
            infer_expert_parallel_size (int): Expert parallel size during inference.
            max_num_seqs (int): Maximum number of sequences to process simultaneously. Default is 1.
            max_model_len (int): Maximum model length (in tokens). Default is 2048.
            dtype (str): Data type for model weights. Default is "bfloat16".
            gpu_memory_utilization (float): GPU memory utilization factor. Default is 0.5.
            trust_remote_code (bool): Whether to trust remote code.
            enable_expert_parallel (bool): Whether to enable expert parallel.
            infer_backend (str): Inference backend to use. Default is "vllm".
        """
        FileCheck.check_data_path_is_valid(tokenizer_name_or_path)

        if prompt_type_path is not None:
            FileCheck.check_data_path_is_valid(prompt_type_path)

        super().__init__(
            tokenizer_name_or_path=tokenizer_name_or_path,
            train_tensor_parallel_size=train_tensor_parallel_size,
            train_pipeline_parallel_size=train_pipeline_parallel_size,
            prompt_type=prompt_type,
            prompt_type_path=prompt_type_path,
            train_expert_parallel_size=train_expert_parallel_size,
            train_context_parallel_size=train_context_parallel_size,
            infer_tensor_parallel_size=infer_tensor_parallel_size,
            infer_pipeline_parallel_size=infer_pipeline_parallel_size,
            infer_expert_parallel_size=infer_expert_parallel_size,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            enable_expert_parallel=enable_expert_parallel
        )
        self.infer_backend = infer_backend
        self.eplb_map = None
        self.global_redundant_expert_num = 0
        self.infer_local_num_experts = -1