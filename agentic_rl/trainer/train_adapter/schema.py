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

from typing import Optional, List, Literal

from pydantic import BaseModel, field_validator, ConfigDict, model_validator

from agentic_rl.base.utils.file_utils import FileCheck

_verl_ckpt_type = Literal["model", "optimizer", "extra", "hf_model"]


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)


class MindspeedRLConfig(BaseConfig):
    data_path: str
    test_data_path: Optional[str] = None
    load_params_path: str
    save_params_path: str
    epochs: int = 1
    seq_length: int = 8192
    global_batch_size: int = 16
    save_interval: int = 1000
    train_iters: int = 1
    mini_batch_size: int = 16
    micro_batch_size: int = 1
    tensor_model_parallel_size: int = 4
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    adv_estimator: Literal['group_norm', 'gae'] = 'group_norm'
    recompute_num_layers: int = 1
    recompute_granularity: Optional[str] = None
    recompute_method: Optional[str] = None
    shape_order: Optional[str] = None
    moe_alltoall_overlap_comm: bool = False
    swap_optimizer: bool = False
    moe_tp_extend_ep: bool = False
    gemm_gradient_accumulation_fusion: bool = False
    variable_seq_lengths: bool = False
    reset_position_ids: bool = False
    use_remove_padding: bool = False
    blocking: bool = False

    @field_validator(
        "epochs",
        "seq_length",
        "save_interval",
        "train_iters",
        "mini_batch_size",
        "micro_batch_size",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
        "global_batch_size",
    )
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Value {v} must be positive.")
        return v

    @field_validator("data_path", "load_params_path", "save_params_path")
    @classmethod
    def validate_path_exists(cls, v):
        FileCheck.check_data_path_is_valid(v)
        return v

    @field_validator("test_data_path")
    @classmethod
    def validate_test_path_exists(cls, v):
        if v:
            FileCheck.check_data_path_is_valid(v)
        return v


class VerlConfig(BaseConfig):
    train_files: str
    val_files: str
    total_epochs: int = 2
    total_training_steps: Optional[int] = None
    save_freq: int = 1000
    ppo_mini_batch_size: int = 16
    ppo_max_token_len_per_gpu: int = 24000
    ppo_epochs: int = 1
    project_name: str = "default-agent"
    experiment_name: str = "default-experiment"
    max_response_length: int = 2048
    train_batch_size: int = 8
    val_batch_size: int = 512
    dataloader_num_workers: int = 8
    nnodes: int = 1

    # adv
    adv_estimator: Literal['grpo', 'gae'] = 'grpo'

    # lr
    warmup_style: Literal['constant', 'cosine'] = "constant"
    # cosine lr only
    min_lr_ratio: float = 0.0
    num_cycles: float = 0.5

    # checkpoint
    ckpt_content: List[_verl_ckpt_type] = ['model', 'optimizer', 'extra']

    # policy loss
    policy_loss_mode: Literal['vanilla', 'clip-cov', 'kl-cov', 'gpg'] = 'vanilla'
    policy_loss_clip_cov_ratio: float = 0.0002
    policy_loss_clip_cov_lb: float = 1.0
    policy_loss_clip_cov_ub: float = 5.0
    policy_loss_kl_cov_ratio: float = 0.0002
    policy_loss_ppo_kl_coef: float = 0.1

    # FSDP
    fsdp_param_offload: bool = False
    fsdp_optimizer_offload: bool = False

    # others
    loss_agg_mode: Literal['token-mean', 'seq-mean-token-sum', 'seq-mean-token-mean'] = 'token-mean'
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    kl_loss_type: Literal["kl", "abs", "mse", "low_var_kl", "full"] = "low_var_kl"
    grad_clip: float = 1.0
    entropy_from_logits_with_chunking: bool = False

    # trainer
    balance_batch: bool = True
    val_before_train: bool = True
    val_only: bool = False
    test_freq: int = -1

    # data
    truncation: Literal['error', 'left', 'right', 'middle'] = 'error'

    @field_validator("total_training_steps")
    @classmethod
    def validate_None_or_positive(cls, v):
        if v is None or v > 0:
            return v
        raise ValueError(f"Value {v} must be None or positive.")

    @field_validator("test_freq")
    @classmethod
    def validate_disable_or_positive(cls, v):
        if v == -1 or (v > 0):
            return v
        raise ValueError(f"Value {v} must be -1 or positive.")

    @field_validator(
        "total_epochs",
        "save_freq",
        "ppo_mini_batch_size",
        "ppo_max_token_len_per_gpu",
        "ppo_epochs",
        "max_response_length",
        "train_batch_size",
        "val_batch_size",
        "dataloader_num_workers",
        "nnodes",
        "num_cycles",
        "policy_loss_clip_cov_ratio",
        "policy_loss_clip_cov_lb",
        "policy_loss_clip_cov_ub",
        "policy_loss_kl_cov_ratio",
        "grad_clip",
    )
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Value {v} must be positive.")
        return v

    @field_validator("train_files", "val_files")
    @classmethod
    def validate_paths(cls, v):
        FileCheck.check_data_path_is_valid(v)
        return v

    @field_validator(
        "policy_loss_ppo_kl_coef",
        "kl_loss_coef",
        "min_lr_ratio"
    )
    @classmethod
    def validate_fraction(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Value {v} must be between 0 and 1.")
        return v

    @field_validator("ckpt_content")
    @classmethod
    def unique_list_validator(cls, v):
        if len(v) != len(set(v)):
            raise ValueError(f"Value in list {v} must be unique.")
        return v

    @model_validator(mode="after")
    def validate_nested_config(self):
        if self.policy_loss_clip_cov_lb >= self.policy_loss_clip_cov_ub:
            raise ValueError("clip_cov upper bound should be larger than its lower bound!")
        return self


class GlobalConfig(BaseConfig):
    # Shared Parameters
    tokenizer_name_or_path: str
    model_name: str
    agent_name: str
    agent_engine_wrapper_path: str
    train_backend: Literal["mindspeed_rl", "verl"]
    use_stepwise_advantage: bool = False
    max_steps: int = 5
    use_tensorboard: bool = False
    test_before_train: bool = False
    test_only: bool = False

    # Inference / Model Params
    infer_tensor_parallel_size: int = 4
    infer_pipeline_parallel_size: int = 1
    infer_expert_parallel_size: int = 1
    max_num_seqs: int = 1024
    max_num_batched_tokens: int = 8192
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.85
    max_tokens: int = 8192
    enable_prefix_caching: bool = False
    num_scheduler_steps: int = 1
    offload_train_optimizer: bool = False
    offload_train_grad: bool = False
    offload_train_param: bool = False
    enable_expert_parallel: bool = False
    trust_remote_code: bool = False

    # Sampling / Generation
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    top_k: int = 20
    top_p: float = 1.0
    min_p: float = 0.01
    temperature: float = 0.6
    enforce_eager: bool = True

    # RL Hyperparams
    use_kl_in_reward: bool = False
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.001
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"] = "kl"
    kl_coef: float = 0.05
    gamma: float = 1.0
    lam: float = 1.0
    kl_horizon: int = 10000
    kl_target: float = 0.1
    kl_ctrl_type: Literal["fixed", "adaptive"] = "fixed"

    # Optimizer
    lr: float = 1.0e-06
    min_lr: float = 0.0
    lr_warmup_fraction: float = 0.0
    clip_grad: float = 1.0
    weight_decay: float = 0.01

    # Resources / Training
    num_gpus_per_node: int = 8
    num_node: int = 1
    max_prompt_length: int = 2048
    rollout_n: int = 2
    actor_rollout_dispatch_size: int = 2

    # Sub-configs
    mindspeed_rl: Optional[MindspeedRLConfig] = None
    verl: Optional[VerlConfig] = None

    dataset_additional_keys: Optional[List[str]] = None

    @field_validator("tokenizer_name_or_path", "agent_engine_wrapper_path")
    @classmethod
    def validate_paths(cls, v):
        FileCheck.check_data_path_is_valid(v)
        return v

    @field_validator(
        "gpu_memory_utilization",
        "top_p", "min_p",
        "lr_warmup_fraction",
        "gamma",
        "lam",
        "weight_decay",
        "clip_ratio",
    )
    @classmethod
    def validate_fraction(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Value {v} must be between 0 and 1.")
        return v

    @field_validator(
        "num_gpus_per_node",
        "num_node",
        "max_num_seqs",
        "rollout_n",
        "max_model_len",
        "kl_horizon",
        "lr",
        "clip_grad",
        "kl_coef",
        "temperature",
        "infer_tensor_parallel_size",
        "infer_pipeline_parallel_size",
        "infer_expert_parallel_size",
        "max_num_batched_tokens",
        "max_tokens",
        "max_prompt_length",
        "max_steps",
        "actor_rollout_dispatch_size",
    )
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Value {v} must be positive.")
        return v

    @field_validator(
        "entropy_coeff",
        "kl_target",
    )
    @classmethod
    def validate_non_negative(cls, v):
        if v < 0:
            raise ValueError(f"Value {v} must be non-negative.")
        return v

    @model_validator(mode="after")
    def validate_backend_config(self):
        if self.train_backend == "mindspeed_rl" and self.mindspeed_rl is None:
            raise ValueError("mindspeed_rl config section is required when train_backend is 'mindspeed_rl'")
        if self.train_backend == "verl" and self.verl is None:
            raise ValueError("verl config section is required when train_backend is 'verl'")
        return self
