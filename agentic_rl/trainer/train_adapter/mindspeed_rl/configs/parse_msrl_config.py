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

from typing import Dict, Any
from omegaconf import OmegaConf

from mindspeed_rl import MegatronConfig, GenerateConfig, RLConfig
from mindspeed_rl.config_cls.validate_config import validate_rl_args

from agentic_rl.configs.agentic_rl_config import AgenticRLConfig
from agentic_rl.trainer.train_adapter.parse_config import ConfigParser
from agentic_rl.base.utils.checker import validate_params

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "qwen2.5-7b": {
        "use_mcore_models": True,
        "num_layers": 28,
        "hidden_size": 3584,
        "ffn_hidden_size": 18944,
        "num_attention_heads": 28,
        "rotary_base": 1000000,
        "max_position_embeddings": 32768,
        "make_vocab_size_divisible_by": 1,
        "padded_vocab_size": 152064,
        "untie_embeddings_and_output_weights": True,
        "add_qkv_bias": True,
        "disable_bias_linear": True,
        "group_query_attention": True,
        "num_query_groups": 4,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "swiglu": True,
        "attention_softmax_in_fp32": True,
    },
    "qwen3-0.6b": {
        "use_mcore_models": True,
        "num_layers": 28,
        "hidden_size": 1024,
        "ffn_hidden_size": 3072,
        "num_attention_heads": 16,
        "kv_channels": 128,
        "rotary_base": 1000000,
        "max_position_embeddings": 40960,
        "make_vocab_size_divisible_by": 1,
        "padded_vocab_size": 151936,
        "untie_embeddings_and_output_weights": False,
        "disable_bias_linear": True,
        "group_query_attention": True,
        "num_query_groups": 8,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "swiglu": True,
        "attention_softmax_in_fp32": True,
        "qk_layernorm": True,
    },
    "qwen3-4b": {
        "use_mcore_models": True,
        "num_layers": 36,
        "hidden_size": 2560,
        "ffn_hidden_size": 9728,
        "num_attention_heads": 32,
        "rotary_base": 1000000,
        "max_position_embeddings": 40960,
        "make_vocab_size_divisible_by": 1,
        "padded_vocab_size": 151936,
        "untie_embeddings_and_output_weights": False,
        "add_qkv_bias": False,
        "qk_layernorm": True,
        "disable_bias_linear": True,
        "group_query_attention": True,
        "num_query_groups": 8,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "swiglu": True,
        "attention_softmax_in_fp32": True,
        "kv_channels": 128,
    },
    "qwen3-8b": {
        "use_mcore_models": True,
        "num_layers": 36,
        "hidden_size": 4096,
        "ffn_hidden_size": 12288,
        "num_attention_heads": 32,
        "rotary_base": 1000000,
        "max_position_embeddings": 40960,
        "make_vocab_size_divisible_by": 1,
        "padded_vocab_size": 151936,
        "untie_embeddings_and_output_weights": True,
        "add_qkv_bias": False,
        "qk_layernorm": True,
        "disable_bias_linear": True,
        "group_query_attention": True,
        "num_query_groups": 8,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "swiglu": True,
        "attention_softmax_in_fp32": True,
    },
    "qwen3-30b-a3b": {
        "use_mcore_models": True,
        "num_layers": 48,
        "hidden_size": 2048,
        "ffn_hidden_size": 8192,
        "num_attention_heads": 32,
        "rotary_base": 1000000,
        "max_position_embeddings": 40960,
        "make_vocab_size_divisible_by": 1,
        "padded_vocab_size": 151936,
        "untie_embeddings_and_output_weights": True,
        "disable_bias_linear": True,
        "group_query_attention": True,
        "num_query_groups": 4,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "swiglu": True,
        "attention_softmax_in_fp32": True,
        "attention_bias": False,
        "qk_layernorm": True,

        "num_experts": 128,
        "moe_router_topk": 8,
        "moe_router_load_balancing_type": "aux_loss",
        "moe_intermediate_size": 768,
        "moe_grouped_gemm": True,
        "moe_permutation_async_comm": True,
        "moe_token_dispatcher_type": "alltoall",
        "moe_aux_loss_coeff": 0.001,

        "kv_channels": 128,
        "norm_topk_prob": True,
        "no_gradient_accumulation_fusion": True,
    },
    "qwen3-32b": {
        "use_mcore_models": True,
        "num_layers": 64,
        "hidden_size": 5120,
        "ffn_hidden_size": 25600,
        "num_attention_heads": 64,
        "rotary_base": 1000000,
        "max_position_embeddings": 40960,
        "make_vocab_size_divisible_by": 1,
        "padded_vocab_size": 151936,
        "untie_embeddings_and_output_weights": True,
        "disable_bias_linear": True,
        "group_query_attention": True,
        "num_query_groups": 8,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "swiglu": True,
        "attention_softmax_in_fp32": True,
        "attention_bias": False,
        "qk_layernorm": True,

        "kv_channels": 128,
        "no_gradient_accumulation_fusion": True,
    },
    "qwen3-235b-a22b": {
        "use_mcore_models": True,
        "spec": ["mindspeed_llm.tasks.models.spec.qwen3_spec", "layer_spec"],
        "num_layers": 94,
        "num_experts": 128,
        "moe_router_topk": 8,
        "moe_intermediate_size": 1536,
        "hidden_size": 4096,
        "ffn_hidden_size": 12288,
        "num_attention_heads": 64,
        "group_query_attention": True,
        "num_query_groups": 4,
        "untie_embeddings_and_output_weights": True,
        "disable_bias_linear": True,
        "qk_layernorm": True,
        "kv_channels": 128,
        "norm_topk_prob": True,
        "position_embedding_type": "rope",
        "use_rotary_position_embeddings": True,
        "rotary_base": 1000000,
        "max_position_embeddings": 40960,
        "padded_vocab_size": 151936,
        "make_vocab_size_divisible_by": 1,
        "normalization": "RMSNorm",
        "norm_epsilon": 1e-6,
        "swiglu": True,
        "moe_grouped_gemm": True,
        "moe_permutation_async_comm": True,
        "moe_token_dispatcher_type": "alltoall",
        "use_fused_moe_token_permute_and_unpermute": True,
        "moe_router_load_balancing_type": "aux_loss",
        "moe_aux_loss_coeff": 0.001,
    }
}


def _gen_megatron_config(config: Dict[str, Any]) -> MegatronConfig:
    model_name = config.get("model_name")
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} is not supported. Expected models: {list(MODEL_CONFIGS.keys())}")

    ms_config = config.get("mindspeed_rl", {})

    training_params = {
        "model": model_name,
        "use_fused_rmsnorm": True,
        "use_mcore_models": True,
        "sequence_parallel": True,
        "use_flash_attn": True,
        "no_masked_softmax_fusion": True,
        "attention_softmax_in_fp32": True,
        "no_gradient_accumulation_fusion": True,
        "use_fused_swiglu": True,
        "use_fused_rotary_pos_emb": True,
        "bf16": True,
        "use_distributed_optimizer": True,
        "tokenizer_type": "PretrainedFromHF",
        "tokenizer_name_or_path": config.get("tokenizer_name_or_path"),
        "global_batch_size": ms_config.get("global_batch_size", 2),
        "seq_length": ms_config.get("seq_length", 8192),
        "save_interval": ms_config.get("save_interval", 1000),
        "train_iters": ms_config.get("train_iters", 1),
        "stage": "ray_grpo",
        "attention_dropout": 0,
        "init_method_std": 0.01,
        "hidden_dropout": 0,
        "distributed_backend": "nccl",
        "no_shared_storage": True,
        "variable_seq_lengths": ms_config.get("variable_seq_lengths", False),
        "norm_epsilon": 0.000001,
        "dataset_additional_keys": config.get("dataset_additional_keys"),
        "data_path": ms_config.get("data_path"),
        "reset_position_ids": ms_config.get("reset_position_ids", False),
        "micro_batch_size": ms_config.get("micro_batch_size", 1),
        "tensor_model_parallel_size": ms_config.get("tensor_model_parallel_size", 4),
        "pipeline_model_parallel_size": ms_config.get("pipeline_model_parallel_size", 1),
        "lr": config.get("lr"),
        "lr_decay_style": "constant",
        "min_lr": config.get("min_lr"),
        "weight_decay": config.get("weight_decay"),
        "lr_warmup_fraction": config.get("lr_warmup_fraction"),
        "clip_grad": config.get("clip_grad"),
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "initial_loss_scale": 4096,
        "finetune": True,
        "load": ms_config.get("load_params_path"),
        "save": ms_config.get("save_params_path"),
        "no_load_optim": True,
        "no_load_rng": True,
        "recompute_granularity": ms_config.get("recompute_granularity", "full"),
        "recompute_method": ms_config.get("recompute_method", "block"),
        "recompute_num_layers": ms_config.get("recompute_num_layers", 1),

        "expert_model_parallel_size": ms_config.get("expert_model_parallel_size", 1),
        "shape_order": ms_config.get("shape_order"),
        "moe_alltoall_overlap_comm": ms_config.get("moe_alltoall_overlap_comm", False),
        "swap_optimizer": ms_config.get("swap_optimizer", False),
        "moe_tp_extend_ep": ms_config.get("moe_tp_extend_ep", False),
        "gemm_gradient_accumulation_fusion": ms_config.get("gemm_gradient_accumulation_fusion", False),
    }

    megatron_config = MegatronConfig(
        training_config=OmegaConf.create(training_params),
        model_config=OmegaConf.create({model_name: MODEL_CONFIGS[model_name]}),
    )
    return megatron_config


def _gen_rl_config(config: Dict[str, Any]) -> RLConfig:
    ms_config = config.get("mindspeed_rl", {})

    rl_config = RLConfig(
        OmegaConf.create(
            {
                "actor_resource": {"num_npus": config.get("num_gpus_per_node") * config.get("num_node")},
                "use_remove_padding": ms_config.get("use_remove_padding", False),
                "use_integrated_worker": True,
                "guarantee_order": False,
                "blocking": ms_config.get("blocking", False),
                "gamma": config.get("gamma", 1.0),
                "lam": config.get("lam", 1.0),
                "adv_estimator": ms_config.get("adv_estimator", "group_norm"),
                "kl_horizon": config.get("kl_horizon", 10000),
                "kl_target": config.get("kl_target", 0.1),
                "kl_penalty": config.get("kl_penalty"),
                "kl_ctrl_type": config.get("kl_ctrl_type", "fixed"),
                "init_kl_coef": config.get("kl_coef"),
                "mini_batch_size": ms_config.get("mini_batch_size"),
                "max_prompt_length": config.get("max_prompt_length"),
                "epochs": ms_config.get("epochs"),
                "clip_ratio": config.get("clip_ratio"),
                "entropy_coeff": config.get("entropy_coeff"),
                "shuffle_mini_batch": False,
                "n_samples_per_prompt": config.get("rollout_n"),
                "actor_rollout_dispatch_size": config.get("actor_rollout_dispatch_size"),
                "rule_reward": True,
                "verifier_function": ["env_reward"],
                "verifier_weight": [1.0],
                "verifier_parallel": 1,
                "verifier_timeout": 300,
                "use_tensorboard": config.get("use_tensorboard"),
            }
        )
    )
    return rl_config


def _gen_generate_config(config: Dict[str, Any]) -> GenerateConfig:
    generate_config = GenerateConfig(
        OmegaConf.create(
            {
                "sampling_config": {
                    "logprobs": 1,
                    "max_tokens": config.get("max_tokens", 8192),
                    "top_p": config.get("top_p", 0.9),
                    "top_k": config.get("top_k", 50),
                    "min_p": config.get("min_p", 0.01),
                    "temperature": config.get("temperature", 0.2),
                    "detokenize": False,
                },
                "tokenizer_name_or_path": config.get("tokenizer_name_or_path"),
                "enforce_eager": config.get("enforce_eager", True),
                "enable_expert_parallel": config.get("enable_expert_parallel", False),
                "trust_remote_code": config.get("trust_remote_code", False),
                "infer_tensor_parallel_size": config.get("infer_tensor_parallel_size", 4),
                "infer_pipeline_parallel_size": config.get("infer_pipeline_parallel_size", 1),
                "infer_expert_parallel_size": config.get("infer_expert_parallel_size", 1),
                "max_num_seqs": config.get("max_num_seqs", 32),
                "max_model_len": config.get("max_model_len", 16384),
                "max_num_batched_tokens": config.get("max_num_batched_tokens", 16384),
                "dtype": config.get("dtype", "bfloat16"),
                "gpu_memory_utilization": config.get("gpu_memory_utilization", 0.7),
                "offload_train_optimizer": config.get("offload_train_optimizer", False),
                "offload_train_grad": config.get("offload_train_grad", False),
                "offload_train_param": config.get("offload_train_param", False),
                "enable_prefix_caching": True,
                "num_scheduler_steps": 1,
            }
        )
    )
    return generate_config


class MSRLConfigParser(ConfigParser):
    """Parses and transforms configuration for the MindSpeed-RL training backend."""

    @validate_params(
        config=dict(
            validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) for k in x.keys()),
            message="config must be a dictionary with string keys",
        )
    )
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def process_config(self) -> Dict[str, Any]:
        """Process and validate the configuration."""
        global_config = self._validate_config()
        config_dict = global_config.model_dump()

        agentic_rl_config = AgenticRLConfig()
        agentic_rl_config.agent_name = global_config.agent_name
        agentic_rl_config.agent_engine_wrapper_path = global_config.agent_engine_wrapper_path
        agentic_rl_config.use_stepwise_advantage = global_config.use_stepwise_advantage
        agentic_rl_config.test_only = global_config.test_only
        agentic_rl_config.test_before_train = global_config.test_before_train
        agentic_rl_config.test_data_path = global_config.mindspeed_rl.test_data_path
        agentic_rl_config.max_steps = global_config.max_steps

        actor_config = _gen_megatron_config(config_dict)
        ref_config = actor_config
        reward_config = actor_config
        rl_config = _gen_rl_config(config_dict)
        generate_config = _gen_generate_config(config_dict)

        try:
            validate_rl_args(actor_config, ref_config, reward_config, rl_config, generate_config)
        except ValueError as e:
            raise ValueError(f"Config validation error: {e}") from e

        return {
            "agentic_rl_config": agentic_rl_config,
            "actor_config": actor_config,
            "ref_config": ref_config,
            "reward_config": reward_config,
            "rl_config": rl_config,
            "generate_config": generate_config,
        }
