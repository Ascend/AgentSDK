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
import os
import re
from pathlib import Path
from typing import Dict, Any

from mindspeed_rl import MegatronConfig, GenerateConfig, RLConfig
from mindspeed_rl.config_cls.validate_config import validate_rl_args
from omegaconf import OmegaConf

from agentic_rl.base.utils.file_utils import FileCheck
from agentic_rl.configs.agentic_rl_config import AgenticRLConfig


def _gen_megatron_config(tokenizer_name_or_path: str, data_path: str, load_path: str,
                         save_path: str, train_iters: int) -> MegatronConfig:
    # cut use_mc2
    megatron_config = MegatronConfig(
        training_config=OmegaConf.create({'model': 'qwen25-7b', 'use_fused_rmsnorm': True, 'use_mcore_models': True,
                                          'sequence_parallel': True, 'use_flash_attn': True,
                                          'no_masked_softmax_fusion': True, 'attention_softmax_in_fp32': True,
                                          'no_gradient_accumulation_fusion': True, 'use_fused_swiglu': True,
                                          'use_fused_rotary_pos_emb': True, 'bf16': True,
                                          'use_distributed_optimizer': True,
                                          'tokenizer_type': 'PretrainedFromHF',
                                          'tokenizer_name_or_path': tokenizer_name_or_path,
                                          'global_batch_size': 2,
                                          'seq_length': 8192, 'save_interval': 1000, 'train_iters': train_iters,
                                          'stage': 'ray_grpo', 'attention_dropout': 0, 'init_method_std': 0.01,
                                          'hidden_dropout': 0,
                                          'distributed_backend': 'nccl', 'no_shared_storage': True,
                                          'variable_seq_lengths': True, 'norm_epsilon': 0.00001,
                                          'dataset_additional_keys': ['labels'], 'data_path': data_path,
                                          'split': '100,0,0',
                                          'reset_position_ids': True,
                                          # actor_config
                                          'micro_batch_size': 1, 'tensor_model_parallel_size': 4,
                                          'pipeline_model_parallel_size': 1, 'lr': '1e-7', 'lr_decay_style': 'constant',
                                          'min_lr': 0.0, 'weight_decay': 0.0, 'lr_warmup_fraction': 0.0,
                                          'clip_grad': 1.0,
                                          'adam_beta1': 0.9, 'adam_beta2': 0.999, 'initial_loss_scale': 4096,
                                          'finetune': True,
                                          'load': load_path, 'save': save_path, 'no_load_optim': True,
                                          'no_load_rng': True
                                          }),
        model_config=OmegaConf.create({
            'qwen25-7b': {'use_mcore_models': True, 'num_layers': 28, 'hidden_size': 3584, 'ffn_hidden_size': 18944,
                          'num_attention_heads': 28, 'rotary_base': 1000000, 'max_position_embeddings': 32768,
                          'make_vocab_size_divisible_by': 1, 'padded_vocab_size': 152064,
                          'untie_embeddings_and_output_weights': True, 'add_qkv_bias': True,
                          'disable_bias_linear': True, 'group_query_attention': True, 'num_query_groups': 4,
                          'position_embedding_type': 'rope', 'normalization': 'RMSNorm',
                          'swiglu': True, 'attention_softmax_in_fp32': True
                          }
        })
    )
    return megatron_config


def _gen_rl_config() -> RLConfig:
    rl_config = RLConfig(
        OmegaConf.create({
            'actor_resource': {
                'num_npus': 8
            },
            'use_remove_padding': True,
            'use_integrated_worker': True,
            'guarantee_order': False,
            'blocking': False,
            'gamma': 1.0,
            'lam': 0.95,
            'adv_estimator': 'group_norm',
            'kl_penalty': 'kl',
            'kl_ctrl_type': 'fixed',
            'init_kl_coef': 0.05,
            'mini_batch_size': 16,
            'max_prompt_length': 8192,
            'epochs': 1,
            'clip_ratio': 0.2,
            'entropy_coeff': 0.001,
            'shuffle_mini_batch': False,
            'n_samples_per_prompt': 2,
            'actor_rollout_dispatch_size': 2,
            'rule_reward': True,
            'verifier_function': ["env_reward"],
            'verifier_weight': [1.0],
            'verifier_parallel': 1,
            'verifier_timeout': 300,
            'use_tensorboard': False,
        })
    )
    return rl_config


def _gen_generate_config() -> GenerateConfig:
    generate_config = GenerateConfig(
        OmegaConf.create({
            'sampling_config': {
                "logprobs": 1,
                "max_tokens": 8192,
                "top_p": 0.9,
                "top_k": 50,
                "min_p": 0.01,
                "temperature": 0.2,
                "detokenize": False,
            },
            'enforce_eager': True,
            'trust_remote_code': False,
            'infer_tensor_parallel_size': 4,
            'infer_pipeline_parallel_size': 1,
            'infer_expert_parallel_size': 1,
            'max_num_seqs': 32,
            'max_model_len': 16384,
            'max_num_batched_tokens': 16384,
            'dtype': 'bfloat16',
            'gpu_memory_utilization': 0.4,
            'offload_train_optimizer': True,
            'offload_train_grad': True,
            'offload_train_param': True
        })
    )
    if generate_config.tokenizer_name_or_path == "/path/to/tokenizer":
        generate_config.tokenizer_name_or_path = ""
    return generate_config


class ConfigParser:
    """Configuration parser with field validation support.

    This class is responsible for parsing and validating the configuration dictionary.
    It checks for required keys, validates types, and ensures that paths and values meet the expected format.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the ConfigParser with the provided configuration dictionary.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters.
        """
        self.config = config

    def _validate_config(self):
        """Validate the configuration dictionary.

        This method checks for the presence of required keys, validates their types,
        and ensures that paths and values meet the expected format.
        """
        required_key_and_types: Dict[str, Any] = {
            "tokenizer_name_or_path": (str, "path"),
            "data_path": (str, "data_path"),
            "load_params_path": (str, "path"),
            "save_params_path": (str, "path"),
            "train_iters": (int, "value"),
            "agent_name": (str, "value"),
            "agent_engine_wrapper_path": (str, "path")
        }
        for key, values in required_key_and_types.items():
            if key not in self.config:
                raise ValueError(f"The yaml's {key} is required.")

            if not isinstance(self.config.get(key), values[0]):
                raise TypeError(f"The yaml's {key} should be {values[0]} type.")

            config_value = self.config.get(key)

            if values[1] == "path":
                FileCheck.check_data_path_is_valid(config_value)

            if values[1] == "value" and isinstance(config_value, int) and self.config.get(key) <= 0:
                raise ValueError(f"The yaml's {key} should be greater than 0.")

            if (values[1] == "value" and isinstance(self.config.get(key), str) and
                    not re.search(r'^[A-Za-z][A-Za-z0-9_]*$', self.config.get(key))):
                raise ValueError(f"The yaml's {key} should start with a letter and "
                                 f"contains only letters, digits or underscores.")

            if values[1] == "data_path":
                self._validate_data_path(key)

    def _validate_data_path(self, required_key: str):
        """
        Validate the data path specified by the given key in the configuration.

        This method:
        1. Parses the path to extract the directory and the prefix (the last part of the path).
        2. Validates that the directory exists and meets the required file system permissions.
        3. Checks that the directory is not empty.
        4. Ensures that all files in the directory start with the prefix, to maintain naming consistency.

        Args:
            required_key (str): The configuration key that holds the data path to validate.

        Raises:
            ValueError: If the directory is empty or any file does not start with the expected prefix.
        """
        path_obj = Path(self.config.get(required_key))
        directory = str(path_obj.parent)
        prefix = path_obj.name

        FileCheck.check_data_path_is_valid(directory)

        valid_data_files_suffix = {
            "_packed_attention_mask_document.bin",
            "_packed_attention_mask_document.idx",
            "_packed_input_ids_document.bin",
            "_packed_input_ids_document.idx",
            "_packed_labels_document.bin",
            "_packed_labels_document.idx",
        }

        for suffix in valid_data_files_suffix:
            if not os.path.exists(os.path.join(directory, prefix + suffix)):
                raise ValueError(f"The yaml's {required_key} indicates a path where has no file with suffix {suffix}.")

    def process_config(self) -> dict[str, Any]:
        """Process and validate the configuration.

        This method validates the configuration and generates the final configuration dictionary
        for the training and inference pipeline.

        Returns:
            dict[str, Any]: A dictionary containing the parsed and validated configuration.
        """
        self._validate_config()

        agentic_rl_config = AgenticRLConfig()
        agentic_rl_config.agent_name = self.config.get("agent_name")
        agentic_rl_config.agent_engine_wrapper_path = self.config.get("agent_engine_wrapper_path")

        actor_config = _gen_megatron_config(self.config.get("tokenizer_name_or_path"),
                                            self.config.get("data_path"),
                                            self.config.get("load_params_path"),
                                            self.config.get("save_params_path"),
                                            self.config.get("train_iters"))
        # when rl_config.use_integrated_worker is True set, currently default value is True
        ref_config = actor_config

        reward_config = actor_config

        rl_config = _gen_rl_config()
        generate_config = _gen_generate_config()

        try:  # validate all config parameters
            validate_rl_args(actor_config, ref_config, reward_config, rl_config, generate_config)
        except ValueError as e:
            raise ValueError(f"Config validation error: {e}") from e

        return {
            "agentic_rl_config": agentic_rl_config,
            "actor_config": actor_config,
            "ref_config": ref_config,
            "reward_config": reward_config,
            "rl_config": rl_config,
            "generate_config": generate_config
        }
