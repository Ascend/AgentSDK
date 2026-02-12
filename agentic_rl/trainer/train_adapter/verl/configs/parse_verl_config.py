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

from typing import Dict, Any, List
from omegaconf import OmegaConf

from agentic_rl.trainer.train_adapter.verl.configs.default_config import DEFAULT_CONFIG
from agentic_rl.configs.agentic_rl_config import AgenticRLConfig, GenConfig, SamplingConfig
from agentic_rl.trainer.train_adapter.parse_config import ConfigParser
from agentic_rl.base.log.loggers import Loggers
 
logger = Loggers(__name__)


def _gen_gen_config(config: Dict[str, Any]) -> GenConfig:
    """
    Generate the final generation (inference) configuration from the
    flattened input configuration.

    Args:
        config (Dict[str, Any]):
            Flattened input configuration dictionary, typically produced
            by `_process_input_config`. Values may originate from user input
            or defaults.

    Returns:
        GenConfig:
            Fully constructed generation configuration object containing:
            - Sampling behavior (SamplingConfig)
            - Tokenization and model loading options
            - Parallelism and batching parameters
            - Memory and scheduling related flags
    """

    sampling_config = SamplingConfig(
        logprobs=config.get("logprobs", 1),
        max_tokens=config.get("max_tokens", 8192),
        top_p=config.get("top_p", 0.9),
        top_k=config.get("top_k", 50),
        min_p=config.get("min_p", 0.01),
        temperature=config.get("temperature", 0.2),
        detokenize=config.get("detokenize", False),
        seed=config.get("seed", None),
    )

    gen_conf = GenConfig(
        limit_mm_image_per_prompt=config.get("limit_mm_image_per_prompt", 1),
        limit_mm_video_per_prompt=config.get("limit_mm_video_per_prompt", 0),
        tokenizer_name_or_path=config.get("tokenizer_name_or_path", ""),
        trust_remote_code=config.get("trust_remote_code", False),
        dtype=config.get("dtype", "bfloat16"),
        infer_tensor_parallel_size=config.get("infer_tensor_parallel_size", 4),
        infer_pipeline_parallel_size=config.get("infer_pipeline_parallel_size", 1),
        infer_expert_parallel_size=config.get("infer_expert_parallel_size", 1),
        max_num_seqs=config.get("max_num_seqs", 32),
        max_num_batched_tokens=config.get("max_num_batched_tokens", 16384),
        max_model_len=config.get("max_model_len", 16384),
        gpu_memory_utilization=config.get("gpu_memory_utilization", 0.4),
        offload_train_optimizer=config.get("offload_train_optimizer", True),
        offload_train_grad=config.get("offload_train_grad", True),
        offload_train_param=config.get("offload_train_param", True),
        enable_prefix_caching=config.get("enable_prefix_caching", False),
        num_scheduler_steps=config.get("num_scheduler_steps", 1),
        enforce_eager=config.get("enforce_eager", True),
        torchair_graph=config.get("torchair_graph", False),
        enable_expert_parallel=config.get("enable_expert_parallel", False),
        ascend_scheduler_config_enabled=config.get("ascend_scheduler_config_enabled", True),
        sampling_config=sampling_config,
    )
    return gen_conf

    
def _validate_verl_constraints(verl_config: Dict[str, Any]) -> None:
    """
    Validate and enforce mandatory VERL configuration constraints.

    This function mutates the provided VERL configuration in place.
    It does NOT return a new configuration object.

    Mandatory constraints enforced:
    - actor_rollout_ref.hybrid_engine must be True
    - actor_rollout_ref.rollout.mode must be 'async'

    Args:
        verl_config (DictConfig):
            VERL configuration object to be validated and corrected.
    """

    if not verl_config.actor_rollout_ref.hybrid_engine:
        logger.warning(
            "You are seeing this warning because you have changed some default configurations"
            "and will cause errors during training."
            "VERL training backend require 'hybrid_engine=True'. Setting 'hybrid_engine=True' automatically!"
            )
        verl_config.actor_rollout_ref.hybrid_engine = True
    
    if verl_config.actor_rollout_ref.rollout.mode != 'async':
        logger.warning(
            "You are seeing this warning because you have changed some default configurations"
            "and will cause errors during training."
            "VERL training backend require 'rollout.mode=async'. Setting 'rollout.mode=async' automatically!"
            )
        verl_config.actor_rollout_ref.rollout.mode = 'async'
 
 
class VerlConfigParser(ConfigParser):
    """Parses and transforms configuration for the VERL training backend.
 
    Attributes:
        _CONFIG_MAPPING (Dict[str, str]): A mapping from flat configuration keys
            to their corresponding nested paths in the VERL configuration structure.
    """
 
    _CONFIG_MAPPING: Dict[str, str] = {
        # Data
        "train_files": "data.train_files",
        "val_files": "data.val_files",
        "train_batch_size": "data.train_batch_size",
        "val_batch_size": "data.val_batch_size",
        "max_prompt_length": "data.max_prompt_length",
        "max_response_length": "data.max_response_length",
        "dataloader_num_workers": "data.dataloader_num_workers",
        "truncation": "data.truncation",
        # Trainer
        "total_epochs": "trainer.total_epochs",
        "total_training_steps": "trainer.total_training_steps",
        "save_freq": "trainer.save_freq",
        "project_name": "trainer.project_name",
        "experiment_name": "trainer.experiment_name",
        "nnodes": "trainer.nnodes",
        "num_gpus_per_node": "trainer.n_gpus_per_node",
        "balance_batch": "trainer.balance_batch",
        "val_before_train": "trainer.val_before_train",
        "val_only": "trainer.val_only",
        "test_freq": "trainer.test_freq",
        # Actor
        "lr": "actor_rollout_ref.actor.optim.lr",
        "lr_warmup_fraction": "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio",
        "warmup_style": "actor_rollout_ref.actor.optim.warmup_style",
        "min_lr_ratio": "actor_rollout_ref.actor.optim.min_lr_ratio",
        "num_cycles": "actor_rollout_ref.actor.optim.num_cycles",
        "weight_decay": "actor_rollout_ref.actor.optim.weight_decay",
        "ppo_epochs": "actor_rollout_ref.actor.ppo_epochs",
        "ppo_mini_batch_size": "actor_rollout_ref.actor.ppo_mini_batch_size",
        "ppo_max_token_len_per_gpu": "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
        "entropy_coeff": "actor_rollout_ref.actor.entropy_coeff",
        "clip_ratio": "actor_rollout_ref.actor.clip_ratio",
        "ckpt_content": "actor_rollout_ref.actor.checkpoint.save_contents",
        "policy_loss_mode": "actor_rollout_ref.actor.policy_loss.loss_mode",
        "policy_loss_clip_cov_ratio": "actor_rollout_ref.actor.policy_loss.clip_cov_ratio",
        "policy_loss_clip_cov_lb": "actor_rollout_ref.actor.policy_loss.clip_cov_lb",
        "policy_loss_clip_cov_ub": "actor_rollout_ref.actor.policy_loss.clip_cov_ub",
        "policy_loss_kl_cov_ratio": "actor_rollout_ref.actor.policy_loss.kl_cov_ratio",
        "policy_loss_ppo_kl_coef": "actor_rollout_ref.actor.policy_loss.ppo_kl_coef",
        "fsdp_param_offload": "actor_rollout_ref.actor.fsdp_config.param_offload",
        "fsdp_optimizer_offload": "actor_rollout_ref.actor.fsdp_config.fsdp_optimizer_offload",
        "loss_agg_mode": "actor_rollout_ref.actor.loss_agg_mode",
        "use_kl_loss": "actor_rollout_ref.actor.use_kl_loss",
        "kl_loss_coef": "actor_rollout_ref.actor.kl_loss_coef",
        "kl_loss_type": "actor_rollout_ref.actor.kl_loss_type",
        "ulysses_sequence_parallel_size": "actor_rollout_ref.actor.ulysses_sequence_parallel_size",
        "entropy_from_logits_with_chunking": "actor_rollout_ref.actor.entropy_from_logits_with_chunking",
        "grad_clip": "actor_rollout_ref.actor.grad_clip",

        # Rollout
        "dtype": "actor_rollout_ref.rollout.dtype",
        "enforce_eager": "actor_rollout_ref.rollout.enforce_eager",
        "rollout_n": "actor_rollout_ref.rollout.n",
        "max_model_len": "actor_rollout_ref.rollout.max_model_len",
        "temperature": "actor_rollout_ref.rollout.temperature",
        "top_k": "actor_rollout_ref.rollout.top_k",
        "top_p": "actor_rollout_ref.rollout.top_p",
        "gpu_memory_utilization": "actor_rollout_ref.rollout.gpu_memory_utilization",
        "max_num_seqs": "actor_rollout_ref.rollout.max_num_seqs",
        "max_num_batched_tokens": "actor_rollout_ref.rollout.max_num_batched_tokens",
        "infer_tensor_parallel_size": "actor_rollout_ref.rollout.tensor_model_parallel_size",
        
        # Algorithm
        "use_kl_in_reward": "algorithm.use_kl_in_reward",
        "gamma": "algorithm.gamma",
        "lam": "algorithm.lam",
        "adv_estimator": "algorithm.adv_estimator",
        "kl_ctrl_type": "algorithm.kl_ctrl.type",
        "kl_penalty": "algorithm.kl_penalty",
        "kl_coef": "algorithm.kl_ctrl.kl_coef",
        "kl_horizon": "algorithm.kl_ctrl.horizon",

        # Models
        "tokenizer_name_or_path": "actor_rollout_ref.model.path",
    }
 
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the VerlConfigParser."""
        super().__init__(config)
 
    @staticmethod
    def _set_nested_value(data: Dict[str, Any], path: str, value: Any) -> None:
        """Sets a value in a nested dictionary structure using a dot-separated path.
 
        Args:
            data: The dictionary to modify.
            path: A dot-separated string indicating the key path (e.g., "a.b.c").
            value: The value to set at the specified path.
        """
        keys = path.split(".")
        current_level = data
 
        for key in keys[:-1]:
            current_level = current_level.setdefault(key, {})
        current_level[keys[-1]] = value

    def _process_input_config(self) -> Dict[str, Any]:
        """
        Process and validate the raw input configuration, and split it into
        logical sub-configurations used by different parts of the system.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "agentic_rl_config" (AgenticRLConfig):
                    Runtime configuration for Agentic RL execution.
                - "input_config" (Dict[str, Any]):
                    Flattened user configuration, including defaults and overrides,
                    used by non-VERL components (e.g. generation config).
                - "verl_config" (Dict[str, Any]):
                    VERL-specific configuration mapped into a hierarchical dict
                    structure, intended to be merged with VERL defaults later.
        """
        logger.info("Starting VERL configuration processing.")
 
        global_config = self._validate_config()
 
        agentic_rl_config = AgenticRLConfig()
        agentic_rl_config.agent_name = global_config.agent_name
        agentic_rl_config.agent_engine_wrapper_path = global_config.agent_engine_wrapper_path
        agentic_rl_config.train_backend = global_config.train_backend
        agentic_rl_config.load_format = 'auto'
 
        input_config: Dict[str, Any] = global_config.model_dump()
 
        # Ensure dataset_additional_keys has a default value if missing
        dataset_keys: List[str] = input_config.get("dataset_additional_keys")
        if not dataset_keys:
            logger.info("dataset_additional_keys not found or empty. Defaulting to ['labels'].")
            input_config["dataset_additional_keys"] = ["labels"]
 
        verl_config = {}
        merged_config = input_config.copy()
        if global_config.verl:
            verl_model_dump = global_config.verl.model_dump()
            merged_config.update(verl_model_dump)
            input_config['tensorboard_flush_interval'] = verl_model_dump.get("tensorboard_flush_interval", 50)
 
        self._apply_config_mapping(verl_config, merged_config)
 
        logger.info("VERL configuration processing completed.")
        return {"agentic_rl_config": agentic_rl_config, "input_config": input_config, "verl_config": verl_config}
 
    def process_config(self):
        """
        Perform the full configuration processing pipeline.

        This function:
        1. Parses and splits the user input configuration.
        2. Merges user-provided VERL config with VERL default config.
        3. Applies mandatory VERL constraints (in-place mutation).
        4. Generates the final generation-related configuration.

        Returns:
            Tuple[
                AgenticRLConfig,
                Dict[str, Any],
                DictConfig,
                Any
            ]:
                - agentic_rl_config:
                    Runtime configuration for Agentic RL execution.
                - input_cfg:
                    Flattened input configuration dictionary.
                - res_verl_cfg:
                    Final VERL configuration (OmegaConf DictConfig),
                    after merging defaults and enforcing constraints.
                - gen_cfg:
                    Generation-related configuration derived from input_cfg.
        """
        parsed_input_cfg = self._process_input_config()
        agentic_rl_config = parsed_input_cfg["agentic_rl_config"]
        input_cfg = parsed_input_cfg["input_config"]
        verl_cfg = parsed_input_cfg["verl_config"]
        default_cfg = OmegaConf.create(DEFAULT_CONFIG)
        usr_cfg = OmegaConf.create(verl_cfg)
        res_verl_cfg = OmegaConf.merge(default_cfg, usr_cfg)
        _validate_verl_constraints(res_verl_cfg)
        gen_cfg = _gen_gen_config(input_cfg)
        return agentic_rl_config, input_cfg, res_verl_cfg, gen_cfg
 
    def _apply_config_mapping(self, target_config: Dict[str, Any], source_config: Dict[str, Any]) -> None:
        """Applies the configuration mapping from source to target.
 
        Args:
            target_config: The dictionary to update with nested values.
            source_config: The flat dictionary containing source values.
        """
        for source_key, target_path in self._CONFIG_MAPPING.items():
            if source_key in source_config:
                value = source_config[source_key]
                self._set_nested_value(target_config, target_path, value)

