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
from typing import Dict, Callable
import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.checker import validate_params

logger = Loggers(__name__)


class InferParallelConfig:
    @validate_params(
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
    )
    def __init__(
        self,
         infer_tensor_parallel_size: int,
         infer_pipeline_parallel_size: int,
         infer_expert_parallel_size: int
    ):
        self.infer_tensor_parallel_size = infer_tensor_parallel_size
        self.infer_pipeline_parallel_size = infer_pipeline_parallel_size
        self.infer_expert_parallel_size = infer_expert_parallel_size


class BaseMegatronWeightLoader(ABC):
    def __init__(self):
        self.model_megatron_weight_loader_registry = {}  # Framework-specific model loader registry

    @staticmethod
    @validate_params(
        actor_weights=dict(validator=lambda x: isinstance(x, dict), message="actor_weights must be a dictionary"),
        model=dict(validator=lambda x: isinstance(x, nn.Module), message="model must be a nn.Module instance"),
        infer_parallel_config=dict(
            validator=lambda x: isinstance(x, InferParallelConfig),
            message="infer_parallel_config must be an instance of InferParallelConfig",
        ),
        hf_config=dict(
            validator=lambda x: isinstance(x, PretrainedConfig),
            message="hf_config must be an instance of PretrainedConfig",
        ),
    )
    def qwen_megatron_weight_loader(
        actor_weights: Dict, model: nn.Module, infer_parallel_config: InferParallelConfig, hf_config: PretrainedConfig
    ) -> nn.Module:
        params_dict = dict(model.named_parameters())

        for name, loaded_weight in actor_weights.items():
            if name not in params_dict.keys():
                logger.debug(f"Skipping weight {name} as it's not in model parameters")
                continue
            try:
                BaseMegatronWeightLoader.process_qkv_weight(name, loaded_weight, infer_parallel_config, hf_config)
                BaseMegatronWeightLoader.load_single_weight(params_dict, name, loaded_weight)
            except (RuntimeError, ValueError) as e:
                raise RuntimeError(f"Failed to load weight {name}: {str(e)}") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error occurred during loading weight {name}: {str(e)}") from e
        return model


    @staticmethod
    @validate_params(
        query_key_value=dict(
            validator=lambda x: isinstance(x, torch.Tensor), message="query_key_value must be a torch.Tensor"
        ),
        infer_parallel_config=dict(
            validator=lambda x: isinstance(x, InferParallelConfig),
            message="infer_parallel_config must be an instance of InferParallelConfig",
        ),
        hf_config=dict(
            validator=lambda x: isinstance(x, PretrainedConfig),
            message="hf_config must be an instance of PretrainedConfig",
        ),
    )
    def qkv_split_weight(
        query_key_value: torch.Tensor,
        infer_parallel_config: InferParallelConfig,
        hf_config: PretrainedConfig
    ) -> torch.Tensor:
        """Common QKV weight splitting logic"""
        try:
            infer_tensor_parallel_size = infer_parallel_config.infer_tensor_parallel_size

            # Validate configuration
            if not hasattr(hf_config, 'num_attention_heads') or not hasattr(hf_config, 'num_key_value_heads'):
                logger.error("Missing required attributes in hf_config")
                raise ValueError("hf_config must have num_attention_heads and num_key_value_heads attributes")

            if hf_config.num_attention_heads % infer_tensor_parallel_size != 0:
                logger.error(f"num_attention_heads ({hf_config.num_attention_heads}) must be divisible by "
                           f"infer_tensor_parallel_size ({infer_tensor_parallel_size})")
                raise ValueError("Incompatible attention heads and tensor parallel size")

            if hf_config.num_key_value_heads % infer_tensor_parallel_size != 0:
                logger.error(f"num_key_value_heads ({hf_config.num_key_value_heads}) must be divisible by "
                           f"infer_tensor_parallel_size ({infer_tensor_parallel_size})")
                raise ValueError("Incompatible key-value heads and tensor parallel size")

            nh = hf_config.num_attention_heads // infer_tensor_parallel_size
            ng = hf_config.num_key_value_heads // infer_tensor_parallel_size

            if ng == 0:
                raise ValueError("Invalid parallel configuration resulting in zero heads per GPU")

            repeats = nh // ng
            qkv_weight = query_key_value.reshape(
                ng,
                repeats + 2,
                query_key_value.shape[0] // ng // (repeats + 2),
                query_key_value.shape[1],
            )
            hidden_size = qkv_weight.shape[-1]
            qw = qkv_weight[:, :repeats, ...].reshape(-1, hidden_size)
            kw = qkv_weight[:, repeats: repeats + 1, ...].reshape(-1, hidden_size)
            vw = qkv_weight[:, repeats + 1:, ...].reshape(-1, hidden_size)
            return qw, kw, vw
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Failed to split QKV weight due to tensor operation: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred during splitting QKV weight: {str(e)}") from e

    @staticmethod
    @validate_params(
        query_key_value=dict(
            validator=lambda x: isinstance(x, torch.Tensor), message="query_key_value must be a torch.Tensor"
        ),
        infer_parallel_config=dict(
            validator=lambda x: isinstance(x, InferParallelConfig),
            message="infer_parallel_config must be an instance of InferParallelConfig",
        ),
        hf_config=dict(
            validator=lambda x: isinstance(x, PretrainedConfig),
            message="hf_config must be an instance of PretrainedConfig",
        ),
    )
    def qkv_split_bias(
        query_key_value: torch.Tensor,
        infer_parallel_config: InferParallelConfig,
        hf_config: PretrainedConfig
    ) -> torch.Tensor:
        try:
            infer_tensor_parallel_size = infer_parallel_config.infer_tensor_parallel_size

            # Validate configuration
            if not hasattr(hf_config, 'num_attention_heads') or not hasattr(hf_config, 'num_key_value_heads'):
                logger.error("Missing required attributes in hf_config")
                raise ValueError("hf_config must have num_attention_heads and num_key_value_heads attributes")

            if hf_config.num_attention_heads % infer_tensor_parallel_size != 0:
                logger.error(f"num_attention_heads ({hf_config.num_attention_heads}) must be divisible by "
                           f"infer_tensor_parallel_size ({infer_tensor_parallel_size})")
                raise ValueError("Incompatible attention heads and tensor parallel size")

            if hf_config.num_key_value_heads % infer_tensor_parallel_size != 0:
                logger.error(f"num_key_value_heads ({hf_config.num_key_value_heads}) must be divisible by "
                           f"infer_tensor_parallel_size ({infer_tensor_parallel_size})")
                raise ValueError("Incompatible key-value heads and tensor parallel size")

            nh = hf_config.num_attention_heads // infer_tensor_parallel_size
            ng = hf_config.num_key_value_heads // infer_tensor_parallel_size

            if ng == 0:
                logger.error("Number of key-value heads per GPU is zero")
                raise ValueError("Invalid parallel configuration resulting in zero heads per GPU")

            repeats = nh // ng
            bias_weight = query_key_value.reshape(
                ng,
                repeats + 2,
                query_key_value.shape[0] // ng // (repeats + 2)
            )
            qw = bias_weight[:, :repeats, ...].reshape(-1)
            kw = bias_weight[:, repeats: repeats + 1, ...].reshape(-1)
            vw = bias_weight[:, repeats + 1:, ...].reshape(-1)
            return qw, kw, vw
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Failed to split QKV bias due to tensor operation: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred during splitting QKV bias: {str(e)}") from e


    @staticmethod
    @validate_params(
        name=dict(validator=lambda x: isinstance(x, str), message="name must be a string"),
        loaded_weight=dict(
            validator=lambda x: isinstance(x, torch.Tensor), message="loaded_weight must be a torch.Tensor"
        ),
        infer_parallel_config=dict(
            validator=lambda x: isinstance(x, InferParallelConfig),
            message="infer_parallel_config must be an instance of InferParallelConfig",
        ),
        hf_config=dict(
            validator=lambda x: isinstance(x, PretrainedConfig),
            message="hf_config must be an instance of PretrainedConfig",
        ),
    )
    def process_qkv_weight(
        name: str,
        loaded_weight: torch.Tensor,
        infer_parallel_config: InferParallelConfig,
        hf_config: PretrainedConfig
    ) -> None:
        """Process QKV weight by splitting and concatenating Q, K, V components."""
        if "qkv" not in name:
            return

        if name.endswith(".bias"):
            q_weight, k_weight, v_weight = BaseMegatronWeightLoader.qkv_split_bias(
                loaded_weight, infer_parallel_config, hf_config
            )
            loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
        else:
            q_weight, k_weight, v_weight = BaseMegatronWeightLoader.qkv_split_weight(
                loaded_weight, infer_parallel_config, hf_config
            )
            loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))

    @staticmethod
    @validate_params(
        params_dict=dict(validator=lambda x: isinstance(x, dict), message="params_dict must be a dictionary"),
        name=dict(validator=lambda x: isinstance(x, str), message="name must be a string"),
        loaded_weight=dict(
            validator=lambda x: isinstance(x, torch.Tensor), message="loaded_weight must be a torch.Tensor"
        ),
    )
    def load_single_weight(params_dict: Dict, name: str, loaded_weight: torch.Tensor):
        """Common parameter loading logic"""
        if name not in params_dict:
            logger.warning(f"Parameter {name} not found in params_dict")
            return
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", BaseMegatronWeightLoader.default_weight_loader)
        try:
            weight_loader(param, loaded_weight)
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Failed to load single weight: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred during loading single weight: {str(e)}") from e

    @staticmethod
    @validate_params(
        param=dict(validator=lambda x: isinstance(x, torch.Tensor), message="param must be a torch.Tensor"),
        loaded_weight=dict(
            validator=lambda x: isinstance(x, torch.Tensor), message="loaded_weight must be a torch.Tensor"
        ),
    )
    def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        """Default weight loader"""
        try:
            if param.size() != loaded_weight.size():
                raise ValueError("The parameter size does not match the loaded weight size.")
            if param.data.dtype != loaded_weight.data.dtype:
                raise ValueError("if we want to shared weights, the data type should also be the same")
            param.data = loaded_weight.data
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Failed to load weight: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred during loading weight: {str(e)}") from e

    @abstractmethod
    def get_supported_architectures(self):
        """Get supported architectures (implemented by subclass)"""
        pass

    @abstractmethod
    def update_megatron_weight_loader(self):
        """Update method (implemented by subclass)"""
        pass

    @validate_params(
        model_key=dict(validator=lambda x: isinstance(x, str), message="model_key must be a string"),
        loader_func=dict(validator=lambda x: callable(x), message="loader_func must be a callable"),
    )
    def register_model_loader(self, model_key: str, loader_func: Callable):
        """Register model loader method"""
        if model_key in self.model_megatron_weight_loader_registry:
            logger.warning(f"Overriding existing loader for model_key: {model_key}")
        self.model_megatron_weight_loader_registry[model_key] = loader_func
        logger.info(f"Successfully registered loader for model: {model_key}")

    @validate_params(
        actor_weights=dict(validator=lambda x: isinstance(x, dict), message="actor_weights must be a dictionary"),
        model=dict(validator=lambda x: isinstance(x, nn.Module), message="model must be a nn.Module instance"),
        infer_parallel_config=dict(
            validator=lambda x: isinstance(x, InferParallelConfig),
            message="infer_parallel_config must be an instance of InferParallelConfig",
        ),
        hf_config=dict(
            validator=lambda x: isinstance(x, PretrainedConfig),
            message="hf_config must be an instance of PretrainedConfig",
        ),
    )
    def load_megatron_weights(self, actor_weights: Dict, model: nn.Module,
                              infer_parallel_config: InferParallelConfig, hf_config: PretrainedConfig) -> nn.Module:
        """Common entry point for weight loading"""
        try:
            model_weight_loader = self._get_model_weight_loader(model.__class__.__name__)
            model = model_weight_loader(actor_weights, model, infer_parallel_config, hf_config)
            # NOTE(sgm) to reduce peak memory usage, we offload vllm model to cpu
            # after init, and we need this after sync model weights for in first iter.
            model = self.finalize_loading(model)
            return model
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Failed to load Megatron weights: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred during loading Megatron weights: {str(e)}") from e

    @validate_params(
        param=dict(validator=lambda x: isinstance(x, torch.Tensor), message="param must be a torch.Tensor"),
        loaded_weight=dict(
            validator=lambda x: isinstance(x, torch.Tensor), message="loaded_weight must be a torch.Tensor"
        ),
    )
    def parallel_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        """Parallel Linear weight loader."""
        try:
            if param.size() != loaded_weight.size():
                raise ValueError("The parameter size does not match the loaded weight size.")
            if param.data.dtype != loaded_weight.data.dtype:
                raise ValueError("if we want to shared weights, the data type should also be the same")
            param.data = loaded_weight.data
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Failed to load parallel weight due to tensor operation: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred during loading parallel weight: {str(e)}") from e

    @validate_params(
        model=dict(validator=lambda x: isinstance(x, nn.Module), message="model must be a nn.Module instance"),
    )
    def finalize_loading(self, model: nn.Module) -> nn.Module:
        """Post-loading processing"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check if CUDA is installed correctly.")
        return model.cuda()

    @validate_params(arch=dict(validator=lambda x: isinstance(x, str), message="arch must be a string"))
    def _get_model_weight_loader(self, arch: str):
        if arch not in self.model_megatron_weight_loader_registry:
            raise ValueError(
                f"Unsupported model arch: {arch}. Supported architectures: {self.get_supported_architectures()}."
            )
        return self.model_megatron_weight_loader_registry[arch]
