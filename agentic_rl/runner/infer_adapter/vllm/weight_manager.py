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

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

from agentic_rl.base.weight_loaders.megatron_weight_loaders import InferParallelConfig
from agentic_rl.runner.infer_adapter.vllm.vllm_megatron_weight_loaders import VllmMegatronWeightLoaders
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.checker import validate_params

logger = Loggers(__name__)


class WeightManager:
    """Manages model weight loading and synchronization for VLLM inference engine."""

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
        load_format=dict(
            validator=lambda x: isinstance(x, str) and len(x) > 0,
            message="load_format must be a non-empty string"
        ),
    )
    def __init__(
        self,
        infer_tensor_parallel_size: int,
        infer_pipeline_parallel_size: int,
        infer_expert_parallel_size: int,
        load_format: str = "megatron",
    ):
        self.infer_tensor_parallel_size = infer_tensor_parallel_size
        self.infer_pipeline_parallel_size = infer_pipeline_parallel_size
        self.infer_expert_parallel_size = infer_expert_parallel_size
        self.load_format = load_format
        try:
            self.vllm_megatron_weight_loaders = VllmMegatronWeightLoaders()
        except RuntimeError as e:
            raise RuntimeError(f"WeightManager initialization failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error: WeightManager initialization failed: {e}") from e

    @staticmethod
    def _process_mla_weights(model):
        """Process MLA (Multi-Head Latent Attention) weights after loading."""
        try:
            if not hasattr(model, "model"):
                logger.warning("Model does not have 'model' attribute, skipping MLA weight processing")
                return

            if not hasattr(model.model, "start_layer") or not hasattr(model.model, "end_layer"):
                logger.warning(
                    "Model does not have start_layer or end_layer attributes, skipping MLA weight processing"
                )
                return

            for i in range(model.model.start_layer, model.model.end_layer):
                mla = model.model.layers[i].self_attn.mla_attn.impl
                if hasattr(mla, "w_kc"):
                    mla.w_kc = None
                    mla.w_vc = None
                if hasattr(mla, "W_UV"):
                    mla.W_UV = None
                    mla.W_UK_T = None
                mla.process_weights_after_loading(None)
        except (AttributeError, IndexError) as e:
            raise RuntimeError(f"Failed to process MLA weights due to missing attribute: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error: MLA weight processing failed: {e}") from e

    def initialize_weight_loader(self):
        """Initialize the weight loader based on the load format."""
        try:
            if self.load_format == "megatron":
                self.vllm_megatron_weight_loaders.update_megatron_weight_loader()
            else:
                logger.warning(f"Unsupported load format: {self.load_format}. No weight loader initialized.")
        except AttributeError as e:
            raise RuntimeError(f"Failed to initialize weight loader due to missing attribute: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during weight loader initialization: {e}") from e

    @validate_params(
        params=dict(validator=lambda x: isinstance(x, dict), message="params must be a dict"),
        model=dict(validator=lambda x: isinstance(x, nn.Module), message="model must be a nn.Module instance"),
        hf_config=dict(
            validator=lambda x: isinstance(x, PretrainedConfig),
            message="hf_config must be a PretrainedConfig instance"
        ),
    )
    def load_megatron_weights(self, params, model, hf_config):
        """
        Load weights using Megatron format.

        Args:
            params: Model parameters
            model: The model instance
            hf_config: HuggingFace model configuration
        """
        try:
            infer_parallel_config = InferParallelConfig(
                self.infer_tensor_parallel_size,
                self.infer_pipeline_parallel_size,
                self.infer_expert_parallel_size * self.infer_tensor_parallel_size,
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to create InferParallelConfig: {e}")
            raise ValueError(f"Invalid parallel configuration parameters: {e}") from e

        try:
            self.vllm_megatron_weight_loaders.load_megatron_weights(params, model, infer_parallel_config, hf_config)
            logger.info("Successfully loaded Megatron weights")
        except FileNotFoundError as e:
            logger.error(f"Weight files not found: {e}")
            raise FileNotFoundError(f"Failed to load weights - files not found: {e}") from e
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid weight format or configuration: {e}")
            raise ValueError(f"Failed to load weights - invalid format: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error loading Megatron weights: {e}")
            raise RuntimeError(f"Unexpected error: Weight loading failed: {e}") from e

        # Process MLA weights if present
        try:
            if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
                if hasattr(model.model.layers[0], "self_attn") and hasattr(model.model.layers[0].self_attn, "mla_attn"):
                    self._process_mla_weights(model)
                    logger.info("Successfully processed MLA weights")
        except (AttributeError, IndexError) as e:
            logger.warning(f"Could not check for MLA weights - model structure unexpected: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred during MLA weight check: {e}") from e
