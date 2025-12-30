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
from typing import Dict
import gc
import torch

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.checker import validate_params

logger = Loggers(__name__)


class MemoryManager:
    """Manages memory operations for VLLM inference engine."""

    def __init__(self):
        self.cpu_model: Dict[str, torch.Tensor] = {}

    @staticmethod
    def clear_gpu_memory():
        """
        Clear GPU memory and run garbage collection.

        Note:
            If CUDA cache clearing fails (e.g., CUDA not available), logs a warning
            but continues with garbage collection.
        """
        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            logger.warning(f"Failed to clear CUDA cache (CUDA may not be available): {e}")
        except Exception as e:
            logger.warning(f"Unexpected error clearing CUDA cache: {e}")

        try:
            gc.collect()
        except ValueError as e:
            logger.warning(f"Garbage collection failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during running garbage collection: {e}")

    @staticmethod
    def _clear_mla_weights(model):
        """
        Clear MLA (Multi-Head Latent Attention) weights if present.

        Args:
            model: The model instance

        Note:
            This method logs warnings for individual layer failures but continues
            processing to ensure other layers are cleared even if one fails.
        """
        try:
            if not (hasattr(model, 'model') and hasattr(model.model.layers[-1].self_attn, "mla_attn")):
                return
        except (AttributeError, IndexError) as e:
            logger.warning(f"Model does not have MLA structure, skipping MLA weight clearing: {e}")
            return
        except Exception as e:
            logger.warning(f"Unexpected error checking for MLA structure: {e}")
            return

        def _process_one_layer(idx):
            try:
                mla = model.model.layers[idx].self_attn.mla_attn.impl
                if hasattr(mla, "w_kc"):
                    mla.w_kc = None
                    mla.w_vc = None
                if hasattr(mla, "W_UV"):
                    mla.W_UV = None
                    mla.W_UK_T = None
            except (IndexError, AttributeError) as e:
                logger.warning(f"Failed to clear MLA weights for layer {idx}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error clearing MLA weights for layer {idx}: {e}")

        for i in range(model.model.start_layer, model.model.end_layer):
            _process_one_layer(i)

    @staticmethod
    def _get_rank_from_env() -> int:
        """
        Get rank from environment variable.

        Returns:
            int: The rank value from RANK environment variable

        Raises:
            KeyError: If RANK environment variable is not set
            ValueError: If RANK value is not a valid integer
        """
        import os

        try:
            rank_str = os.environ['RANK']
        except KeyError as e:
            logger.error("RANK environment variable is not set")
            raise KeyError("RANK environment variable is required but not set") from e

        try:
            rank_value = int(rank_str)
            if rank_value < 0 or rank_value >= 8:
                raise ValueError("RANK environment variable must be an integer in [0, 8)")

            return rank_value
        except ValueError as e:
            raise ValueError(f"RANK environment variable must be an integer, got: '{rank_str}'") from e

    @validate_params(
        model=dict(
            validator=lambda x: x is not None and hasattr(x, "named_parameters"),
            message="model must be not None and have named_parameters method",
        ),
    )
    def create_cpu_model_copy(self, model) -> Dict[str, torch.Tensor]:
        """
        Create a CPU copy of model parameters.

        Args:
            model: The model instance

        Returns:
            Dict mapping parameter names to CPU tensors

        Raises:
            RuntimeError: If tensor allocation fails (e.g., out of memory)
            Exception: If iteration over model parameters fails
        """
        cpu_model = {}
        try:
            for name, params in model.named_parameters():
                cpu_model[name] = torch.empty_like(params, device="cpu")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to allocate CPU memory: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during creating CPU model copy: {e}") from e
        
        logger.info(f"Successfully created CPU model copy with {len(cpu_model)} parameters")
        return cpu_model

    @validate_params(
        model=dict(
            validator=lambda x: x is not None and hasattr(x, "named_parameters"),
            message="model must be not None and have named_parameters method",
        ),
        cpu_model=dict(validator=lambda x: isinstance(x, dict), message="cpu_model must be a dict"),
    )
    def offload_model_weights(self, model, cpu_model: Dict[str, torch.Tensor]):
        """
        Offload model weights to CPU.

        Args:
            model: The model instance
            cpu_model: CPU model parameter storage

        Raises:
            RuntimeError: If rank retrieval fails or parameter offloading fails
            KeyError: If cpu_model is missing a required parameter
        """
        try:
            rank = int(self._get_rank_from_env())
            logger.info(f"offload_model_weights rank={rank}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Failed to retrieve rank for offload operation: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error when retrieving rank for offload operation: {e}") from e

        def _process_parameter(name, params):
            try:
                if name not in cpu_model:
                    raise KeyError(f"Parameter '{name}' not found in cpu_model")
                params.data = cpu_model[name]
            except KeyError as e:
                raise RuntimeError(f"Cannot offload parameter '{name}': not found in cpu_model") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error: failed to offload parameter '{name}': {e}") from e

        try:
            for name, params in model.named_parameters():
                _process_parameter(name, params)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error: model weight offloading failed: {e}") from e

        self._clear_mla_weights(model)
