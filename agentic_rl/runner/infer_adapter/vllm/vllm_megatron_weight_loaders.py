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

from agentic_rl.base.weight_loaders.megatron_weight_loaders import BaseMegatronWeightLoader

try:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE as _VLLM_FusedMoE  # type: ignore
    FusedMoE = _VLLM_FusedMoE  # expose as module-level name for patching
except ImportError:  # pragma: no cover - fallback when fused_moe isn't available
    class FusedMoE:  # type: ignore
        pass


class VllmMegatronWeightLoaders(BaseMegatronWeightLoader):
    def __init__(self):
        super().__init__()
        try:
            self.register_model_loader("Qwen2ForCausalLM",
                                       BaseMegatronWeightLoader.qwen_megatron_weight_loader)
            self.register_model_loader("CustomQwen2ForCausalLM",
                                       BaseMegatronWeightLoader.qwen_megatron_weight_loader)
            self.register_model_loader("Qwen3ForCausalLM",
                                       BaseMegatronWeightLoader.qwen_megatron_weight_loader)
            self.register_model_loader("CustomQwen3ForCausalLM",
                                       BaseMegatronWeightLoader.qwen_megatron_weight_loader)
        except (AttributeError, TypeError) as e:
            raise RuntimeError(f"Failed to register model loaders: {e}") from e

    def get_supported_architectures(self):
        try:
            from vllm.model_executor.models import ModelRegistry
            return ModelRegistry.get_supported_archs()
        except ImportError as e:
            raise ImportError(f"Failed to import vllm.model_executor.models.ModelRegistry: {e}") from e
        except AttributeError as e:
            raise AttributeError(f"ModelRegistry.get_supported_archs() is not available: {e}") from e

    def update_megatron_weight_loader(self):
        try:
            from vllm.model_executor.layers.linear import (
                ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear,
                RowParallelLinear, ReplicatedLinear
            )
            from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
        except ImportError as e:
            raise ImportError(f"Failed to import vllm layer classes: {e}") from e

        layer_weight_megatron_loader_registry = {
            ColumnParallelLinear: BaseMegatronWeightLoader.parallel_weight_loader,
            MergedColumnParallelLinear: BaseMegatronWeightLoader.parallel_weight_loader,
            QKVParallelLinear: BaseMegatronWeightLoader.parallel_weight_loader,
            RowParallelLinear: BaseMegatronWeightLoader.parallel_weight_loader,
            VocabParallelEmbedding: BaseMegatronWeightLoader.parallel_weight_loader,
            ParallelLMHead: BaseMegatronWeightLoader.parallel_weight_loader,
            ReplicatedLinear: BaseMegatronWeightLoader.parallel_weight_loader,
            FusedMoE: BaseMegatronWeightLoader.parallel_weight_loader
        }

        for layer_class, weight_loader in layer_weight_megatron_loader_registry.items():
            try:
                layer_class.weight_loader = weight_loader
            except (AttributeError, TypeError) as e:
                raise RuntimeError(f"Failed to set weight_loader for {layer_class.__name__}: {e}") from e
