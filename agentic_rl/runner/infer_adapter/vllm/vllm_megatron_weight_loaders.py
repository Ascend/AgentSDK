#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------


from agentic_rl.base.weight_loaders.megatron_weight_loaders import BaseMegatronWeightLoader


class VllmMegatronWeightLoaders(BaseMegatronWeightLoader):
    def __init__(self):
        super().__init__()
        self.register_model_loader("LlamaForCausalLM",
                                   BaseMegatronWeightLoader.llama_megatron_core_weight_loader)
        self.register_model_loader("Qwen2ForCausalLM",
                                   BaseMegatronWeightLoader.qwen_megatron_weight_loader)
        self.register_model_loader("CustomQwen2ForCausalLM",
                                   BaseMegatronWeightLoader.qwen_megatron_weight_loader)
        self.register_model_loader("Qwen3ForCausalLM",
                                   BaseMegatronWeightLoader.qwen_megatron_weight_loader)
        self.register_model_loader("DeepseekV3ForCausalLM",
                                   BaseMegatronWeightLoader.deepseek_megatron_weight_loader)
        self.register_model_loader("DeepseekV2ForCausalLM",
                                   BaseMegatronWeightLoader.deepseek_megatron_weight_loader)
        self.register_model_loader("CustomDeepseekV3ForCausalLM",
                                   BaseMegatronWeightLoader.deepseek_megatron_weight_loader)
        self.register_model_loader("Qwen2_5_VLForConditionalGeneration",
                                   BaseMegatronWeightLoader.qwen_vl_megatron_weight_loader)
        self.register_model_loader("CustomQwen3MoeForCausalLM",
                                   BaseMegatronWeightLoader.qwen_megatron_weight_loader)

    def get_supported_architectures(self):
        from vllm.model_executor.models import ModelRegistry
        return ModelRegistry.get_supported_archs()

    def update_megatron_weight_loader(self):
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear,
            RowParallelLinear, ReplicatedLinear
        )
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

        LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY = {
            ColumnParallelLinear: BaseMegatronWeightLoader.parallel_weight_loader,
            MergedColumnParallelLinear: BaseMegatronWeightLoader.parallel_weight_loader,
            QKVParallelLinear: BaseMegatronWeightLoader.parallel_weight_loader,
            RowParallelLinear: BaseMegatronWeightLoader.parallel_weight_loader,
            VocabParallelEmbedding: BaseMegatronWeightLoader.parallel_weight_loader,
            ParallelLMHead: BaseMegatronWeightLoader.parallel_weight_loader,
            ReplicatedLinear: BaseMegatronWeightLoader.parallel_weight_loader,
            FusedMoE: BaseMegatronWeightLoader.parallel_weight_loader
        }

        for layer_class, weight_loader in LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY.items():
            layer_class.weight_loader = weight_loader
