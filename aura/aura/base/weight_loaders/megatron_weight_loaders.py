#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig


class InferParallelConfig:
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
        self.model_megatron_weight_loader_registry = {}  # 框架特定模型加载器注册表

    def register_model_loader(self, model_key: str, loader_func):
        """Registration model loading method"""
        self.model_megatron_weight_loader_registry[model_key] = loader_func

    def load_megatron_weights(self, actor_weights: Dict, model: nn.Module,
                              infer_parallel_config: InferParallelConfig, hf_config: PretrainedConfig) -> nn.Module:
        """General weight loading entry"""
        model_weight_loader = self._get_model_weight_loader(model.__class__.__name__)
        model = model_weight_loader(actor_weights, model, infer_parallel_config, hf_config)
        # NOTE(sgm) to reduce peak memory usage, we offload vllm model to cpu
        # after init, and we need this after sync model weights for in first iter.
        model = self.finalize_loading(model)
        return model

    def _get_model_weight_loader(self, arch: str):
        if arch in self.model_megatron_weight_loader_registry:
            return self.model_megatron_weight_loader_registry[arch]
        supported_arches = self.get_supported_architectures()
        raise ValueError(f"Model architectures {arch} are not supported for now. "
                         f"Supported architectures: {supported_arches}")

    @abstractmethod
    def get_supported_architectures(self):
        """The method for obtaining the supported architecture (implemented by the subclass)"""
        pass

    @abstractmethod
    def update_megatron_weight_loader(self):
        """Update method (implemented by subclasses)"""
        pass

    @staticmethod
    def llama_megatron_core_weight_loader(
            actor_weights: Dict, model: nn.Module,
            infer_parallel_config: InferParallelConfig,
            hf_config: PretrainedConfig
    ) -> nn.Module:
        params_dict = dict(model.named_parameters())
        for name, loaded_weight in actor_weights.items():
            if name.endswith(".bias") and name not in params_dict:
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if name not in params_dict.keys():
                continue
            if "lm_head" in name:  # lm_head is not needed since it is tied with embedding
                continue
            if "qkv" in name:
                q_weight, k_weight, v_weight = BaseMegatronWeightLoader.qkv_split_weight(loaded_weight,
                                                                                         infer_parallel_config,
                                                                                         hf_config)
                loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
            BaseMegatronWeightLoader.load_single_weight(params_dict, name, loaded_weight)
        return model

    @staticmethod
    def qwen_megatron_weight_loader(
            actor_weights: Dict,
            model: nn.Module,
            infer_parallel_config: InferParallelConfig,
            hf_config: PretrainedConfig
    ) -> nn.Module:
        params_dict = dict(model.named_parameters())
        for name, loaded_weight in actor_weights.items():
            if name not in params_dict.keys():
                continue
            if "qkv" in name:
                if name.endswith('.bias'):
                    q_weight, k_weight, v_weight = (
                        BaseMegatronWeightLoader.qkv_split_bias(loaded_weight, infer_parallel_config, hf_config))
                    loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
                else:
                    q_weight, k_weight, v_weight = (
                        BaseMegatronWeightLoader.qkv_split_weight(loaded_weight, infer_parallel_config, hf_config))
                    loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
            if "mlp.experts.w13_weight" in name:
                tmp = loaded_weight.view(
                    hf_config.num_experts // infer_parallel_config.infer_expert_parallel_size,
                    hf_config.hidden_size,
                    -1
                ).transpose(2, 1).contiguous()
                # maybe w13_weight has been transposed by vllm
                if loaded_weight.shape != tmp.shape:
                    tmp = tmp.transpose(2, 1).contiguous()
                loaded_weight.copy_(tmp)
            if "mlp.experts.w2_weight" in name:
                tmp = loaded_weight.view(
                    hf_config.num_experts // infer_parallel_config.infer_expert_parallel_size,
                    -1,
                    hf_config.hidden_size
                ).transpose(2, 1).contiguous()
                if loaded_weight.shape != tmp.shape:
                    tmp = tmp.transpose(2, 1).contiguous()
                loaded_weight.copy_(tmp)
            BaseMegatronWeightLoader.load_single_weight(params_dict, name, loaded_weight)
        return model

    @staticmethod
    def qwen_vl_megatron_weight_loader(
            actor_weights: Dict,
            model: nn.Module,
            infer_parallel_config: InferParallelConfig,
            hf_config: PretrainedConfig
    ) -> nn.Module:
        params_dict = dict(model.named_parameters())
        vision_config = type('obj', (object,), {
            'num_attention_heads': hf_config.vision_config.num_heads,
            'num_key_value_heads': hf_config.vision_config.num_heads,
        })

        for name, loaded_weight in actor_weights.items():
            if name not in params_dict.keys():
                continue
            if "qkv" in name:
                if 'visual' in name:
                    if name.endswith('.bias'):
                        q_weight, k_weight, v_weight = BaseMegatronWeightLoader.qkv_split_bias(loaded_weight,
                                                                                               infer_parallel_config,
                                                                                               vision_config)
                        loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
                    else:
                        q_weight, k_weight, v_weight = BaseMegatronWeightLoader.qkv_split_weight(loaded_weight,
                                                                                                 infer_parallel_config,
                                                                                                 vision_config)
                        loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
                else:
                    if name.endswith('.bias'):
                        q_weight, k_weight, v_weight = BaseMegatronWeightLoader.qkv_split_bias(loaded_weight,
                                                                                               infer_parallel_config,
                                                                                               hf_config)
                        loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
                    else:
                        q_weight, k_weight, v_weight = BaseMegatronWeightLoader.qkv_split_weight(loaded_weight,
                                                                                                 infer_parallel_config,
                                                                                                 hf_config)
                        loaded_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))

            BaseMegatronWeightLoader.load_single_weight(params_dict, name, loaded_weight)

        return model

    @staticmethod
    def deepseek_megatron_weight_loader(
            actor_weights: Dict,
            model: nn.Module,
            infer_parallel_config: InferParallelConfig,
            hf_config: PretrainedConfig
    ) -> nn.Module:
        params_dict = dict(model.named_parameters())
        for name, loaded_weight in actor_weights.items():
            if "qkv" in name:
                split_dim = hf_config.q_lora_rank if hf_config.q_lora_rank else \
                    (hf_config.qk_nope_head_dim + hf_config.qk_rope_head_dim) * hf_config.num_attention_heads
                q_name = name.replace("qkv_proj", "q_a_proj" if hf_config.q_lora_rank else "q_proj")
                kv_name = name.replace("qkv_proj", "kv_a_proj_with_mqa")
                BaseMegatronWeightLoader.load_single_weight(params_dict, q_name, loaded_weight[:split_dim])
                BaseMegatronWeightLoader.load_single_weight(params_dict, kv_name, loaded_weight[split_dim:])
                continue
            if name not in params_dict.keys():
                raise ValueError(f"unexpected key {name} in deepseek_megatron_weight_loader")
            if "mlp.experts.w13_weight" in name:
                loaded_weight.copy_(
                    loaded_weight.view(hf_config.n_routed_experts // infer_parallel_config.infer_expert_parallel_size,
                                       hf_config.hidden_size, -1).transpose(2, 1).contiguous())
            if "mlp.experts.w2_weight" in name:
                loaded_weight.copy_(
                    loaded_weight.view(hf_config.n_routed_experts // infer_parallel_config.infer_expert_parallel_size,
                                       -1, hf_config.hidden_size).transpose(2, 1).contiguous())
            BaseMegatronWeightLoader.load_single_weight(params_dict, name, loaded_weight)
        return model

    @staticmethod
    def qkv_split_weight(
            query_key_value: torch.Tensor,
            infer_parallel_config: InferParallelConfig,
            hf_config: PretrainedConfig
    ) -> torch.Tensor:
        """General QKV Weight Partitioning Logic"""
        infer_tensor_parallel_size = infer_parallel_config.infer_tensor_parallel_size
        nh = hf_config.num_attention_heads // infer_tensor_parallel_size
        ng = max(hf_config.num_key_value_heads // infer_tensor_parallel_size, 1)
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

    @staticmethod
    def qkv_split_bias(
            query_key_value: torch.Tensor,
            infer_parallel_config: InferParallelConfig,
            hf_config: PretrainedConfig
    ) -> torch.Tensor:
        infer_tensor_parallel_size = infer_parallel_config.infer_tensor_parallel_size
        nh = hf_config.num_attention_heads // infer_tensor_parallel_size
        ng = max(hf_config.num_key_value_heads // infer_tensor_parallel_size, 1)
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

    @staticmethod
    def load_single_weight(params_dict: Dict, name: str, loaded_weight: torch.Tensor):
        """General parameter loading logic"""
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", BaseMegatronWeightLoader.default_weight_loader)
        weight_loader(param, loaded_weight)

    @staticmethod
    def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        """Default weight loader"""
        if param.size() != loaded_weight.size():
            raise ValueError("The parameter size does not match the loaded weight size.")
        if param.data.dtype != loaded_weight.data.dtype:
            raise ValueError("if we want to shared weights, the data type should also be the same")
        param.data = loaded_weight.data

    def parallel_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        """Parallel Linear weight loader."""
        if param.size() != loaded_weight.size():
            error_msg = (
                f"the parameter size is not align with the loaded weight size, param size: {param.size()}, "
                f"loaded_weight size: {loaded_weight.size()}"
            )
            raise ValueError(error_msg)
        if param.data.dtype != loaded_weight.data.dtype:
            raise ValueError("if we want to shared weights, the data type should also be the same")
        param.data = loaded_weight.data

    def finalize_loading(self, model: nn.Module) -> nn.Module:
        """Post-processing after loading"""
        return model.cuda()
