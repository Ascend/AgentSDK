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

import argparse
import json
import logging as logger
import os
from collections import defaultdict

import safetensors
import torch
import safetensors.torch

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)

HIDDEN_SIZE = 4096
NUM_EXPERTS = 128
NUM_ATTENTION_HEADS = 64
NUM_KEY_VALUE_HEADS = 4
HEAD_DIM = 128


# noinspection DuplicatedCode
class CkptConvert(object):
    """
    Converts a HuggingFace checkpoint to Megatron format.

    Args:
        hf_model_path (str): HuggingFace model path.
        mg_save_path (str): Megatron model save path.
        num_layers (int): Number of transformer layers.
        tp_size (int, optional): Degree of tensor model parallelism. Defaults to 1.
        pp_size (int, optional): Degree of pipeline model parallelism. Defaults to 1.
        ep_size (int, optional): Degree of expert model parallelism. Defaults to 1.
        vpp_stage (int, optional): The stage number in the virtual pipeline parallelism. Defaults to None.
        num_layer_list (str, optional): Specifies the number of parallel pipeline layers. If None, all blocks have the same number of layers. Defaults to None.
        noop_layers (str, optional): should be skipped during conversion. Defaults to None.
        moe_grouped_gemm (bool, optional): Whether to use grouped GEMM for MoE layers.
        moe_tp_extend_ep (bool, optional): Whether to use tp group to extend experts parallism.
    """

    def __init__(
        self,
        hf_model_path: str,
        mg_save_path: str,
        num_layers: int,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
        num_layer_list: str = None,
        noop_layers: str = None,
        vpp_stage: int = None,
        moe_grouped_gemm: bool = False,
        moe_tp_extend_ep: bool = False,
    ):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.num_layers = num_layers
        self.vpp_stage = vpp_stage
        if vpp_stage is not None:
            self.vpp_size = self.num_layers // self.pp_size // self.vpp_stage
        self.hf_model_path = hf_model_path
        self.mg_save_path = mg_save_path
        self.num_layer_list = num_layer_list
        self.noop_layers = noop_layers
        self.moe_grouped_gemm = moe_grouped_gemm
        self.moe_tp_extend_ep = moe_tp_extend_ep

        self.hidden_size = HIDDEN_SIZE
        self.num_experts = NUM_EXPERTS
        self.num_attention_heads = NUM_ATTENTION_HEADS
        self.num_key_value_heads = NUM_KEY_VALUE_HEADS
        self.head_dim = HEAD_DIM

        self._valid_parameter()

        if self.vpp_stage is None:
            self.pprank_layer_idxs = defaultdict()
            self.get_pprank_hf_layeridxs()
        else:
            self.vpprank_layer_idxs = defaultdict(dict)
            self.get_vpprank_hf_layeridxs()

    @staticmethod
    def qlora_nf4_weight(weight):
        """Quantize weights"""
        quantweight = bnb.nn.Params4bit(weight, requires_grad=weight.requires_grad, quant_type="nf4").to('npu').cpu()
        return quantweight.data, quantweight.quant_state

    @staticmethod
    def load_hf_model(file_path):
        """Load safetensors file"""
        return safetensors.torch.load_file(file_path)

    @staticmethod
    def mg_path_process(mg_path):
        """megatron model path"""
        iter_mg_path = os.path.join(mg_path, "iter_0000001")
        if not os.path.exists(mg_path):
            os.makedirs(mg_path, exist_ok=True)

        with open(os.path.join(mg_path, "latest_checkpointed_iteration.txt"), 'w') as f:
            f.write("1")
        return iter_mg_path

    def generate_mg_weights_dir(self, tp_rank, pp_rank, ep_rank):
        """Generate the megatron weight directory."""
        if self.ep_size == 1 and self.pp_size == 1:
            prefix = f"mp_rank_{tp_rank:02}"
        elif self.ep_size == 1:
            prefix = f"mp_rank_{tp_rank:02}_{pp_rank:03}"
        elif self.pp_size == 1:
            prefix = f"mp_rank_{tp_rank:02}_{ep_rank:03}"
        else:
            prefix = f"mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}"
        return prefix

    def _valid_parameter(self):
        if self.num_layer_list is None:
            if self.num_layers % self.pp_size != 0:
                raise ValueError(
                    f"number of layers ({self.num_layers}) should be divisible by the pipeline parallel size ({self.pp_size})"
                )
            if self.vpp_stage is not None and self.num_layers % self.pp_size % self.vpp_stage != 0:
                raise ValueError(
                    f"number of pp_stage ({self.num_layers % self.pp_size}) should be divisible by the vpp_stage ({self.vpp_stage})"
                )
        else:
            layer_list = list(map(int, self.num_layer_list.split(',')))

            if self.vpp_stage is not None:
                raise ValueError("num_layer_list and vpp cannot be configured at the same time")
            if len(layer_list) != self.pp_size:
                raise ValueError(
                    f"length of layer_list ({len(layer_list)}) should be equal to pipeline parallel size ({self.pp_size})"
                )
            if sum(layer_list) != self.num_layers:
                raise ValueError(
                    f"sum of layer_list ({sum(layer_list)}) should be equal to num_layers ({self.num_layers})"
                )
            if self.noop_layers is not None:
                raise ValueError("num_layer_list and noop_layers cannot be configured at the same time")
            if self.num_layers != 61:
                raise ValueError(
                    f"num_layer_list supports only full parameters, expected num_layers=61, got {self.num_layers}"
                )

    def get_layer_files_map(self):
        """layer -> safetensors file map"""
        layer_map_dict = defaultdict(set)
        weights_map_file_path = os.path.join(self.hf_model_path, "model.safetensors.index.json")

        with open(weights_map_file_path) as f:
            weights_map = json.load(f)
        weights_map = weights_map["weight_map"]

        for key, value in weights_map.items():
            if key.startswith("model.layers."):
                layer_name = int(key.split('model.layers.')[1].split('.')[0])
                layer_map_dict[layer_name].add(value)
            else:
                layer_map_dict[key].add(value)
        return layer_map_dict

    def get_pprank_hf_layeridxs(self) -> None:
        """pp_rank -> hf layer map"""
        num_noop_layers = 0 if self.noop_layers is None else len(list(map(int, self.noop_layers.split(","))))
        num_real_layers = self.num_layers - num_noop_layers
        num_layer_list_ = [i for i in range(num_real_layers)]

        if self.num_layer_list is None:
            layers_each_pp = [self.num_layers // self.pp_size] * self.pp_size
            if self.noop_layers is not None:
                for layer in list(map(int, self.noop_layers.split(","))):
                    cur_pp_rank = layer // (self.num_layers // self.pp_size)
                    layers_each_pp[cur_pp_rank] -= 1
        else:
            layers_each_pp = list(map(int, self.num_layer_list.split(',')))

        for pp_rank in range(self.pp_size):
            self.pprank_layer_idxs[pp_rank] = [num_layer_list_.pop(0) for _ in range(layers_each_pp[pp_rank])]

    def get_vpprank_hf_layeridxs(self) -> None:
        """vpp_rank -> hf layer map"""
        num_noop_layers = 0 if self.noop_layers is None else len(list(map(int, self.noop_layers.split(","))))
        num_real_layers = self.num_layers - num_noop_layers
        num_layer_list_ = [i for i in range(num_real_layers)]

        if self.vpp_stage is not None:
            layers_each_vpp = [[self.vpp_stage] * self.vpp_size for _ in range(self.pp_size)]
            # examples: num_layers8,pp2,vpp_stage2  [[0 1, 4 5], [2 3, 6 7]]
            # no noop layer --> layers_each_vpp:[[2,2], [2,2]]
            # noop4,5 --> layers_each_vpp:[[2,0], [2,2]]
            if self.noop_layers is not None:
                for layer in list(map(int, self.noop_layers.split(","))):
                    vpp_idx = layer // self.vpp_stage // self.pp_size
                    pp_idx = layer % (self.pp_size * self.vpp_stage) // self.vpp_stage
                    layers_each_vpp[pp_idx][vpp_idx] -= 1

            for pp_rank in range(self.pp_size):
                for vpp_rank in range(self.vpp_size):
                    self.vpprank_layer_idxs[pp_rank][vpp_rank] = [
                        num_layer_list_.pop(0) for _ in range(layers_each_vpp[pp_rank][vpp_rank])
                    ]

    def load_matched_hf_weights(self, pp_rank, vpp_rank=None):
        """Read the safetensors file corresponding to the layer of pp_rank."""
        if vpp_rank is None:
            layer_list = self.pprank_layer_idxs[pp_rank]
        else:
            layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank].copy()
        layer_files_map_dict = self.get_layer_files_map()

        st_filename_list = []
        for layer in layer_list:
            st_filename_list.extend(list(layer_files_map_dict[layer]))

        if pp_rank == 0:
            st_filename_list.extend(list(layer_files_map_dict["model.embed_tokens.weight"]))

        if pp_rank == self.pp_size - 1:
            st_filename_list.extend(list(layer_files_map_dict["model.norm.weight"]))
            st_filename_list.extend(list(layer_files_map_dict["lm_head.weight"]))

        st_filename_list = list(set(st_filename_list))
        st_filename_list.sort()

        all_pp_weights = {}
        for filename in st_filename_list:
            cur_weights = self.load_hf_model(os.path.join(self.hf_model_path, filename))
            all_pp_weights.update(cur_weights)

        return all_pp_weights

    def set_model_preprocess(self, weights_dict, mg_model):
        """Embedding layer process"""
        emb_weight = weights_dict.pop("model.embed_tokens.weight")

        for ep_rank in range(self.ep_size):
            emb_weight_lst = torch.chunk(emb_weight, self.tp_size, dim=0)
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank]["embedding.word_embeddings.weight"] = emb_weight_lst[tp_rank].clone()

    def set_model_postprocess(self, weights_dict, mg_model):
        """Final norm & LM Head process"""
        final_norm = weights_dict.pop("model.norm.weight")
        lm_head = weights_dict.pop("lm_head.weight")

        for ep_rank in range(self.ep_size):
            lm_head_lst = torch.chunk(lm_head, self.tp_size, dim=0)
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank]["decoder.final_layernorm.weight"] = final_norm.clone()
                mg_model[ep_rank][tp_rank]["output_layer.weight"] = lm_head_lst[tp_rank].clone()

    def set_model_layer_norm(self, hf_layer_idx, local_layer_idx, weights_dict, mg_model):
        """Layernorm process"""
        input_norm = weights_dict.pop(f"model.layers.{hf_layer_idx}.input_layernorm.weight")
        post_attn_norm = weights_dict.pop(f"model.layers.{hf_layer_idx}.post_attention_layernorm.weight")

        input_norm_key = f"decoder.layers.{local_layer_idx}.input_layernorm.weight"
        post_norm_key = f"decoder.layers.{local_layer_idx}.pre_mlp_layernorm.weight"

        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][input_norm_key] = input_norm.clone()
                mg_model[ep_rank][tp_rank][post_norm_key] = post_attn_norm.clone()

    def set_model_layer_attn(self, hf_layer, local_layer_idx, weights_dict, mg_model):
        """Attention layer process"""

        def _generate_attn_layers_key(local_idx):
            prefix = f"decoder.layers.{local_idx}"
            qkv_key = f"{prefix}.self_attention.linear_qkv.weight"
            dense_key = f"{prefix}.self_attention.linear_proj.weight"
            q_norm_key = f"{prefix}.self_attention.q_layernorm.weight"
            k_norm_key = f"{prefix}.self_attention.k_layernorm.weight"

            return qkv_key, dense_key, q_norm_key, k_norm_key

        hf_q_proj = weights_dict.pop(f"model.layers.{hf_layer}.self_attn.q_proj.weight")
        hf_k_proj = weights_dict.pop(f"model.layers.{hf_layer}.self_attn.k_proj.weight")
        hf_v_proj = weights_dict.pop(f"model.layers.{hf_layer}.self_attn.v_proj.weight")
        hf_o_proj = weights_dict.pop(f"model.layers.{hf_layer}.self_attn.o_proj.weight")

        q_layernorm = weights_dict.pop(f"model.layers.{hf_layer}.self_attn.q_norm.weight")
        k_layernorm = weights_dict.pop(f"model.layers.{hf_layer}.self_attn.k_norm.weight")

        nh = self.num_attention_heads
        ng = self.num_key_value_heads
        dim = self.head_dim

        qkv_weight = torch.cat(
            [
                hf_q_proj.reshape((ng, dim * nh // ng, -1)),
                hf_k_proj.reshape((ng, dim, -1)),
                hf_v_proj.reshape((ng, dim, -1)),
            ],
            dim=1,
        ).reshape((-1, self.hidden_size))
        qkv_weight_lst = torch.chunk(qkv_weight, self.tp_size, dim=0)

        qkv_key, dense_key, q_norm_key, k_norm_key = _generate_attn_layers_key(local_layer_idx)

        for ep_rank in range(self.ep_size):
            dense_lst = torch.chunk(hf_o_proj, self.tp_size, dim=1)

            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][qkv_key] = qkv_weight_lst[tp_rank].clone()
                mg_model[ep_rank][tp_rank][dense_key] = dense_lst[tp_rank].clone()
                mg_model[ep_rank][tp_rank][q_norm_key] = q_layernorm.clone()
                mg_model[ep_rank][tp_rank][k_norm_key] = k_layernorm.clone()

    def set_model_layer_mlp(self, hf_layer_idx, local_layer_idx, weights_dict, mg_model):
        """MLP layer process"""

        def _generate_moe_layer_key(local_idx):
            prefix = f"decoder.layers.{local_layer_idx}"
            router_key = f"{prefix}.mlp.router.weight"
            experts_weight1_key = f"{prefix}.mlp.experts.weight1"
            experts_weight2_key = f"{prefix}.mlp.experts.weight2"
            return router_key, experts_weight1_key, experts_weight2_key

        # moe layer
        mlp_router_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.gate.weight")
        mlp_router_weight = mlp_router_weight[: self.num_experts, :]

        experts_linear_fc1_list = []
        experts_linear_fc2_list = []

        for expert_idx in range(self.num_experts):
            gate_proj = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight")
            up_proj = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.experts.{expert_idx}.up_proj.weight")

            expert_tp_size = self.tp_size
            if self.moe_tp_extend_ep:
                expert_tp_size = 1

            gate_w_list = torch.chunk(gate_proj, expert_tp_size, dim=0)
            up_w_list = torch.chunk(up_proj, expert_tp_size, dim=0)
            fc1_weight = torch.cat([torch.cat(weights, dim=0) for weights in zip(gate_w_list, up_w_list)], dim=0)

            fc2_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.experts.{expert_idx}.down_proj.weight")

            experts_linear_fc1_list.append(fc1_weight.t())
            experts_linear_fc2_list.append(fc2_weight.t())

        # generate weights key
        router_key, experts_weight1_key, experts_weight2_key = _generate_moe_layer_key(local_layer_idx)

        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][router_key] = mlp_router_weight.clone()

        if self.moe_grouped_gemm:
            gemm_fc1 = torch.cat(experts_linear_fc1_list).view(self.hidden_size, -1)
            gemm_fc2 = torch.cat(experts_linear_fc2_list).view(-1, self.hidden_size)
            if self.moe_tp_extend_ep:
                gemm_fc1_ep = torch.chunk(
                    gemm_fc1.view(self.num_experts, self.hidden_size, -1), self.ep_size * self.tp_size, dim=0
                )
                gemm_fc2_ep = torch.chunk(
                    gemm_fc2.view(self.num_experts, -1, self.hidden_size), self.ep_size * self.tp_size, dim=0
                )
            else:
                gemm_fc1_ep = torch.chunk(gemm_fc1.view(self.num_experts, self.hidden_size, -1), self.ep_size, dim=0)
                gemm_fc2_ep = torch.chunk(gemm_fc2.view(self.num_experts, -1, self.hidden_size), self.ep_size, dim=0)

            for ep_rank in range(self.ep_size):
                if not self.moe_tp_extend_ep:
                    gemm_fc1_ep_tp = torch.chunk(gemm_fc1_ep[ep_rank], self.tp_size, dim=2)
                    gemm_fc2_ep_tp = torch.chunk(gemm_fc2_ep[ep_rank], self.tp_size, dim=1)
                for tp_rank in range(self.tp_size):
                    if self.moe_tp_extend_ep:
                        mg_model[ep_rank][tp_rank][experts_weight1_key] = (
                            gemm_fc1_ep[ep_rank * self.tp_size + tp_rank].reshape(self.hidden_size, -1).clone()
                        )
                        mg_model[ep_rank][tp_rank][experts_weight2_key] = (
                            gemm_fc2_ep[ep_rank * self.tp_size + tp_rank].reshape(-1, self.hidden_size).clone()
                        )
                    else:
                        mg_model[ep_rank][tp_rank][experts_weight1_key] = (
                            gemm_fc1_ep_tp[tp_rank].reshape(self.hidden_size, -1).clone()
                        )
                        mg_model[ep_rank][tp_rank][experts_weight2_key] = (
                            gemm_fc2_ep_tp[tp_rank].reshape(-1, self.hidden_size).clone()
                        )
        else:
            num_local_experts = self.num_experts // self.ep_size
            for ep_rank in range(self.ep_size):
                for local_experts_idx in range(num_local_experts):
                    local_prefix = f"decoder.layers.{local_layer_idx}.mlp.experts.local_experts"
                    local_fc1_key = f"{local_prefix}.{local_experts_idx}.linear_fc1.weight"
                    local_fc2_key = f"{local_prefix}.{local_experts_idx}.linear_fc2.weight"

                    global_experts_idx = local_experts_idx + ep_rank * num_local_experts
                    local_fc1_weight = experts_linear_fc1_list[global_experts_idx].t()
                    local_fc2_weight = experts_linear_fc2_list[global_experts_idx].t()

                    local_fc1_lst = torch.chunk(local_fc1_weight, self.tp_size, dim=0)
                    local_fc2_lst = torch.chunk(local_fc2_weight, self.tp_size, dim=1)

                    for tp_rank in range(self.tp_size):
                        mg_model[ep_rank][tp_rank][local_fc1_key] = local_fc1_lst[tp_rank].clone()
                        mg_model[ep_rank][tp_rank][local_fc2_key] = local_fc2_lst[tp_rank].clone()

    def generate_pp_local_layer_idx(self):
        """generate each pp local layer index"""
        pp_local_layer_idx = defaultdict()

        for pp_rank in range(self.pp_size):
            if self.num_layer_list is not None:
                layer_list = list(map(int, self.num_layer_list.split(',')))
                pp_local_layer_idx[pp_rank] = [i for i in range(layer_list[pp_rank])]
            else:
                pp_local_layer_idx[pp_rank] = [i for i in range(self.num_layers // self.pp_size)]

        if self.noop_layers is not None:
            noop_list = list(map(int, self.noop_layers.split(",")))
            num_layers_each_pp = self.num_layers // self.pp_size
            for num_noop_layers in noop_list:
                pp_idx = num_noop_layers // num_layers_each_pp
                local_noop_idx = num_noop_layers % num_layers_each_pp
                pp_local_layer_idx[pp_idx].remove(local_noop_idx)

        return pp_local_layer_idx

    def generate_vpp_local_layer_idx(self):
        vpp_local_layer_idx = defaultdict()
        for pp_rank in range(self.pp_size):
            vpp_local_layer_idx[pp_rank] = defaultdict()

        for pp_rank in range(self.pp_size):
            for vpp_rank in range(self.vpp_size):
                vpp_local_layer_idx[pp_rank][vpp_rank] = [i for i in range(self.vpp_stage)]

        if self.noop_layers is not None:
            noop_list = list(map(int, self.noop_layers.split(",")))
            num_layers_each_pp = self.num_layers // self.pp_size
            for num_noop_layer in noop_list:
                pp_idx = num_noop_layer % (self.pp_size * self.vpp_stage) // self.vpp_stage
                vpp_idx = num_noop_layer // self.vpp_stage // self.pp_size
                local_noop_idx = num_noop_layer % num_layers_each_pp % self.vpp_stage
                vpp_local_layer_idx[pp_idx][vpp_idx].remove(local_noop_idx)

        return vpp_local_layer_idx

    def run(self):
        """save magetron format checkpoint"""
        pp_local_layer_idx = self.generate_pp_local_layer_idx()
        save_model_path = self.mg_path_process(self.mg_save_path)

        if self.vpp_stage is None:
            for pp_rank in range(self.pp_size):
                mg_model = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

                pp_weights = self.load_matched_hf_weights(pp_rank)
                if pp_rank == 0:
                    self.set_model_preprocess(pp_weights, mg_model)

                layer_list = self.pprank_layer_idxs[pp_rank]

                local_idx = 0
                cur_pp_local_idx = pp_local_layer_idx[pp_rank]

                for hf_layer in layer_list:
                    logger.info(f"Converting the weights of layer {hf_layer}.")
                    local_layer_idx = cur_pp_local_idx[local_idx]
                    self.set_model_layer_norm(hf_layer, local_layer_idx, pp_weights, mg_model)
                    self.set_model_layer_attn(hf_layer, local_layer_idx, pp_weights, mg_model)
                    self.set_model_layer_mlp(hf_layer, local_layer_idx, pp_weights, mg_model)
                    local_idx += 1

                if pp_rank == self.pp_size - 1:
                    self.set_model_postprocess(pp_weights, mg_model)

                for ep_rank in range(self.ep_size):
                    for tp_rank in range(self.tp_size):
                        save_prefix = self.generate_mg_weights_dir(tp_rank=tp_rank, pp_rank=pp_rank, ep_rank=ep_rank)
                        parallel_save_path = os.path.join(save_model_path, save_prefix)
                        os.makedirs(parallel_save_path)
                        save_file_name = os.path.join(parallel_save_path, "model_optim_rng.pt")
                        logger.info(f"Saving to {save_file_name}")

                        torch.save(
                            {"model": mg_model[ep_rank][tp_rank], "checkpoint_version": 3.0, "iteration": 1},
                            save_file_name,
                            pickle_protocol=4,
                            _use_new_zipfile_serialization=True,
                        )
        else:
            vpp_local_layer_idx = self.generate_vpp_local_layer_idx()
            for pp_rank in range(self.pp_size):
                mg_model = defaultdict()
                for vpp_rank in range(self.vpp_size):
                    pp_weights = self.load_matched_hf_weights(pp_rank, vpp_rank)
                    mg_model[vpp_rank] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
                    vpp_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]

                    if pp_rank == 0 and vpp_rank == 0:
                        self.set_model_preprocess(pp_weights, mg_model[vpp_rank])

                    local_idx = 0
                    cur_vpp_local_idx = vpp_local_layer_idx[pp_rank][vpp_rank]

                    for hf_layer in vpp_list:
                        logger.info(f"Converting the weights of layer {hf_layer}.")
                        local_layer_idx = cur_vpp_local_idx[local_idx]
                        self.set_model_layer_norm(hf_layer, local_layer_idx, pp_weights, mg_model[vpp_rank])
                        self.set_model_layer_attn(hf_layer, local_layer_idx, pp_weights, mg_model[vpp_rank])
                        self.set_model_layer_mlp(hf_layer, local_layer_idx, pp_weights, mg_model[vpp_rank])
                        local_idx += 1

                    if pp_rank == self.pp_size - 1 and vpp_rank == self.vpp_size - 1:
                        self.set_model_postprocess(pp_weights, mg_model[vpp_rank])

                for ep_rank in range(self.ep_size):
                    for tp_rank in range(self.tp_size):
                        save_prefix = self.generate_mg_weights_dir(tp_rank=tp_rank, pp_rank=pp_rank, ep_rank=ep_rank)
                        parallel_save_path = os.path.join(save_model_path, save_prefix)
                        os.makedirs(parallel_save_path, exist_ok=True)
                        save_file_name = os.path.join(parallel_save_path, "model_optim_rng.pt")
                        logger.info(f"Saving to {save_file_name}")
                        model_dict = {"checkpoint_version": 3.0, "iteration": 1}

                        for vpp_rank in range(self.vpp_size):
                            model_key = f"model{vpp_rank}"
                            model_dict[model_key] = mg_model[vpp_rank][ep_rank][tp_rank]

                        torch.save(model_dict, save_file_name, pickle_protocol=4, _use_new_zipfile_serialization=True)

        logger.info("Done!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', type=str, required=True, help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True, help='Directory to save model checkpoint to')
    parser.add_argument(
        '--target-tensor-parallel-size', type=int, default=1, help='Target tensor model parallel size, defaults to 1.'
    )
    parser.add_argument(
        '--target-pipeline-parallel-size',
        type=int,
        default=1,
        help='Target pipeline model parallel size, defaults to 1.',
    )
    parser.add_argument(
        '--target-expert-parallel-size', type=int, default=1, help='Target expert model parallel size, defaults to 1.'
    )
    parser.add_argument(
        '--num-layers-per-virtual-pipeline-stage',
        type=int,
        default=None,
        help='Number of layers per virtual pipeline stage',
    )
    parser.add_argument('--moe-grouped-gemm', action='store_true', help='Usr moe grouped gemm.')
    parser.add_argument("--noop-layers", type=str, default=None, help='Specity the noop layers.')
    parser.add_argument(
        '--num-layer-list', type=str, help='a list of number of layers, seperated by comma; e.g., 4,4,4,4'
    )
    parser.add_argument('--num-layers', type=int, default=61, help='Number of transformer layers.')
    parser.add_argument(
        "--moe-tp-extend-ep",
        action='store_true',
        help="Use tp group to extend experts parallism instead of sharding weight tensor of experts in tp group",
    )
    parser.add_argument("--hidden-size", type=int, default=4096, help="Hidden size")
    parser.add_argument("--num-attention-heads", type=int, default=64, help="Num attention heads")

    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    logger.info(f"Arguments: {args}")
    global HIDDEN_SIZE
    global NUM_ATTENTION_HEADS
    HIDDEN_SIZE = args.hidden_size
    NUM_ATTENTION_HEADS = args.num_attention_heads
    converter = CkptConvert(
        hf_model_path=args.load_dir,
        mg_save_path=args.save_dir,
        num_layers=args.num_layers,
        tp_size=args.target_tensor_parallel_size,
        pp_size=args.target_pipeline_parallel_size,
        ep_size=args.target_expert_parallel_size,
        num_layer_list=args.num_layer_list,
        noop_layers=args.noop_layers,
        moe_grouped_gemm=args.moe_grouped_gemm,
        moe_tp_extend_ep=args.moe_tp_extend_ep,
        vpp_stage=args.num_layers_per_virtual_pipeline_stage,
    )
    converter.run()


if __name__ == '__main__':
    main()
