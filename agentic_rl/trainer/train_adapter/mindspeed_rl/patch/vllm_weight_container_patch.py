# -*- coding: utf-8 -*-
#
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
# 
import os
import torch
import torch.distributed as dist
from functools import partial

from mindspeed_rl.workers.resharding.utils import get_tensor_parallel_partition_dim, _build_infer_param_dict, \
    is_tensor_parallel_param, get_tp_group, is_fake_tp_param
from mindspeed_rl.workers.resharding.memory_buffer import (
    build_experts_memory_buffer, 
    get_weight_buffer_meta_from_buffer
)
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.runner.infer_adapter.vllm.extension.custom_worker_extensions import broadcast_if_gpu

logger = Loggers(__name__).get_logger()


def __init__(
    self, megatron_model, vllm_model, model_config,
    infer_tensor_parallel_size,
    infer_pipeline_parallel_size,
    infer_expert_parallel_size,
    num_layer_list,
    moe_tp_extend_ep=False,
    parallel_state=None,
    weight_adaptor=None,
    enable_validate=False,
    noop_layers=None,
) -> None:
    """Initialize the patched MegatronStyleVllmWeightContainer."""
    ep_mode_env = os.getenv("ONE_STEP_OFF_EP_MODE", "false")
    self.one_step_off_ep_mode = ep_mode_env == "true"
    logger.info(f"one step off ep mode: {self.one_step_off_ep_mode}")
    self.vllm_model = vllm_model
    self.model_config = model_config
    self.megatron_model = megatron_model
    self.parallel_state = parallel_state
    self.weight_adaptor = weight_adaptor
    # num_hidden_layers from the HF config.json reachable via the tokenizer path
    self._num_hidden_layers = self.model_config.num_hidden_layers
    self._noop_layers = None
    if noop_layers is not None:
        self._noop_layers = [int(layer_idx) for layer_idx in noop_layers.split(',')]
        self._num_hidden_layers += len(self._noop_layers)

    # pp configs
    self._pp_rank = self.parallel_state.get_pipeline_model_parallel_rank()
    self._pp_group = self.parallel_state.get_pipeline_model_parallel_group()
    self._pp_size = self.parallel_state.get_pipeline_model_parallel_world_size()
    self._world_size = dist.get_world_size()

    # vpp configs
    self._num_layer_list = self._build_num_layer_list(num_layer_list)
    vpp_rank_val = self.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    self._vpp_rank = vpp_rank_val if vpp_rank_val else 0
    vpp_size_val = self.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    self._vpp_size = vpp_size_val if vpp_size_val else 1
    self._vpp_layer_list = self._build_vpp_layer_list(self._num_layer_list)
    # noop_layers global-to-local mapping
    self._global2local_map = (
        self._build_global2local_map(self._vpp_layer_list, self._vpp_size, self._noop_layers)
        if self._noop_layers is not None
        else None
    )

    # tp configs
    self._tp_size = self.parallel_state.get_tensor_model_parallel_world_size()
    self._tp_group = self.parallel_state.get_tensor_model_parallel_group()

    # ep configs
    self._ep_size = self.parallel_state.get_expert_model_parallel_world_size()

    if moe_tp_extend_ep:
        self._ep_group = self.parallel_state.get_tensor_and_expert_parallel_group()
        self._ep_size = self._tp_size * self._ep_size
    else:
        self._ep_group = self.parallel_state.get_expert_model_parallel_group()

    self.num_experts = 0
    self.num_local_experts = 0
    if hasattr(self.model_config, "n_routed_experts"):
        self.num_experts = self.model_config.n_routed_experts
        self.num_local_experts = self.num_experts // self._ep_size
    elif hasattr(self.model_config, "num_experts"):
        self.num_experts = self.model_config.num_experts
        self.num_local_experts = self.num_experts // self._ep_size

    # infer configs
    self._infer_tp_size = infer_tensor_parallel_size
    self._infer_pp_size = infer_pipeline_parallel_size
    self._infer_ep_size = infer_expert_parallel_size
    self.moe_tp_extend_ep = moe_tp_extend_ep

    # infer_expert_tensor_parallel_size and num_process is fixed.
    self.infer_expert_tensor_parallel_size = 1
    self.num_process = 1
    self._infer_ep_size = self._infer_ep_size * self._infer_tp_size
    self.experts_memory_expand_N = self._infer_ep_size // self._ep_size

    # validate parallel configs
    self._validate_parallel_config()

    # md5 validate
    self.enable_validate = enable_validate
    self.origin_params_for_md5 = None
    self.infer_params_for_md5 = None

    self._rank = dist.get_rank()
    if not self.one_step_off_ep_mode:
        self._init_tensor_model_parallel_allgather_group()
        self._init_pipeline_model_parallel_allgather_group()
        self._init_tensor_model_parallel_split_group()
    self._init_weight_buffers()


def _validate_parallel_config_patch(self) -> None:
    """Validate that train/infer parallel configs are compatible."""
    if self.one_step_off_ep_mode:
        return
    if self._infer_pp_size != 1:
        raise ValueError("infer_pp_size != 1 not supported yet")

    if self._infer_ep_size % self._ep_size != 0:
        raise ValueError("The training expert size should be divisible by the inference expert size.")
    if self._ep_size > 1 and not self.moe_tp_extend_ep:
        raise ValueError("To enable training EP, you need to enable moe_tp_extend_ep and use GroupedMLP.")
    if self._pp_size < self._infer_pp_size:
        raise ValueError(
            "The training pipeline parallel size should be greater than or equal to the inference pipeline "
            "parallel size.")
    if self._pp_size % self._infer_pp_size != 0:
        raise ValueError(
            "The training pipeline parallel size should be an integer multiple of the inference pipeline parallel "
            "size.")
    if self._tp_size > self._infer_tp_size and self._tp_size % self._infer_tp_size != 0:
        raise ValueError(
            "The training tensor parallel size should be an integer multiple of the inference tensor parallel size.")
    # For tp increase, train_tp * dp >= infer_tp, train_tp * dp % infer_tp == 0
    if self._tp_size < self._infer_tp_size:
        if (self._world_size // self._pp_size < self._infer_tp_size or
                (self._world_size // self._pp_size) % self._infer_tp_size != 0):
            raise ValueError(
                f"Do not support split train tp size {self._tp_size} to infer tp size {self._infer_tp_size} "
                f"with train dp size {(self._world_size // (self._tp_size * self._pp_size))}.")

def split_tp_params_patch(self, param: torch.Tensor, name: str) -> torch.Tensor:
    """Split tensor-parallel parameters for inference.

    1. Get full train params through allgather.
    2. Split train_tp params into groups (size: infer_tp_size).
    3. Return the corresponding param from group based on infer tp rank.
    """
    if self._infer_tp_size <= self._tp_size or is_fake_tp_param(name, self.moe_tp_extend_ep):
        return param

    tp_group = get_tp_group()

    if is_tensor_parallel_param(param):
        if self._tp_size > 1:
            # allocate a new tensor with proper size
            infer_params = [torch.empty_like(param) for _ in range(self._tp_size)]
            torch.distributed.all_gather(infer_params, param, group=tp_group)
        else:
            infer_params = [param]
        if "linear_fc1.weight" in name:
            # if the tensor is gate and proj
            gate_lst = []
            up_lst = []
            for infer_param in infer_params:
                gate, up = infer_param.chunk(2)
                gate_lst.append(gate)
                up_lst.append(up)
            gate = torch.cat(gate_lst, dim=0)
            up = torch.cat(up_lst, dim=0)

            gate_splits = torch.chunk(gate, self._infer_tp_size, dim=0)
            up_splits = torch.chunk(up, self._infer_tp_size, dim=0)

            new_params_list = [
                torch.cat([gate_splits[i], up_splits[i]], dim=0)
                for i in range(self._infer_tp_size)
            ]
        elif "qkv" in name and self._infer_tp_size > self.model_config.num_key_value_heads:
            g = self.model_config.num_key_value_heads
            partition_dim = get_tensor_parallel_partition_dim(param)
            infer_params = torch.cat(infer_params, dim=partition_dim)
            split_params = torch.chunk(infer_params, g, dim=partition_dim)
            n = self.model_config.num_attention_heads
            repeats = n // g
            rep = self._infer_tp_size // g
            step = repeats // rep
            h = infer_params.size(-1)

            global_rank = self._rank
            infer_tp_rank_in_group = global_rank % self._infer_tp_size
            i = infer_tp_rank_in_group // rep
            j = infer_tp_rank_in_group % rep
            qkv = split_params[i].reshape(repeats + 2, -1, h)
            return torch.cat([qkv[step * j: step * j + step], qkv[repeats:]],
                             dim=0).reshape(-1, h)
        else:
            partition_dim = get_tensor_parallel_partition_dim(param)
            infer_params = torch.cat(infer_params, dim=partition_dim)
            split_params = torch.chunk(infer_params, self._infer_tp_size, dim=partition_dim)
            new_params_list = list(split_params)

        # make_list
        param_list = new_params_list

    else:
        param_list = [param] * self._infer_tp_size

    global_rank = self._rank
    infer_tp_rank_in_group = global_rank % self._infer_tp_size
    return param_list[infer_tp_rank_in_group]

def _update_weight_buffers_ep_patch(self) -> None:
    """Build temporary expert memory buffers and broadcast expert weights across PP ranks."""
    for cur_pp_rank in range(self._pp_size):
        pp_rank = self._pp_rank

        # Step 1: set up a temporary experts buffer for the current PP rank
        combined_names_per_pp = []
        vpp_stages = self.weight_names_per_pp[cur_pp_rank]
        for weight_names_per_stage in vpp_stages:
            combined_names_per_pp.extend(weight_names_per_stage)
        self.weight_buffer_meta = self.weight_adaptor.get_weight_buffer_meta(
            self.vllm_model, combined_names_per_pp,
        )
        self.experts_weight_buffer_meta = get_weight_buffer_meta_from_buffer(
            self.weight_buffer_meta,
        )

        # Build the experts buffer on the same device as the final
        # pp-weight buffer to avoid cross-device copy_
        target_dev = next(
            iter(self.weight_buffers[cur_pp_rank].memory_buffers.values())
        ).data.device
        self.experts_memory_buffers = build_experts_memory_buffer(
            self.experts_weight_buffer_meta,
            self.experts_memory_expand_N,
            device=target_dev,
        )

        # Step 2: copy the corresponding weights into the experts buffer
        if cur_pp_rank == pp_rank:
            weight_names = self.weight_names_per_pp[pp_rank]
            weight_names_meta = self.weight_adaptor.convert_weight_name_meta(weight_names)
            normal_layer_func = partial(
                self.weight_adaptor.global2local_layer,
                num_layer_list=self._vpp_layer_list,
                global2local_map=self._global2local_map,
            )
            name_pairs = sorted(list(set([
                (
                    name,
                    vpp_rank,
                    self.weight_adaptor.replace_name_i2t(
                        normal_layer_func(name, vpp_rank=vpp_rank)
                    ),
                )
                for vpp_rank, names_per_vpp in enumerate(weight_names_meta)
                for name in names_per_vpp
            ])))
            true_megatron_model = self._unwrap_megatron_model(self.megatron_model)

            # Collect all weights for the current PP rank
            megatron_params_dict = {}
            for vpp_rank in range(self._vpp_size):
                megatron_params_dict[vpp_rank] = dict(
                    true_megatron_model[vpp_rank].named_buffers()
                )
                megatron_params_dict[vpp_rank].update(
                    true_megatron_model[vpp_rank].named_parameters()
                )
                megatron_params_dict[vpp_rank] = (
                    self.weight_adaptor.adjust_megatron_param_dict(
                        megatron_params_dict[vpp_rank], self._tp_size,
                    )
                )

            for hf_name, vpp_rank, megatron_name in name_pairs:
                is_expert_ep = (
                    (self._infer_ep_size > 1 or self._ep_size > 1)
                    and "mlp.experts" in megatron_name
                )
                if is_expert_ep:
                    megatron_param = megatron_params_dict[vpp_rank][megatron_name]
                    dtype = self.experts_weight_buffer_meta[hf_name]['dtype']
                    self.experts_memory_buffers[dtype].copy_by_name(hf_name, megatron_param)

        # Step 3: broadcast expert weights from the experts memory buffer
        global_src = dist.get_global_rank(group=self._pp_group, group_rank=cur_pp_rank)

        for dtype, experts_memory_buffer in self.experts_memory_buffers.items():
            broadcast_if_gpu(
                experts_memory_buffer.data,
                src=global_src, group=self._pp_group,
            )
            ep_expand_rank = self._rank // self._ep_size

            for name, tensor_indices_value in sorted(experts_memory_buffer.tensor_indices.items()):
                shape = tensor_indices_value[1]  # expanded (N * ...) shape
                index = ep_expand_rank % self.experts_memory_expand_N
                experts_tensor = experts_memory_buffer.get_by_name(name)

                target_device = (
                    self.weight_buffers[cur_pp_rank]
                    .memory_buffers[dtype].data.device
                )
                if experts_tensor.device != target_device:
                    experts_tensor = experts_tensor.to(
                        target_device, non_blocking=False,
                    )
                experts_tensor_reshape = experts_tensor.view(shape)
                weight_tensor_infer = experts_tensor_reshape[index]
                self.weight_buffers[cur_pp_rank].copy_by_name(name, weight_tensor_infer)

            # Release the experts buffer for this dtype
            self.experts_memory_buffers[dtype] = None

        self.experts_memory_buffers = None

def _collect_name_pairs_for_pp(self, pp_rank: int) -> list:
    """Collect sorted (hf_name, vpp_rank, megatron_name) triples for a PP rank."""
    weight_names = self.weight_names_per_pp[pp_rank]
    weight_names_meta = self.weight_adaptor.convert_weight_name_meta(weight_names)
    normal_layer_func = partial(
        self.weight_adaptor.global2local_layer,
        num_layer_list=self._vpp_layer_list,
        global2local_map=self._global2local_map
    )
    name_pairs = sorted(list(set([
        (name, vpp_rank,
        self.weight_adaptor.replace_name_i2t(normal_layer_func(name, vpp_rank=vpp_rank)))
        for vpp_rank, names_per_vpp in enumerate(weight_names_meta)
        for name in names_per_vpp
    ])))
    return name_pairs

def _get_simple_ep_params(self) -> dict:
    """
    Produce simple, local shards:
    - Non-experts: full tensors (train-TP gathered) for this PP slice only.
    - Experts (w13, w2, gate/router): local experts only, in training layout (no reordering).
    Also attach a __simple_ep_meta__ block so the generation unit can stitch.
    """
    true_megatron_model = self._unwrap_megatron_model(self.megatron_model)

    # Build per-vpp param dict in training layout (same as intra-pp code)
    megatron_params_dict = {}
    for vpp_rank in range(self._vpp_size):
        pd = dict(true_megatron_model[vpp_rank].named_buffers())
        pd.update(true_megatron_model[vpp_rank].named_parameters())
        pd = self.weight_adaptor.adjust_megatron_param_dict(pd, self._tp_size)
        megatron_params_dict[vpp_rank] = pd

    # Compute EP shard info (already "combined" if moe_tp_extend_ep was enabled)
    if self._ep_size > 1:
        try:
            combined_ep_rank = dist.get_rank(group=self._ep_group)
        except (RuntimeError, ValueError):
            logger.warning("Failed to query rank for the expert parallel group")
            combined_ep_rank = 0
        num_local_experts = getattr(self, "num_local_experts", None)
        if num_local_experts is None:
            # fallback if not set by config
            total_experts = getattr(self, "num_experts", 0)
            num_local_experts = total_experts // max(self._ep_size, 1)
        expert_offset = combined_ep_rank * num_local_experts
    else:
        combined_ep_rank = 0
        num_local_experts = self.num_experts
        expert_offset = 0

    # Collect weights for *this* pp rank only (no inter-pp broadcast; other ranks handle their pp)
    pp_rank = self._pp_rank
    name_pairs = self._collect_name_pairs_for_pp(pp_rank)

    out = {}
    slices_meta = {}  # per-tensor slice descriptor for expert tensors

    # NOTE: We keep everything on current device; call .cpu() before sending if you prefer.
    for hf_name, vpp_rank, megatron_name in name_pairs:
        param = megatron_params_dict[vpp_rank][megatron_name]

        # Gather TP if train_TP>1 to materialize full param (still local PP slice).
        is_expert = ("mlp.experts" in megatron_name)  # w13 / w2
        is_w13 = ("w13_weight" in hf_name) or ("weight1" in megatron_name)
        is_w2  = ("w2_weight"  in hf_name) or ("weight2" in megatron_name)

        if is_expert and (is_w13 or is_w2):
            # Save EXACT training 2-D layout
            #   w13: (H, E_local * per)
            #   w2 : (E_local * per, H)
            out[hf_name] = param.detach().clone()   # <— no extra reshape here

            if self._ep_size > 1:
                axis = 1 if is_w13 else 0          # w13 experts along columns; w2 along rows
                E    = int(num_local_experts)
                slices_meta[hf_name] = {
                    "axis": int(axis),
                    "offset": int(expert_offset),   # rank * E_local
                    "length": int(E),               # E_local
                    "stride": 1,                    # contiguous experts
                    "layout": "contiguous",
                }
        else:
            # Non-expert tensors
            out[hf_name] = param.detach().clone()

    # Attach a tiny metadata block for stitching
    out["__simple_ep_meta__"] = {
        "format": "simple_ep_v1",
        "moe_tp_extend_ep": bool(self.moe_tp_extend_ep),
        "tp_size": int(self._tp_size),
        "pp_size": int(self._pp_size),
        "train_tp_rank": int(self.parallel_state.get_tensor_model_parallel_rank()),
        "ep_world_size": int(self._ep_size),
        "vpp_size": int(self._vpp_size),
        "pp_rank": int(self._pp_rank),
        "vpp_rank": int(self._vpp_rank),
        "combined_ep_rank": int(combined_ep_rank),
        "num_experts_total": int(getattr(self, "num_experts", 0)),
        "num_local_experts": int(num_local_experts),
        "expert_offset": int(expert_offset),
        "slices": slices_meta,  # per-tensor slice specs for experts
    }
    return out

def get_infer_params_patch(self) -> dict:
    """Collect and reshape all training weights into inference-ready parameters."""
    if self.one_step_off_ep_mode:
        return self._get_simple_ep_params()

    self._update_weight_buffers_intra_pp()
    self._update_weight_buffers_inter_pp()

    # Prerequisite for _update_weight_buffers_ep + _send_receive_experts
    if self.moe_tp_extend_ep and self._infer_ep_size >= self._ep_size:
        self._update_weight_buffers_ep()
        self._send_receive_experts()

    params = self._get_all_params()

    params = _build_infer_param_dict(params=params)
    return params