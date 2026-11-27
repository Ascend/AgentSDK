#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.

import typing
from collections.abc import Callable, Iterable
from typing import Set

import torch

from vllm.logger import logger
from vllm.model_executor.models.qwen3_moe import Qwen3MoeModel
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)


def load_weights_patch(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Set[str]:
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    ignore_suffixes = (
        ".bias", "_bias",
        ".k_scale", "_k_scale",
        ".v_scale", "_v_scale",
        ".weight_scale", "_weight_scale",
        ".input_scale", "_input_scale"
    )

    params_dict = dict(self.named_parameters())
    loaded_params: Set[str] = set()
    expert_params_mapping = self.get_expert_mapping()

    for name, loaded_weight in weights:
        for (param_name, weight_name, shard_id) in stacked_params_mapping:
            if weight_name not in name:
                continue

            if "mlp.experts" in name:
                continue

            name = name.replace(weight_name, param_name)

            if name.endswith(ignore_suffixes) and name not in params_dict:
                continue

            if is_pp_missing_parameter(name, self):
                continue

            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if weight_loader == default_weight_loader:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)
            break
        else:
            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue

                is_expert_weight = True

                name_mapped = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name_mapped, self):
                    continue

                if name_mapped.endswith(ignore_suffixes) and name_mapped not in params_dict:
                    continue

                if name_mapped not in params_dict:
                    continue

                param = params_dict[name_mapped]
                weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                success = weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True)
                if success:
                    name = name_mapped
                    break
            else:
                if is_expert_weight:
                    continue

                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                if name.endswith("kv_scale"):
                    remapped_kv_scale_name = name.replace(".kv_scale", ".attn.kv_scale")
                    if remapped_kv_scale_name not in params_dict:
                        logger.warning_once(
                            "Found kv scale in the checkpoint (e.g. %s), "
                            "but not found the expected name in the model (e.g. %s). "
                            "kv-scale is not loaded.",
                            name,
                            remapped_kv_scale_name,
                        )
                        continue
                    else:
                        name = remapped_kv_scale_name

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        loaded_params.add(name)

    return loaded_params


Qwen3MoeModel.load_weights = load_weights_patch
