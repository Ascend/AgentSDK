#!/usr/bin/env python3
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
import datasets

import third_party.rl
from mindspeed_rl.workers.scheduler.launcher import RayActorGroup
from .launcher import create_actor_handlers_patch

from mindspeed_rl.workers.resharding.megatron_sharding_manager import MegatronShardingManager
from .megatron_sharding_manager import enter_infer_mode_patch, exit_infer_mode_patch, exit_train_mode_patch

from mindspeed_rl.models.loss.grpo_actor_loss_func import GRPOActorLossFunc
from .grpo_actor_loss_func import _get_policy_loss_input

from mindspeed_rl.trainer.utils import compute_utils
from .compute_utils_patch import compute_group_norm_advantage_return_patch


RayActorGroup.create_actor_handlers = create_actor_handlers_patch
MegatronShardingManager.enter_infer_mode = enter_infer_mode_patch
MegatronShardingManager.exit_infer_mode = exit_infer_mode_patch
MegatronShardingManager.exit_train_mode = exit_train_mode_patch
GRPOActorLossFunc._get_policy_loss_input = _get_policy_loss_input
compute_utils.compute_group_norm_advantage_return = compute_group_norm_advantage_return_patch

from mindspeed_rl.models.base.base_training_engine import BaseTrainingEngine
from .base_training_engine import _split_batches_with_dynamic_bsz
BaseTrainingEngine._split_batches_with_dynamic_bsz = _split_batches_with_dynamic_bsz

from mindspeed_rl.workers.resharding.memory_buffer import MemoryBuffer, ModelWeightBuffer
from mindspeed_rl.workers.resharding import memory_buffer
from .memory_buffer_patch import __init__, copy_by_name, rebuild_with_device, build_experts_memory_buffer_patch
MemoryBuffer.__init__ = __init__
MemoryBuffer.copy_by_name = copy_by_name
memory_buffer.build_experts_memory_buffer = build_experts_memory_buffer_patch
ModelWeightBuffer.rebuild_with_device = rebuild_with_device

from mindspeed_rl.workers.resharding.vllm_weight_container import MegatronStyleVllmWeightContainer
from .vllm_weight_container_patch import __init__, _validate_parallel_config_patch, split_tp_params_patch, \
    _update_weight_buffers_ep_patch, _get_simple_ep_params, _collect_name_pairs_for_pp, get_infer_params_patch
MegatronStyleVllmWeightContainer.__init__ = __init__
MegatronStyleVllmWeightContainer._validate_parallel_config = _validate_parallel_config_patch
MegatronStyleVllmWeightContainer._collect_name_pairs_for_pp = _collect_name_pairs_for_pp
MegatronStyleVllmWeightContainer._get_simple_ep_params = _get_simple_ep_params
MegatronStyleVllmWeightContainer.split_tp_params = split_tp_params_patch
MegatronStyleVllmWeightContainer._update_weight_buffers_ep = _update_weight_buffers_ep_patch
MegatronStyleVllmWeightContainer.get_infer_params = get_infer_params_patch