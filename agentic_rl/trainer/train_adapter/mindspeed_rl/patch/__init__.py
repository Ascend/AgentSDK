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


def apply_patch():
    """
    Apply patch for adaptor to mindspeed_rl
    """

    from mindspeed_rl.models.loss.grpo_actor_loss_func import GRPOActorLossFunc
    from mindspeed_rl.models.loss.logprob_computer import StandardLogProbComputer
    from mindspeed_rl.trainer.utils import compute_utils as ms_compute_utils
    from mindspeed_rl.workers.scheduler.launcher import RayActorGroup
    from mindspeed_rl.workers import base_worker
    from mindspeed_rl.utils import utils

    from .compute_utils import compute_group_norm_advantage_return_patch
    from .logprob_computer import compute
    from .grpo_actor_loss_func import get_policy_loss_input_patch
    from .launcher import (create_actor_handlers_patch, update_ref_dispatch_size, update_actor_logprob_dispatch_size,
                           update_actor_update_dispatch_size, update_mini_batch_size as group_update_mini_batch_size)
    from .get_current_node_ip import get_current_node_ip_patch

    RayActorGroup.create_actor_handlers = create_actor_handlers_patch
    RayActorGroup.update_ref_dispatch_size = update_ref_dispatch_size
    RayActorGroup.update_actor_logprob_dispatch_size = update_actor_logprob_dispatch_size
    RayActorGroup.update_actor_update_dispatch_size = update_actor_update_dispatch_size
    RayActorGroup.update_mini_batch_size = group_update_mini_batch_size

    StandardLogProbComputer.computer = compute
    GRPOActorLossFunc._get_policy_loss_input = get_policy_loss_input_patch
    ms_compute_utils.compute_group_norm_advantage_return = compute_group_norm_advantage_return_patch
    base_worker.get_current_node_ip = get_current_node_ip_patch
    utils.get_current_node_ip = get_current_node_ip_patch

    from mindspeed_rl.trainer import grpo_trainer_hybrid
    from .grpo_transfer_dock import GRPOTransferDock as GRPOTransferDockPatch
    grpo_trainer_hybrid.GRPOTransferDock = GRPOTransferDockPatch

    from mindspeed_rl.trainer.grpo_trainer_hybrid import RayGRPOTrainer
    from .compute_advantage import compute_advantage
    RayGRPOTrainer.compute_advantage = compute_advantage

    from mindspeed_rl.models.actor_rollout_hybrid import ActorRolloutHybrid
    from .actor_rollout_hybrid import update_mini_batch_size as hybrid_update_mini_batch_size
    ActorRolloutHybrid.update_mini_batch_size = hybrid_update_mini_batch_size

    from mindspeed_rl.models.base.base_training_engine import BaseTrainingEngine
    from .base_training_engine import update_mini_batch_size as engine_update_mini_batch_size, update
    BaseTrainingEngine.update_mini_batch_size = engine_update_mini_batch_size
    BaseTrainingEngine.update = update
