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
from time import time

import ray

from recipe.fully_async_policy.megatron_worker import DetachActorWorker
from verl.single_controller.base.decorator import Dispatch, register

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class MegatronDetachActorWorker(DetachActorWorker):
    """Megatron-based actor worker that supports detached weight saving and synchronization."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def prepare_infer_params_to_cpu(
        self, weight_save_dir: str, sync_group_name: str = "actor_rollout"
    ) -> None:
        """Save model weights to CPU and notify the weight updater.

        Args:
            weight_save_dir: Directory path where weights will be saved.
            sync_group_name: Name of the synchronization group.
        """
        logger.info(f"start saving weights, path={weight_save_dir}")
        os.makedirs(weight_save_dir, exist_ok=True)

        start_time = time()
        self.bridge.save_weights(self.actor.actor_module, weight_save_dir)
        logger.info(f"weight saving completed, path={weight_save_dir}, time={time() - start_time}")

        w_actor = ray.get_actor("weight_updater", namespace="controller_raygroup")
        w_actor.weight_saved.remote(weight_save_dir)
        logger.info("send a weight saving completion notification")
