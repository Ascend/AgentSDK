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
import ray

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


@ray.remote
class ParameterSynchronizer:
    """
    Unified parameter synchronizer for syncing model parameters between actor and rollout.

    Based on the mature synchronization mode implementation of one_step_off_policy.
    Merges the functions of the original multiple synchronizer classes.
    """

    def __init__(self) -> None:
        pass

    def get_current_param_version(self) -> int:
        """Get current parameter version number."""
        logger.info("get_current_param_version()")

    def get_weights_info(self) -> None:
        """Get weights info."""
        logger.info("get_weights_info()")

    def sync_weights(self, version: int, validate: bool = False, global_steps: int = 0) -> None:
        """Sync weights between trainer and rollouter, and update parameter version."""
        logger.info("sync_weights()")

    def wait_last_valid(self) -> None:
        """Wait for the last validation task to complete."""
        logger.info("wait_last_valid()")

    def rollouter_save_checkpoint(self, local_global_step_folder: str) -> None:
        """Trigger rollout to save checkpoint(dataloader)."""
        logger.info("rollouter_save_checkpoint()")
