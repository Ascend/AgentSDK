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
from typing import Callable, Optional, Tuple

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class TrainRegistry:
    """Registry mapping (train_engine, cluster_mode) pairs to rollout/train methods."""

    def __init__(self):
        self._registry: dict[tuple[str, str], tuple[Optional[Callable], Callable]] = {}

    def register(
        self,
        train_engine: str,
        cluster_mode: str,
        rollout_method: Optional[Callable],
        train_method: Callable,
    ) -> None:
        """
        Register a rollout/train method pair for a given engine and cluster mode.

        Args:
            train_engine: Name of the training engine (e.g. "mindspeed_rl", "verl").
            cluster_mode: Cluster mode (e.g. "hybrid", "one_step_off").
            rollout_method: Callable to start rollout, or None.
            train_method: Callable to start training.
        """
        self._registry[(train_engine, cluster_mode)] = (rollout_method, train_method)

    def get_method(
        self, train_engine: str, cluster_mode: str
    ) -> Optional[Tuple[Optional[Callable], Callable]]:
        """
        Retrieve the registered rollout/train method pair.

        Args:
            train_engine: Name of the training engine.
            cluster_mode: Cluster mode.

        Returns:
            A tuple of (rollout_method, train_method) if found, None otherwise.
        """
        return self._registry.get((train_engine, cluster_mode))


registry = TrainRegistry()

try:
    from agentic_rl.trainer.rollout.rollout_main import start_rollout
    from agentic_rl.trainer.train_adapter.mindspeed_rl.hybrid_policy.train_service import train as hybrid_train
    from agentic_rl.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train.train_service import dummy_train
    from agentic_rl.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train.train_service import \
        train as one_step_off_train
    from agentic_rl.trainer.train_adapter.verl.full_async.train_main import start_train as verl_full_async_train
    from agentic_rl.trainer.train_adapter.verl.hybrid.train_main import start_train as verl_hybrid_train

    registry.register("mindspeed_rl", "hybrid", start_rollout, hybrid_train)
    registry.register("mindspeed_rl", "one_step_off", start_rollout, one_step_off_train)
    registry.register("mindspeed_rl", "dummy_train", start_rollout, dummy_train)
    registry.register("verl", "hybrid", None, verl_hybrid_train)
    registry.register("verl", "one_step_off", start_rollout, verl_full_async_train)
except ImportError:
    logger.warning("verl/mindspeed_rl hybrid train is not available, skipping registration.")

try:
    from agentic_rl.trainer.train_adapter.omni_rl.hybrid.train_main import start_train as omni_rl_hybrid_train

    registry.register("omni_rl", "hybrid", None, omni_rl_hybrid_train)
except ImportError:
    logger.warning("omni_rl hybrid train is not available, skipping registration.")
