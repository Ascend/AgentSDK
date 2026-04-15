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
import random
from typing import Any

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class TrainRouter:
    """Routes training requests to executor instances managed by TrainManager."""

    _router = None

    def __init__(self, train_manager) -> None:
        """
        Initialize TrainRouter with a TrainManager actor handle.

        Args:
            train_manager: Ray actor handle for the TrainManager.
        """
        self.train_manager = train_manager

    @classmethod
    async def create(cls) -> "TrainRouter":
        """
        Get or create the singleton TrainRouter instance.

        Returns:
            The singleton TrainRouter instance.
        """
        if cls._router is None:
            from agentic_rl.trainer.train_manager import get_or_create_train_manager

            train_manager = await get_or_create_train_manager()
            cls._router = TrainRouter(train_manager)
            logger.info("Train router created.")
        return cls._router

    async def train(self, name: str) -> Any:
        """
        Route a training request to a randomly selected executor.

        Args:
            name: Name of the training instance to invoke.

        Returns:
            Result of the executor's fit method.
        """
        infer_instance = await self.train_manager.get_instance.remote(name)
        executor = random.choice(infer_instance.executor_list)
        return await executor.execute_method.remote("fit")
