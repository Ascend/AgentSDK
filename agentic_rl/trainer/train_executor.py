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
from typing import Any, Optional

import ray
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from agentic_rl.base.execution.executor import Executor, public_api
from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class TrainExecutor(Executor):
    """Executor that orchestrates distributed training and rollout via Ray."""

    def __init__(
            self,
            cluster_mode: str,
            train_engine: str,
            train_config: Any,
            rollout_config: Any,
            agent_service: str,
            infer_service: str,
            *args,
            **kwargs
    ):
        """
        Initialize the TrainExecutor.

        Args:
            cluster_mode: Deployment topology identifier (e.g. "hybrid").
            train_engine: Name of the RL training framework to use.
            train_config: Raw training configuration (converted to OmegaConf).
            rollout_config: Raw rollout configuration (converted to OmegaConf).
            agent_service: Address or identifier of the agent service.
            infer_service: Address or identifier of the inference service.
            *args: Passed through to the parent Executor.
            **kwargs: Passed through to the parent Executor.
        """
        super().__init__(*args, **kwargs)
        self.cluster_mode = cluster_mode
        self.train_engine = train_engine
        self.train_config = OmegaConf.create(train_config)
        self.rollout_config = OmegaConf.create(rollout_config)

        self.agent_service = agent_service
        self.infer_service = infer_service

        logger.info(f"TrainExecutor: rl_framework={train_engine} is initialized.")

    @classmethod
    async def _run_method(
            cls,
            start_method: Optional[Any],
            is_blocking: bool,
            *args,
            **kwargs
    ) -> None:
        """
        Schedule a Ray remote method on the current node and optionally await it.

        Args:
            start_method: Ray remote function to invoke, or None to skip.
            is_blocking: If True, await the remote future before returning.
            *args: Positional arguments forwarded to the remote call.
            **kwargs: Keyword arguments forwarded to the remote call.
        """
        if start_method is None:
            logger.warning(f"start_method={start_method} is None, args={args}, kwargs={kwargs}.")
            return

        future = start_method.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().node_id,
                soft=False
            )
        ).remote(*args, **kwargs)

        if not is_blocking:
            return
        await future

    async def _run_train_and_rollout(self) -> None:
        """Start the rollout worker (non-blocking) followed by the train worker (blocking)."""
        from agentic_rl.trainer.train_register import registry

        start_rollout, start_train = registry.get_method(
            train_engine=self.train_engine,
            cluster_mode=self.cluster_mode
        )

        await self._run_method(
            start_method=start_rollout,
            is_blocking=False,
            cluster_mode=self.cluster_mode,
            rollout_config=self.rollout_config,
            agent_service=self.agent_service,
            infer_service=self.infer_service
        )

        await self._run_method(
            start_method=start_train,
            is_blocking=True,
            cluster_mode=self.cluster_mode,
            train_config=self.train_config
        )

    @public_api(name="fit")
    async def fit(self, *args, **kwargs) -> None:
        """
        Run the full train-and-rollout pipeline.

        Raises:
            Exception: Re-raises any exception from the training pipeline
                after logging.
        """
        try:
            await self._run_train_and_rollout()
        except Exception as exc:
            logger.error(f"Training pipeline failed: {exc}.")
            raise
