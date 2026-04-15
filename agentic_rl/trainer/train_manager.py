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
from omegaconf import OmegaConf

from agentic_rl.base.execution.executor_manager import ExecutorManager
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.trainer.train_executor import TrainExecutor

logger = Loggers(__name__).get_logger()


class TrainManager(ExecutorManager):
    """Manages training executor instances via Ray actors."""

    async def setup(self, *args, **kwargs) -> None:
        """
        Initialize training instances from configuration.

        Loads AgenticRLConf and creates executor instances for
        each configured train instance.

        Raises:
            Exception: If configuration loading or instance creation fails.
        """
        try:
            # Lazy import to avoid circular dependency at module level
            from agentic_rl.base.conf.conf import AgenticRLConf

            conf = AgenticRLConf.load_config()
            for instance_conf in conf.train_instances:
                logger.info(f"instance_conf: {instance_conf}")
                await self.create_instance(
                    name=instance_conf.name,
                    executor_class=TrainExecutor,
                    executor_num=instance_conf.executor_num,
                    executor_kwargs=OmegaConf.to_container(instance_conf.executor_kwargs, resolve=True),
                    resource_info=OmegaConf.to_container(instance_conf.resource_info),
                )
            logger.info(f"Train manager created, instance list={self.instance_dict.keys()}.")
        except Exception as e:
            logger.error("Train manager setup failed: %s", e)
            raise


async def get_or_create_train_manager():
    """
    Get an existing TrainManager Ray actor or create a new one.

    Returns:
        A Ray actor handle for the TrainManager.
    """
    actor_name = "TrainManager"
    try:
        return ray.get_actor(actor_name)
    except ValueError:
        logger.info(f"Could not find actor {actor_name}, creating a new one.")

    manager = ray.remote(TrainManager).options(name="TrainManager", lifetime="detached").remote()
    await manager.setup.remote()
    return manager
