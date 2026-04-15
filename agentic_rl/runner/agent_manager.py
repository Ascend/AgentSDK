#!/usr/bin/python
# -*- coding: utf-8 -*-
import asyncio
import traceback

import ray
from omegaconf import OmegaConf

from agentic_rl.base.execution.executor_manager import ExecutorManager
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.runner.agent_service.agent_executor import AgentExecutor

logger = Loggers(__name__).get_logger()

class AgentManager(ExecutorManager):
    async def setup(self, *args, **kwargs) -> None:
        try:
            from agentic_rl.base.conf.conf import AgenticRLConf
            conf = AgenticRLConf.load_config()
            for instance_conf in conf.agent_instances:
                logger.info(f"Agent manager instance conf: {instance_conf}")
                await self.create_instance(
                    name=instance_conf.name,
                    executor_class=AgentExecutor,
                    executor_num=instance_conf.executor_num,
                    executor_kwargs=OmegaConf.to_container(instance_conf.executor_kwargs),
                    resource_info=OmegaConf.to_container(instance_conf.resource_info),
                )
            logger.info(f"Agent manager created, instance list={self.instance_dict.keys()}.")
        except Exception as e:
            traceback.print_exc()
            raise e

async def get_or_create_agent_manager():
    actor_name = "AgentManager"
    try:
        return ray.get_actor(actor_name)
    except ValueError as _:
        logger.info(f"Could not find actor {actor_name}, creating a new one.")
    manager = ray.remote(AgentManager).options(name="AgentManager", lifetime="detached").remote()
    await manager.setup.remote()
    return manager

def destroy_agent_manager():
    actor_name = "AgentManager"
    try:
        manager = ray.get_actor(actor_name)
        ray.kill(manager)
    except ValueError as _:
        logger.info(f"Could not find actor {actor_name}, do not destroy.")
