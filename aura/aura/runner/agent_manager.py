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

import traceback

import ray
from omegaconf import OmegaConf

from aura.base.execution.executor_manager import ExecutorManager
from aura.base.log.loggers import Loggers
from aura.runner.agent_service.agent_executor import AgentExecutor

logger = Loggers(__name__).get_logger()


class AgentManager(ExecutorManager):
    async def setup(self, *args, **kwargs) -> None:
        try:
            from aura.base.conf.conf import AgenticRLConf

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
