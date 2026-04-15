#!/usr/bin/env python3
# coding=utf-8

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


# Standard library imports
import asyncio
import traceback

# Third-party library imports
import ray
from omegaconf import OmegaConf

# Internal imports
from agentic_rl.base.execution.executor_manager import ExecutorManager
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.runner.infer_service.infer_executor import InferExecutor
from agentic_rl.runner.infer_service.infer_pd_executor import InferPDSepExecutor

logger = Loggers(__name__).get_logger()


class InferManager(ExecutorManager):
    async def setup(self, *args, **kwargs) -> None:
        try:
            from agentic_rl.base.conf.conf import AgenticRLConf
            conf = AgenticRLConf.load_config()
            conf_kwargs = OmegaConf.to_container(conf)

            if 'pd_mode' in conf_kwargs and conf_kwargs['pd_mode']:
                executor_class = InferPDSepExecutor
                instances = conf.infer_pd_instances
            else:
                executor_class = InferExecutor
                instances = conf.infer_instances
            logger.info(f"instances: {instances}")
            for instance_conf in instances:
                await self.create_instance(
                    name=instance_conf.name,
                    executor_class=executor_class,
                    executor_num=instance_conf.executor_num,
                    executor_kwargs=OmegaConf.to_container(instance_conf.executor_kwargs),
                    resource_info=OmegaConf.to_container(instance_conf.resource_info),
                )
            logger.info(f"Infer manager created, instance list={self.instance_dict.keys()}.")
        except Exception as e:
            traceback.print_exc()
            raise e


async def get_or_create_infer_manager():
    actor_name = "InferManager"
    try:
        return ray.get_actor(actor_name)
    except ValueError as _:
        logger.info(f"Could not find actor {actor_name}, creating a new one.")
    manager = ray.remote(InferManager).options(name="InferManager", lifetime="detached").remote()
    await manager.setup.remote()
    return manager

def destroy_infer_manager():
    actor_name = "InferManager"
    try:
        infer_manager = ray.get_actor(actor_name)
        ray.kill(infer_manager)
    except ValueError as _:
        logger.info(f"Could not find actor {actor_name}, do not destroy.")
