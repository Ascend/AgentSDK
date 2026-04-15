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
import asyncio
from typing import Dict, List, Type, Any, Optional

import ray
from pydantic import BaseModel, ConfigDict
from ray.actor import ActorHandle

from agentic_rl.base.execution.executor import Executor
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.resources.resources import ResourceSet, create_resource_set

logger = Loggers(__name__).get_logger()


class ExecutorItem(BaseModel):
    ref: ActorHandle
    resource_set: ResourceSet

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __getattr__(self, name: str):
        """
        When accessing an inexistent attribute, it automatically retrieves from the ref.
        """
        if self.ref is not None and hasattr(self.ref, name):
            return getattr(self.ref, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class ExecutorInstance(BaseModel):
    name: str
    executor_class: Type[Executor]
    executor_num: int
    executor_kwargs: Dict[str, Any]
    resource_info: List = []
    executor_list: List[ExecutorItem] = []


class ExecutorManager:
    def __init__(self, *args, **kwargs):
        self.instance_dict: Dict[str, ExecutorInstance] = {}
        super().__init__(*args, **kwargs)

    async def setup(self, *args, **kwargs) -> None:
        """Some initialization logic that is executed at startup"""
        pass

    @classmethod
    async def _acquire_resource_set(cls, info: List) -> ResourceSet:
        """The resources required for applying for a single Executor instance"""
        if not info:
            return ResourceSet(info=info, ref=None)

        return await create_resource_set(info)

    @classmethod
    async def _release_resource_set(cls, resource_set: ResourceSet) -> None:
        """The resources required to release a single Executor instance"""
        if not resource_set.info and resource_set.ref is None:
            return

        from ray.util import remove_placement_group
        remove_placement_group(resource_set.ref)

    async def _create_executor(
            self,
            executor_class: Type[Executor],
            init_kwargs: Dict[str, Any],
            resource_info: List
    ) -> ExecutorItem:
        """Create a single Executor instance"""
        resource_set = await self._acquire_resource_set(resource_info)

        if resource_set.ref:
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=resource_set.ref,
                placement_group_bundle_index=0
            )
        else:
            scheduling_strategy = None

        executor_ref = ray.remote(executor_class).options(
            scheduling_strategy=scheduling_strategy,
            num_cpus=0
        ).remote(resource_set=resource_set, **init_kwargs)
        await executor_ref.setup.remote()
        return ExecutorItem(ref=executor_ref, resource_set=resource_set)

    async def _remove_executor(self, executor: ExecutorItem, ) -> None:
        """Stop and remove a single Executor"""
        await executor.finalize()
        ray.kill(executor.ref)
        await self._release_resource_set(executor.resource_set)

    async def create_instance(
            self,
            name: str,
            executor_class: Type[Executor],
            executor_num: int,
            executor_kwargs: Optional[Dict[str, Any]] = None,
            resource_info: Optional[List] = None,
    ) -> ExecutorInstance:
        """Create a new instance group of the executor"""
        if name in self.instance_dict:
            raise ValueError(f"Instance '{name}' already exists")

        executor_kwargs = executor_kwargs or {}
        resource_info = resource_info or []
        executor_list = []

        futures = [
            asyncio.create_task(self._create_executor(executor_class, executor_kwargs, resource_info))
            for _ in range(executor_num)
        ]
        for future in futures:
            executor = await future
            executor_list.append(executor)

        instance = ExecutorInstance(
            name=name,
            executor_class=executor_class,
            executor_num=executor_num,
            executor_kwargs=executor_kwargs,
            executor_list=executor_list,
        )
        self.instance_dict[name] = instance
        logger.info(f"[ExecutorManager] Created instance '{name}' with {executor_num} executors, instance={instance}")
        return instance

    async def get_instance(self, name: str) -> Optional[ExecutorInstance]:
        """Obtain the instance group of the actuator with the specified name"""
        if name not in self.instance_dict:
            raise ValueError(f"Instance '{name}' not found")
        return self.instance_dict.get(name)

    async def remove_instance(self, name: str) -> None:
        """Remove and clean up an instance group of actuators"""
        if name not in self.instance_dict:
            logger.info(f"[ExecutorManager] Instance '{name}' not found")
            return

        instance = self.instance_dict.pop(name, None)
        for executor in instance.executor_list:
            await self._remove_executor(executor)

        logger.info(f"[ExecutorManager] Instance '{name}' removed")

    async def finalize(self) -> None:
        """Clean up all the actuator instances"""
        names = list(self.instance_dict.keys())
        for name in names:
            await self.remove_instance(name)
        logger.info("[ExecutorManager] finalized all instances")
