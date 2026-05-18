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

import asyncio
import random
import traceback
from typing import AsyncGenerator, List

from aura.base.log.loggers import Loggers
from aura.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask, Trajectory

logger = Loggers(__name__).get_logger()


class AgentRouter:
    _router = None

    def __init__(self, infer_manager) -> None:
        self.agent_manager = infer_manager

    @classmethod
    async def create(cls) -> "AgentRouter":
        if cls._router is None:
            from aura.runner.agent_manager import get_or_create_agent_manager

            infer_manager = await get_or_create_agent_manager()
            cls._router = AgentRouter(infer_manager)
        return cls._router

    async def stream_generate_trajectory(self, task: AgentTask) -> AsyncGenerator[str, None]:
        try:
            infer_instance = await self.agent_manager.get_instance.remote(task.agent_name)
            executor = random.choice(infer_instance.executor_list)
            async for response in executor.stream_execute_method.remote("stream_generate_trajectory", task):
                response = await response
                if not response:
                    continue
                logger.error(f"Stream chat completion: {response}")
                yield response
        except Exception as e:
            traceback.print_exc()
            raise e

    async def generate_trajectory(
        self, task: AgentTask, mode="Text", addresses=None, server_handles=None
    ) -> Trajectory:
        infer_instance = await self.agent_manager.get_instance.remote(task.agent_name)
        executor = random.choice(infer_instance.executor_list)
        return await executor.execute_method.remote(
            "generate_trajectory", task=task, mode=mode, addresses=addresses, server_handles=server_handles
        )

    async def generate_trajectories(self, tasks: List[AgentTask], mode="Text", addresses=None) -> List[Trajectory]:
        traj_futures = []
        for task in tasks:
            future = asyncio.create_task(self.generate_trajectory(task, mode=mode, addresses=addresses))
            traj_futures.append(future)

        trajectories = await asyncio.gather(*traj_futures)
        return trajectories

    async def cancel_request(self, task: AgentTask):
        infer_instance = await self.agent_manager.get_instance.remote(task.agent_name)
        executor = random.choice(infer_instance.executor_list)
        await executor.execute_method.remote("cancel_request", task)

    async def clear_cache(self, agent_name):
        infer_instance = await self.agent_manager.get_instance.remote(agent_name)
        executor = random.choice(infer_instance.executor_list)
        await executor.execute_method.remote("clear_cache")
