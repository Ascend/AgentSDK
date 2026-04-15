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
import json
import traceback
from typing import AsyncGenerator, Dict, List

from agentic_rl.base.execution.executor import Executor, public_api
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask, Trajectory

logger = Loggers(__name__).get_logger()


class AgentExecutor(Executor):
    def __init__(
            self,
            agent_engine: Dict,
            agent_engine_kwargs: Dict,
            infer_service_params: Dict,
            trajectory_save_dir: str,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.agent_engine = agent_engine
        self.agent_engine_kwargs = agent_engine_kwargs
        self.infer_service_params = infer_service_params
        self.trajectory_save_dir = trajectory_save_dir

        if self.agent_engine == "rllm":
            from agentic_rl.runner.agent_engine_wrapper.rllm.rllm_engine_wrapper import  RLLMEngineWrapper
            self.agent_executor_wrapper = RLLMEngineWrapper(
                infer_service_params=infer_service_params,
                **agent_engine_kwargs
            )
        else:
            raise ValueError(f"{agent_engine} is not supported.")
        logger.info(f"AgentExecutor: agent_engine={agent_engine} is initialized.")

    @public_api(name="stream_generate_trajectory", is_stream=True)
    async def stream_generate_trajectory(self, task: AgentTask, *args, **kwargs) -> AsyncGenerator:
        async def _generate_trajectory_stream(t, sq):
            traj = await self.agent_executor_wrapper.generate_trajectory(task=t, stream_queue=sq)
            sq.put_nowait(None)
            return traj

        logger.info(f'generate agent trajectory: {task}')
        stream_queue = asyncio.Queue()
        future = asyncio.create_task(_generate_trajectory_stream(task, stream_queue))

        while True:
            try:
                event_item = await stream_queue.get()
                if event_item is None:
                    break
                logger.info(f'Received event: {event_item}')
            except Exception as e:
                logger.error(e)
                traceback.print_exc()
                raise e
            yield json.dumps(event_item, ensure_ascii=False)

        trajectory = await future
        logger.info(f"generate trajectory: {trajectory}.")

        if self.trajectory_save_dir.endswith('.jsonl'):
            import ray
            from agentic_rl.memory.episode.backend.json_episode_store import JsonEpisodeStore
            json_episode_store = JsonEpisodeStore(path=self.trajectory_save_dir)
            json_episode_store.store_episode(ray.get(self.agent_executor_wrapper.engine.episode.to_dict.remote()), "")

    @public_api(name="generate_trajectory")
    async def generate_trajectory(self, task: AgentTask, mode="Text", addresses=None, *args, **kwargs) -> Trajectory:
        traj = await self.agent_executor_wrapper.generate_trajectory(task=task, mode=mode, addresses=addresses)
        return traj

    @public_api(name="generate_trajectories")
    async def generate_trajectories(self, tasks: List[AgentTask], addresses=None, *args, **kwargs) -> List[Trajectory]:
        pass

    @public_api(name="cancel_request")
    async def cancel_request(self, task: AgentTask):
        await self.agent_executor_wrapper.cancel_request(task)

    @public_api(name="clear_cache")
    async def clear_cache(self):
        self.agent_executor_wrapper.clear_cache()
