#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

import asyncio
import logging
import os
import uuid
from asyncio import Queue
from typing import List, Protocol, Optional
from transformers import AutoTokenizer

from aura.base.utils.load_object_by_path import load_object_by_path
from aura.runner.agent_engine_wrapper.base_engine_wrapper import BaseEngineWrapper, AgentTask
from aura.runner.agent_engine_wrapper.proxy_client.agent_proxy_client import AgentProxyClient
from aura.runner.agent_engine_wrapper.proxy_client.traj_proxy_client import TrajProxyClient
from aura.runner.agent_engine_wrapper.vaee_v2.default_traj_refine_reward_func import default_token_traj_refine_func
from aura.runner.agent_engine_wrapper.vaee_v2.vaee_types import Episode, RequestRecord

logger = logging.getLogger(__name__)


class TrajRefineRewardCls(Protocol):
    """Trajectory processing function

    refine:
        Responsibilities —— Aggregate/filter/reorganize multiple records to construct a hierarchical trajectory
        Input —— List of record requests
        Output —— Episode hierarchical trajectory

    reward:
        Responsibilities —— Calculate the reward value of the trajectory
        Input —— Episode hierarchical trajectory
        Output —— Episode hierarchical trajectory
    """

    async def refine(self, task_id: str, records: List[RequestRecord], tokenizer=None) -> Episode: ...

    async def reward(
        self,
        episode: Episode,
        answer: Optional[str] = None,
    ) -> Episode: ...


class VirtualAgentEngineExecutionWrapper(BaseEngineWrapper):
    """
    VirtualAgentEngineExecutionWrapper processing flow:
    1. Send a request to the Agent service and wait for execution to complete
    2. Get request data from the TrajProxy service, construct the trajectory, calculate the reward, and return the trajectory
    """

    def __init__(
        self,
        infer_service_params,
        traj_reward_func: str,
        traj_refine_func: str = None,
        tokenizer: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.agent_engine_kwargs = kwargs
        self.sampling_params = infer_service_params
        self.tokenizer_name_or_path = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)

        self.traj_refine_func = (
            load_object_by_path(traj_refine_func) if traj_refine_func else default_token_traj_refine_func
        )
        self.traj_reward_func = load_object_by_path(traj_reward_func)
        self.task_index = 0

        self._init_agent_and_env_args(*args, **kwargs)
        logger.info(">>> VirtualAgentEngineExecutionWrapper args: %s, kwargs", args, kwargs)

    async def generate_trajectory(
        self, task: AgentTask, stream_queue: Queue = None, *args: object, **kwargs: object
    ) -> Episode:
        task = task.model_dump()
        prompt_id = task['prompt_id'] if 'prompt_id' in task else 0
        task_id = f"{task['task_id']}-{prompt_id}-{str(uuid.uuid4())}-{str(os.getpid())}-{self.task_index}"
        info = await self._create_agent_and_env(task_id, task)
        await self._generate_trajectory(task_id, task, info, *args, **kwargs)
        return await self._post_process_trajectory(task_id, task, info, *args, **kwargs)

    def _init_agent_and_env_args(self, *args, **kwargs):
        import signal
        import threading

        _original_signal = signal.signal

        def _noop_signal(*args, **kwargs):
            if threading.current_thread() is not threading.main_thread():
                return
            return _original_signal(*args, **kwargs)

        signal.signal = _noop_signal

        agent_name = kwargs["agent_name"]

        from agents.agents_mapping import get_agent_by_name

        agent = get_agent_by_name(agent_name)
        if agent is None:
            raise RuntimeError(f"Agent {agent_name} not found.")

        self.agent_class = agent.get("agent_class")
        self.agent_args = agent.get("agent_args")
        self.env_class = agent.get("env_class")
        self.env_args = agent.get("env_args")
        self.compute_trajectory_reward_fn = agent.get("compute_trajectory_reward_fn", None)
        self.env_args["tokenizer"] = self.tokenizer
        agent_chat_parser = agent.get("chat_parser", None)
        if agent_chat_parser is not None:
            self.chat_parser = agent_chat_parser(
                self.tokenizer, disable_thinking=self.sampling_params.get("disable_thinking", False)
            )
        else:
            from aura.runner.agent_engine_wrapper.base.parser.chat_template import ChatTemplateParser

            self.chat_parser = ChatTemplateParser.get_parser(
                self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False), toolcall_parser="qwen3_coder"
            )

        env_args = self.env_args | kwargs.get("env_args", {})
        self.trajectory_timeout = env_args.get("trajectory_timeout") if "trajectory_timeout" in env_args else 7200
        self.max_steps = env_args.get("max_steps") if "max_steps" in env_args else 10
        for key, val in env_args.items():
            if val is not None:
                self.env_args[key] = val
        for key, val in self.sampling_params.items():
            if val is not None:
                self.env_args[key] = val

        agent_args = self.agent_args | kwargs.get("agent_args", {})
        self.overlong_filter = agent_args.get("overlong_filter", False)
        for key, val in agent_args.items():
            if val is not None:
                self.agent_args[key] = val

        model_name = self.sampling_params.get("model", "")
        params = {}
        params["infer_params"] = self.sampling_params
        params["extra_params"] = self.agent_engine_kwargs

        agent_proxy_kwargs = self.agent_engine_kwargs.get("agent_proxy_args", {})
        agent_proxy_kwargs["model_name"] = model_name
        agent_proxy_kwargs["params"] = params
        agent_proxy_kwargs["timeout"] = self.trajectory_timeout
        self.agent_proxy_class = agent.get("agent_proxy_class")
        self.agent_proxy_client: AgentProxyClient = self.agent_proxy_class(**agent_proxy_kwargs)

        self.traj_proxy_class = agent.get("traj_proxy_class")
        traj_proxy_kwargs = self.agent_engine_kwargs.get("traj_proxy_args", {})
        traj_proxy_kwargs["model_name"] = model_name
        self.traj_proxy_client: TrajProxyClient = self.traj_proxy_class(**traj_proxy_kwargs)

    async def _create_agent_and_env(self, task_id, task):
        self.env_args["task"] = task
        env = self.env_class.from_dict({**self.env_args})
        env.application_id = task_id
        agent = self.agent_class(**self.agent_args)

        observation, info = await asyncio.to_thread(env.reset)
        logger.info(f">>> initial observation: {observation}")
        info["max_steps"] = self.max_steps

        agent.reset()
        agent.update_from_env(
            observation=observation,  # Raw observation from environment
            reward=0.0,
            done=False,
            info=info,
        )
        messages = agent.chat_completions
        logger.info(f">>> initial messages: {messages}")

        return info

    async def _generate_trajectory(self, task_id, question, info, *args, **kwargs):
        """
        Call remote Agent to generate trajectory
        """
        ret, session_id = await self.agent_proxy_client.get_agent_response(question, task_id, **{"info": info})
        if ret == 0:
            return
        raise Exception(f"proxy_worker get_agent_response failed for task_id: {task_id}")

    async def _post_process_trajectory(self, task_id, task, info, *args, **kwargs) -> Episode:
        """
        Find an agent to retrieve the trajectory and handle the return
        """
        session_id = task_id

        # 1. Get records from TrajProxy
        records = await self.traj_proxy_client.get_records_by_session(session_id)
        records = [RequestRecord(**r) for r in records]

        # 2. Filter all abnormal requests
        records = [r for r in records if r.error_traceback is None]
        if not records:
            logger.warning(f"No records found for session {session_id}")
            raise ValueError(f"No records found for session {session_id}")

        # 3. Trajectory Processing (Aggregation), Return Episode
        episode: Episode = self.traj_refine_func(task_id, task, records, self.tokenizer)

        # 4. Trajectory reward calculation, fill traj_reward, return Episode
        episode: Episode = self.traj_reward_func(episode, task["ground_truth"], info)
        return episode
