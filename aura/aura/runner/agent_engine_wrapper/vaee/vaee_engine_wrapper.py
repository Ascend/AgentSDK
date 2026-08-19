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
import os
import uuid
from asyncio import Queue
from typing import List, Protocol, Optional
from transformers import AutoTokenizer
import concurrent.futures
from functools import partial

from aura.base.utils.load_object_by_path import load_object_by_path
from aura.runner.agent_engine_wrapper.base_engine_wrapper import BaseEngineWrapper, AgentTask
from aura.runner.agent_engine_wrapper.proxy_client.agent_proxy_client import AgentProxyClient
from aura.runner.agent_engine_wrapper.proxy_client.traj_proxy_client import TrajProxyClient, calculate_time_diff_seconds
from aura.runner.agent_engine_wrapper.vaee.default_traj_refine import (
    default_step_traj_refine_func,
    default_token_traj_refine_func,
)
from aura.runner.agent_engine_wrapper.vaee.default_traj_reward import default_traj_reward_func
from aura.runner.agent_engine_wrapper.vaee.vaee_types import Episode, RequestRecord
from aura.runner.agent_engine_wrapper.vaee.tool_env import ToolEnvironment
from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


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

    async def refine(self,
        task_id: str,
        records: List[RequestRecord],
        tokenizer=None,
        *args,
        **kwargs
    ) -> Episode:
        """
        Aggregate/filter/reorganize multiple records to construct a trajectory.

        Args:
            task_id: Task identifier.
            records: List of request records to be refined.
            tokenizer: Tokenizer for processing text.

        Returns:
            Episode: Hierarchical trajectory result.
        """
        ...

    async def reward(self,
        episode: Episode,
        answer: Optional[str] = None,
        *args,
        **kwargs
    ) -> Episode:
        """
        Calculate the reward value of the trajectory.

        Args:
            episode: Hierarchical trajectory to evaluate.
            answer: Optional reference answer for reward calculation.

        Returns:
            Episode: Episode with reward values populated.
        """
        ...


class VirtualAgentEngineExecutionWrapper(BaseEngineWrapper):
    """
    VirtualAgentEngineExecutionWrapper processing flow:
    1. Send a request to the Agent service and wait for execution to complete
    2. Get request data from the TrajProxy service, construct the trajectory, calculate the reward, and return the trajectory
    """

    def __init__(self,
                 infer_service_params,
                 traj_reward_func: str = None,
                 traj_refine_func: str = None,
                 res_reward_func: str = None,
                 tokenizer: str = None,
                 n_parallel_agents=8,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.agent_engine_kwargs = kwargs
        self.sampling_params = infer_service_params
        self.tokenizer_name_or_path = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)

        self.traj_refine_func = (
            load_object_by_path(traj_refine_func) if traj_refine_func else default_token_traj_refine_func
        )
        self.traj_reward_func = (
            load_object_by_path(traj_reward_func) if traj_reward_func else default_traj_reward_func
        )
        self.res_reward_func = load_object_by_path(res_reward_func) if res_reward_func else None
        logger.info(f"[INIT] traj_refine_func loaded: {self.traj_refine_func.__name__ if self.traj_refine_func else None}, "
                     f"origin_path={traj_refine_func}")
        logger.info(f"[INIT] res_reward_func loaded: {self.res_reward_func.__name__ if self.res_reward_func else None}, "
                     f"origin_path={res_reward_func}")
        self.task_index = 0
        self.token_in_token_out = kwargs.get("token_in_token_out", False)
        self.compress_steps = kwargs.get("compress_steps", False)
        self.max_model_len = kwargs.get("max_model_len", 128 * 1024)

        # Create a thread pool executor for environment interactions (i.e. step, reset, close)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel_agents)
        self._init_agent_and_env_args(*args, **kwargs)
        self.tool_timeout = kwargs.get("tool_timeout") if "tool_timeout" in kwargs else 2000
        logger.info(f">>> VirtualAgentEngineExecutionWrapper args: {args}, kwargs: {kwargs}")

    async def cancel_request(self, task):
        pass

    async def calc_rewards(self, episode: Episode, task: dict):
        if self.res_reward_func is None:
            logger.info(f"[calc_rewards] skipped: res_reward_func is None, task_id={episode.id}")
            return

        logger.info(f"[calc_rewards] >>> ENTERED calc_rewards: task_id={episode.id}, "
                     f"num_trajs={len(episode.trajectories)}, "
                     f"res_reward_func={self.res_reward_func.__name__}")
        total_steps = 0
        zero_rewards = 0
        non_zero_rewards = 0
        for traj_idx, traj in enumerate(episode.trajectories):
            tool_env = ToolEnvironment(task, self.res_reward_func, self.max_steps)
            num_steps = len(traj.steps)
            total_steps += num_steps
            logger.info(f"[calc_rewards] traj[{traj_idx}]: num_steps={num_steps}, task_id={episode.id}")
            for i in range(num_steps):
                step = traj.steps[i]
                step.reward, step.done = tool_env.step(step, traj.steps[:i+1], i >= num_steps - 1)
                logger.info(f"[calc_rewards] step result: task_id={episode.id}, traj_idx={traj_idx}, "
                             f"step_idx={i}, is_last={i >= num_steps - 1}, "
                             f"reward={step.reward}, done={step.done}")
                if step.reward == 0.0:
                    zero_rewards += 1
                else:
                    non_zero_rewards += 1
        logger.info(f"[calc_rewards] <<< EXIT calc_rewards: task_id={episode.id}, "
                     f"total_steps={total_steps}, zero_rewards={zero_rewards}, "
                     f"non_zero_rewards={non_zero_rewards}")

    def clear_cache(self):
        pass

    def stat(self, records: list[RequestRecord], episode: Episode) -> dict:

        sorted_records = sorted(records, key=lambda r: r.start_time)

        filtered_records = [record for record in sorted_records if record.response_text != "\n\nSAFE"]

        step_infos = []
        for i, record in enumerate(filtered_records):
            if record.raw_response is None:
                continue

            step_info = {
                    'step_id': i,
                    'llm_time': calculate_time_diff_seconds(record.start_time, record.end_time),
                    'env_time': 0.0
                }

            if i > 0:
                prev_end_time = filtered_records[i - 1].end_time
                step_infos[i - 1]['env_time'] = calculate_time_diff_seconds(prev_end_time, record.start_time)
            step_infos.append(step_info)

        env_step_times = [step_info['env_time'] for step_info in step_infos]
        llm_step_times = [step_info['llm_time'] for step_info in step_infos]
        env_time = sum(env_step_times)
        llm_time = sum(llm_step_times)
        prompt_len = 0
        response_len = 0
        if len(episode.trajectories) > 0 and len(episode.trajectories[0].steps) > 0:
            prompt_len = max([len(step.prompt_ids) for step in episode.trajectories[0].steps])
            response_len = max([len(step.response_ids) for step in episode.trajectories[0].steps])
        metrics = {
            "steps": len(step_infos),
            "reward": episode.trajectories[0].reward if len(episode.trajectories) > 0 else 0.0,
            "toolcall_reward": episode.trajectories[0].toolcall_reward if len(episode.trajectories) > 0 else 0.0,
            "res_reward": episode.trajectories[0].res_reward if len(episode.trajectories) > 0 else 0.0,
            "reward_time": 0.0,
            "env_time": env_time,
            "llm_time": llm_time,
            "total_time": env_time + llm_time,
            "env_step_times": env_step_times,
            "llm_step_times": llm_step_times,
            "prompt_len": prompt_len,
            "response_len": response_len
        }
        return metrics

    async def generate_trajectory(self,
                                  task: AgentTask,
                                  stream_queue: Queue = None,
                                  *args: object, **kwargs: object) -> Episode:
        task = task.model_dump()
        prompt_id = task['prompt_id'] if 'prompt_id' in task else 0
        task_id = f"{task['task_id']}-{prompt_id}-{str(uuid.uuid4())}-{str(os.getpid())},{self.task_index}"
        info = await self._create_agent_and_env(task_id, task)
        task_id = await self._generate_trajectory(task_id, task, info, *args, **kwargs)

        return await self._post_process_trajectory(task_id, task, info, *args, **kwargs)

    def set_noop_signal(self):
        import signal
        import threading

        self._original_signal = signal.signal

        def _noop_signal(*args, **kwargs):
            if threading.current_thread() is not threading.main_thread():
                return
            return self._original_signal(*args, **kwargs)

        signal.signal = _noop_signal

    def _init_agent_and_env_args(self, *args, **kwargs):
        self.set_noop_signal()

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
                self.tokenizer,
                disable_thinking=self.sampling_params.get("disable_thinking", False)
            )
        else:
            from aura.runner.agent_engine_wrapper.base.parser.chat_template import ChatTemplateParser
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer,
                                                             disable_thinking=kwargs.get("disable_thinking", False),
                                                             toolcall_parser="qwen3_coder")

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
        self.agent_proxy_class = agent.get("agent_proxy_class", AgentProxyClient)
        self.agent_proxy_client: AgentProxyClient = self.agent_proxy_class(**agent_proxy_kwargs, executor=self.executor)

        self.traj_proxy_class = agent.get("traj_proxy_class", TrajProxyClient)
        traj_proxy_kwargs = self.agent_engine_kwargs.get("traj_proxy_args", {})
        traj_proxy_kwargs["model_name"] = model_name
        self.traj_proxy_client: TrajProxyClient = self.traj_proxy_class(**traj_proxy_kwargs)
        self.run_id = agent_proxy_kwargs["run_id"] if "run_id" in agent_proxy_kwargs else "app-unknow"

    async def _create_agent_and_env(self, task_id, task):
        """ Compatible with rLLM """
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
            return session_id
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
        episode: Episode = self.traj_refine_func(task_id, task, records, self.tokenizer,
            compress_steps=self.compress_steps,
            max_model_len=self.max_model_len)

        # 4. Trajectory reward calculation, fill traj_reward, return Episode
        if self.res_reward_func:
            await self.calc_rewards(episode, task)

        # 5. Compute trajectory reward, fill traj_reward, return Episode
        episode: Episode = self.traj_reward_func(episode, task["ground_truth"], info)

        # 6. Collect metrics
        try:
            episode.metrics = self.stat(records, episode)
        except Exception as e:
            logger.error(f"stat exception: e={e}")
            episode.metrics = {}

        logger.info(f"trajectory performance status, appID:{task_id}, metrics:{episode.metrics}")
        return episode
