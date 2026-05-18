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

import json
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

from transformers import AutoTokenizer

from aura.base.log.loggers import Loggers
from aura.memory.episode.episode import Episode
from aura.runner.agent_engine_wrapper.base_engine_wrapper import BaseEngineWrapper, AgentTask, Trajectory

logger = Loggers(__name__).get_logger()


class RLLMEngineWrapper(BaseEngineWrapper):
    def __init__(
        self,
        infer_service_params,
        agent_name,
        tokenizer: str,
        traj_proxy_url="",
        agent_proxy_url="",
        traj_proxy_run_id="",
        simplify_think_content=False,
        max_prompt_length=8192,
        max_model_len=16384,
        n_parallel_agents=8,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        from aura.runner.agent_service.chat_proxy import patch_async_openai_global

        patch_async_openai_global(infer_service_params)

        _original_signal = signal.signal

        def _noop_signal(*args, **kwargs):
            if threading.current_thread() is not threading.main_thread():
                return
            return _original_signal(*args, **kwargs)

        signal.signal = _noop_signal

        self.server_addresses = ["0.0.0.0:8000"]
        self.simplify_think_content = simplify_think_content
        self.max_prompt_length = max_prompt_length
        self.max_model_len = max_model_len
        self.n_parallel_agents = n_parallel_agents
        self.token_in_token_out = kwargs.get("token_in_token_out", False)
        # for vaee
        self.agent_engine_kwargs = {
            "agent_name": agent_name,
            "simplify_think_content": simplify_think_content,
            "max_prompt_length": max_prompt_length,
            "max_model_len": max_model_len,
            "n_parallel_agents": n_parallel_agents,
            **kwargs,
        }

        if traj_proxy_url and agent_proxy_url:
            # URL format validation: must start with http:// or https://, followed by IP:port
            import re

            url_pattern = re.compile(r"^https?://\d{1,3}(\.\d{1,3}){3}:\d+$")

            if not url_pattern.match(traj_proxy_url):
                raise ValueError(f"Invalid format for traj_proxy_url: {traj_proxy_url}")

            if not url_pattern.match(agent_proxy_url):
                raise ValueError(f"Invalid format for traj_proxy_url: {traj_proxy_url}")

        self.traj_proxy_url = traj_proxy_url
        self.agent_proxy_url = agent_proxy_url
        self.traj_proxy_run_id = traj_proxy_run_id

        self.tokenizer_name_or_path = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, trust_remote_code=True)
        self.sampling_params = infer_service_params

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
            self.chat_parser = None

        env_args = self.env_args | kwargs.get("env_args", {})
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

        self.traj_proxy_class = agent.get("traj_proxy_class")
        self.agent_proxy_class = agent.get("agent_proxy_class")

        self.episode = Episode.remote(episode_id="RLLMEngineWrapper")
        logger.info(
            f"agent class: {self.agent_class}, env class: {self.env_class}, "
            f"env args: {self.env_args}, max steps: {self.max_steps}"
        )

        self.engine = None

    def update_envs_and_agents(self, envs, agents, iteration, sample_id):
        self.engine.update_envs_and_agents(envs, agents, iteration, sample_id)

    def update_env_and_agent(self, task_id, env, agent, iteration, sample_id):
        self.engine.update_env_and_agent(task_id, env, agent, iteration, sample_id)

    def release_env_and_agent(self, task_id):
        self.engine.release_env_and_agent(task_id)

    def init_envs_and_agents(self, tasks):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """

        def _create_env(i):
            if isinstance(tasks[i], str):
                tasks[i] = json.loads(tasks[i])
            self.env_args["task"] = tasks[i]
            return i, self.env_class.from_dict({**self.env_args})

        def _create_agent(i):
            return i, self.agent_class(**self.agent_args)

        envs = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=64) as executor:
            env_futures = [executor.submit(_create_env, i) for i in range(len(tasks))]
            for future in as_completed(env_futures):
                idx, env = future.result()
                envs[idx] = env

        agents = [None] * len(envs)
        with ThreadPoolExecutor(max_workers=64) as executor:
            agent_futures = [executor.submit(_create_agent, i) for i in range(len(envs))]
            for future in as_completed(agent_futures):
                idx, agent = future.result()
                agents[idx] = agent

        iteration = tasks[0].iteration
        sample_id = tasks[0].sample_id
        self.update_envs_and_agents(envs, agents, iteration, sample_id)
        return envs

    def _create_engine(self):
        if self.engine is not None:
            return

        if self.traj_proxy_url and self.agent_proxy_url:
            from aura.runner.agent_engine_wrapper.vaee.extern_agent_wrapper import ExternAgentWrapper

            logger.info("using ExternAgentWrapper")
            self.engine = ExternAgentWrapper(
                chat_parser=self.chat_parser,
                server_addresses=self.server_addresses,
                agent_class=self.agent_class,
                agent_args=self.agent_args,
                env_class=self.env_class,
                env_args=self.env_args,
                tokenizer=self.tokenizer,
                compute_trajectory_reward_fn=self.compute_trajectory_reward_fn,
                sampling_params=self.sampling_params,
                max_prompt_length=self.max_prompt_length,
                simplify_think_content=self.simplify_think_content,
                max_model_len=self.max_model_len,
                n_parallel_agents=self.n_parallel_agents,
                max_workers=self.n_parallel_agents,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
                max_steps=self.max_steps,
                token_in_token_out=self.token_in_token_out,
                overlong_filter=self.overlong_filter,
                traj_proxy_url=self.traj_proxy_url,  # for vaee
                agent_proxy_url=self.agent_proxy_url,  # for vaee
                traj_proxy_run_id=self.traj_proxy_run_id,
                agent_engine_kwargs=self.agent_engine_kwargs,  # for vaee
            )
        else:
            from aura.runner.agent_engine_wrapper.rllm.agent_execution_engine import AgentExecutionEngine

            self.engine = AgentExecutionEngine(
                chat_parser=self.chat_parser,
                server_addresses=self.server_addresses,
                agent_class=self.agent_class,
                agent_args=self.agent_args,
                env_class=self.env_class,
                env_args=self.env_args,
                tokenizer=self.tokenizer,
                compute_trajectory_reward_fn=self.compute_trajectory_reward_fn,
                sampling_params=self.sampling_params,
                max_prompt_length=self.max_prompt_length,
                simplify_think_content=self.simplify_think_content,
                max_model_len=self.max_model_len,
                n_parallel_agents=self.n_parallel_agents,
                max_workers=self.n_parallel_agents,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
                max_steps=self.max_steps,
                token_in_token_out=self.token_in_token_out,
                overlong_filter=self.overlong_filter,
            )

    async def generate_trajectory(
        self,
        task: AgentTask,
        stream_queue: Queue = None,
        mode="Text",
        addresses: list = None,
        server_handles=None,
        *args,
        **kwargs,
    ) -> Trajectory:
        """
        Trajectory generation: supports both streaming and non-streaming modes.
        """
        iteration = task.iteration
        sample_id = task.sample_id
        task_id = task.task_id
        task = task.model_dump()
        logger.debug(f"generate_trajectory task: {task}, stream_queue={stream_queue}")
        extra_args = task.get("extra_args", {})
        extra_args = {} if extra_args is None else extra_args
        env = self.env_class.from_dict(self.env_args | extra_args | {"task": task})
        prompt_id = task['prompt_id'] if 'prompt_id' in task else 0
        agent = self.agent_class(**self.agent_args)

        self._create_engine()

        self.engine.init_router(addresses)

        self.engine.update_env_and_agent(task_id, env, agent, iteration, sample_id)
        trajectory = None
        async for item in self.engine.trajectory_generator(
            task=task, stream_queue=stream_queue, mode=mode, prompt_id=prompt_id, server_handles=server_handles
        ):
            trajectory = item
        self.engine.release_env_and_agent(task_id)
        return trajectory

    async def cancel_request(self, task: AgentTask):
        logger.info(f"cancel request task_id: {task.task_id}")
        await self.engine.cancel_request(task)

    def clear_cache(self):
        self.engine.clear_cache()
