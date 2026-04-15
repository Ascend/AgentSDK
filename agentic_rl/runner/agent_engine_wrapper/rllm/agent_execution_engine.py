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
import concurrent.futures
import hashlib
import os
import re
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch

import agentic_rl.runner.agent_engine_wrapper.rllm.patch
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.misc.misc import app_stats, colorful_print
from agentic_rl.base.utils.utils import strftime
from agentic_rl.runner.agent_engine_wrapper.base.agent.base_agent import Action, BaseAgent, Trajectory
from agentic_rl.runner.agent_engine_wrapper.base.environment.base_env import BaseEnv
from agentic_rl.runner.agent_engine_wrapper.base.environment.env_utils import compute_mc_return, compute_trajectory_reward
from agentic_rl.runner.agent_engine_wrapper.base.parser.chat_template import ChatTemplateParser
from agentic_rl.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask
from agentic_rl.runner.agent_engine_wrapper.rllm.msg_handler import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)

logger = Loggers(__name__).get_logger()

GLOBAL_INDEX = 0
DEFAULT_DATA_ID = "000000000000000000000000000000000"


def create_application_id(prompt_id: int):
    global GLOBAL_INDEX
    application_id = str(prompt_id) + '-' + str(uuid.uuid4()) + str(os.getpid()) + str(GLOBAL_INDEX)
    GLOBAL_INDEX = GLOBAL_INDEX + 1
    return application_id


def _generate_key(task):
    key = None
    if isinstance(task, dict):
        key = task['task_id'] + "_" + str(task['prompt_id'])
    elif isinstance(task, AgentTask):
        key = task.task_id + "_" + str(task.prompt_id)
    logger.debug(f"generate key {key}")
    return key


class AgentExecutionEngine:
    def __init__(
        self,
        tokenizer=None,
        server_addresses=None,
        chat_parser=None,
        n_parallel_agents=1,
        gamma=0.2,
        api_retries=3,
        retry_limit=3,
        max_steps=5,
        max_prompt_length=1024,
        simplify_think_content=False,
        max_model_len=16384,
        compute_trajectory_reward_fn=compute_trajectory_reward,
        tokenizer_name_or_path=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        env_args=None,
        max_workers=64,
        enforce_max_prompt_length=False,
        overlong_filter=False,
        **kwargs,
    ):
        if agent_args is None:
            agent_args = {}
        if env_args is None:
            env_args = {}

        self.simplify_think_content = simplify_think_content
        self.max_model_len = max_model_len

        self.tokenizer = tokenizer
        self.n_parallel_agents = n_parallel_agents
        self.overlong_filter = overlong_filter

        self.gamma = gamma
        self.retry_limit = retry_limit
        self.api_retries = api_retries
        self.max_steps = max_steps
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length

        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_args = env_args
        self.compute_trajectory_reward_fn = compute_trajectory_reward_fn \
            if compute_trajectory_reward_fn is not None else compute_trajectory_reward

        self.agents = [None for _ in range(n_parallel_agents)]
        self.envs = [None for _ in range(n_parallel_agents)]
        self.agent_dict = {}
        self.env_dict = {}

        self.tool_timeout = env_args.get("tool_timeout") if "tool_timeout" in env_args else 2000
        self.trajectory_timeout = env_args.get("trajectory_timeout") if "trajectory_timeout" in env_args else 7200

        if env_class is not None:
            if not env_class.is_multithread_safe():
                raise TypeError("Environment must be multi-thread safe for async engine")
        self.sampling_params = kwargs.get("sampling_params", {})

        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.server_addresses = None
        self.router = None
        self.init_router(server_addresses)

        logger.info(f"ThreadPoolExecutor size: {max_workers}.")
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(
                self.tokenizer,
                disable_thinking=kwargs.get("disable_thinking", False)
            )
        else:
            self.chat_parser = chat_parser

        self.episode = None
        self.stop = False

        train_time = time.localtime(time.time())
        self.train_id = time.strftime("%Y%m%d%H%M%S", train_time)
        self.iteration = None
        self.sample_id = None
        self.application_ids: dict[str, str] = {}

    def init_router(self, addresses):
        logger.info(f"addresses: {addresses}, router: {self.router}")
        if addresses is None or addresses == [None] or self.router is not None:
            return
        logger.info(f"create router, addresses: {addresses}")
        self.server_addresses = addresses
        from agentic_rl.runner.scheduler.router import Router
        self.router = Router.create(
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            tokenizer=self.tokenizer,
            addresses=self.server_addresses,
            model_name=self.sampling_params.get("model_name", {})
        )

    def init_episode(self, episode):
        self.episode = episode

    async def cancel_trajectories(self):
        self.stop = True
        await self.router.stop()

    def reset(self):
        self.router.reset()
        self.stop = False

    async def get_model_response(self, prompt, application_id, stream_queue=None, **kwargs):
        """
        Compute model response asynchronously based on the engine type.

        This function is multithread safe and routes the request to the appropriate
        engine-specific handler.

        Args:
            prompt: The input prompt to send to the model
            application_id: Unique identifier for the application
            **kwargs: Additional arguments to pass to the model

        Returns:
            The model's response text

        Raises:
            NotImplementedError: If the engine type is not supported
        """
        return await self._get_router_async(prompt, application_id, stream_queue=stream_queue, **kwargs)

    def update_envs_and_agents(self, envs, agents, iteration, sample_id):
        """
        Update the environments and agents.

        Args:
            iteration: iteration number
            sample_id: sample_id
            envs: List of environments to use
            agents: List of agents to use
        """
        if len(agents) != len(envs):
            raise ValueError(f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}")
        self.envs = envs
        for idx, env in enumerate(envs):
            env.idx = idx
        self.agents = agents
        self.n_parallel_agents = len(envs)
        self.iteration = iteration
        self.sample_id = sample_id

    def update_env_and_agent(self, task_id, env, agent, iteration, sample_id):
        """
        Update the environment and agent.

        Args:
            iteration: iteration number
            sample_id: sample_id
            env: a single of environments to use
            agent: a single of agents to use
        """
        env.idx = int(task_id)
        env.sample_id = sample_id
        self.env_dict[task_id] = env
        self.agent_dict[task_id] = agent
        self.iteration = iteration

    def release_env_and_agent(self, task_id):
        if task_id in self.env_dict.keys():
            self.env_dict.pop(task_id)
        if task_id in self.agent_dict.keys():
            self.agent_dict.pop(task_id)

    async def _get_router_async(self, prompt, application_id, stream_queue=None, **kwargs):
        prompt_text = prompt
        messages = prompt
        response = await self.router.chat(
            messages, application_id, self.sampling_params, stream_queue=stream_queue, **kwargs
        )
        return response

    def store_application_id(self, task, application_id):
        key = _generate_key(task)
        if key is None:
            return
        self.application_ids[key] = application_id

    def pop_application_id(self, task):
        key = _generate_key(task)
        if key is None:
            return None
        return self.application_ids.pop(key, None)

    def clear_cache(self):
        self.application_ids.clear()

    async def cancel_request(self, task):
        application_id = self.pop_application_id(task)
        if application_id is None:
            logger.warning(f"get application id for task {task} failed")
            return
        await self.router.cancel_request(application_id)

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, stream_queue=None, mode="Text", **kwargs):
        """Run a single agent's trajectory asynchronously"""
        agent = self.agent_dict[idx] if isinstance(idx, str) else self.agents[idx]
        env = self.env_dict[idx] if isinstance(idx, str) else self.envs[idx]
        env.application_id = application_id

        termination_reason = None
        response_token_len = 0
        response_tokens = []
        response_masks = []
        logprobs_list = []
        vllm_prompt_tokens = []
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0

        episode_steps = []

        llm_step_times = []
        env_step_times = []

        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps

        agent.reset()
        agent.update_from_env(
            observation=observation,
            reward=0.0,
            done=False,
            info=info,
        )
        messages = agent.chat_completions

        if len(messages) > 1:
            original_prompt = messages[1]["content"]
            hash_obj = hashlib.sha256(original_prompt.encode('utf-8'))
            id_32 = hash_obj.hexdigest()[:32]
        else:
            logger.warning(f"The user content is missing; only the system content is provided.")
            id_32 = DEFAULT_DATA_ID

        prompt_tokens, _ = convert_messages_to_tokens_and_masks(
            messages,
            tokenizer=self.tokenizer,
            parser=self.chat_parser,
            contains_first_msg=True,
            contains_generation_msg=True
        )
        prompt_token_len = len(prompt_tokens)
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(
                f"Trajectory {idx}: initial prompt length {prompt_token_len} "
                f"already exceeded max_prompt_length {self.max_prompt_length}, retrying"
            )

        step_idx_last = 0
        max_model_len = self.max_model_len
        max_tokens_old = self.sampling_params.get("max_tokens", 8192)
        for step_idx in range(self.max_steps):
            step_idx_last = step_idx
            if self.stop:
                logger.warning(f"trajectory canceled, appID:{application_id}, step_idx:{step_idx}")
                return None

            prompt_messages = agent.chat_completions.copy()

            if self.simplify_think_content:
                assistant_indices = [
                    i
                    for i, item in enumerate(prompt_messages)
                    if item.get("role") == "assistant"
                ]
                if assistant_indices:
                    for idx in assistant_indices:
                        content = '<think>' + prompt_messages[idx]["content"]
                        modified_content = re.sub(
                            r'\<think>.*?\</think>',
                            '<think>思考过程省略</think>',
                            content,
                            flags=re.DOTALL
                        )
                        prompt_messages[idx]["content"] = modified_content

            curr_step_prompt_length = len(self.tokenizer.encode(
                self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                add_special_tokens=False
            ))

            if not self.enforce_max_prompt_length:
                max_tokens = max_model_len - curr_step_prompt_length
                logger.info(
                    f"appID:{application_id}, step_idx:{step_idx}, max_model_len:{max_model_len},"
                    f" history_response_token_len:{response_token_len}, "
                    f"curr_step_prompt_length:{curr_step_prompt_length}, residual_max_tokens, {max_tokens}"
                )
            else:
                max_tokens = max_tokens_old

                prompt_str = self.chat_parser.parse(
                    prompt_messages,
                    add_generation_prompt=True,
                    is_first_msg=True
                )
                prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    break

                if prompt_len + max_tokens > max_model_len:
                    logger.warning("exit for exceed max model length error...")
                    termination_reason = "EXCEED_MODEL_LENGTH"
                    break

            kwargs["max_tokens"] = max_tokens
            kwargs["step_idx"] = step_idx

            start_time = time.time()

            prompt_for_vllm = prompt_tokens if step_idx == 0 else vllm_prompt_tokens + response_tokens
            try:
                http_response = await self.get_model_response(
                    prompt_for_vllm,
                    application_id,
                    stream_queue=stream_queue,
                    **kwargs
                )
            except Exception as exp:
                traceback.print_exc()
                logger.error(f"run trajectory failed, error: {exp}")
                logprobs_list = [0]
                prompt_id = application_id.split('-', 1)[0]
                trajectory: Trajectory = agent.trajectory
                token_result = {
                    "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                    "response_tokens": torch.tensor([1], dtype=torch.long),
                    "response_masks": torch.tensor([0], dtype=torch.long),
                    "logprobs": logprobs_list,
                    "trajectory_reward": 0,
                    "idx": env.idx,
                    "prompt_id": prompt_id,
                    "chat_completions": agent.chat_completions,
                    "trajectory": trajectory.to_info_dict(),
                    "metrics": {
                        "steps": 1,
                        "reward_time": 0,
                        "env_time": 0,
                        "llm_time": 0,
                        "total_time": 0,
                        "toolcall_reward": 0,
                        "res_reward": 0,
                        "env_step_times": 0,
                        "llm_step_times": 0
                    },
                }
                return token_result

            if step_idx == 0:
                vllm_prompt_tokens = http_response["prompt_tokens"]
                prompt_token_len = len(vllm_prompt_tokens)

            response = http_response["message"]
            if "!!!!!!" in response:
                logger.error(f"run trajectory failed, ========!!!!!, error: {response}")
                logprobs_list = [0]
                prompt_id = application_id.split('-', 1)[0]
                trajectory: Trajectory = agent.trajectory
                token_result = {
                    "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                    "response_tokens": torch.tensor([1], dtype=torch.long),
                    "response_masks": torch.tensor([0], dtype=torch.long),
                    "logprobs": logprobs_list,
                    "trajectory_reward": 0,
                    "idx": env.idx,
                    "prompt_id": prompt_id,
                    "chat_completions": agent.chat_completions,
                    "trajectory": trajectory.to_info_dict(),
                    "metrics": {
                        "steps": 1,
                        "reward_time": 0,
                        "env_time": 0,
                        "llm_time": 0,
                        "total_time": 0,
                        "toolcall_reward": 0,
                        "res_reward": 0,
                        "env_step_times": 0,
                        "llm_step_times": 0
                    },
                }
                return token_result

            vllm_response_tokens = http_response["response_tokens"]
            vllm_response_masks = [1] * len(vllm_response_tokens)

            single_turn_logprobs = http_response["logprobs"]

            if self.stop or response is None:
                logger.warning(f"trajectory canceled, appID:{application_id}, step_idx:{step_idx}")
                return None

            logger.info(f"kwargs: {kwargs}")
            delta_time = time.time() - start_time
            llm_step_times.append(delta_time)
            llm_time += delta_time
            total_time += delta_time
            logger.info(
                f"trajectory performance status, appID:{application_id}, "
                f"step_idx:{step_idx}, start_time:{strftime(start_time)}, "
                f"end_time:{strftime(start_time + delta_time)}, llm_time: {delta_time}"
            )
            app_stats.stat_vllm_step(application_id, step_idx, start_time, start_time + delta_time)

            prompt_response_pair = {
                "prompt": self.chat_parser.parse(
                    prompt_messages,
                    add_generation_prompt=True,
                    is_first_msg=True
                ),
                "response": response,
                "prompt_ids": http_response["prompt_tokens"],
                "completion_ids": http_response["response_tokens"],
                "logprobs": single_turn_logprobs,
            }
            episode_steps.append(prompt_response_pair)

            if stream_queue:
                stream_queue.put_nowait({
                    "event": "run_item_stream_event",
                    "data": {
                        "name": 'message_output_created',
                        "item": response,
                        "type": "run_item_stream_event"
                    }
                })

            action: Action = agent.update_from_model(response)
            action = action.action

            if stream_queue:
                stream_queue.put_nowait({
                    "event": "run_item_stream_event",
                    "data": {
                        "name": 'tool_called',
                        "item": str(action),
                        "type": "run_item_stream_event"
                    }
                })

            start_time = time.time()

            logger.info(f"call tool step: {step_idx} start ...")
            try:
                next_observation, reward, done, info = await asyncio.wait_for(
                    loop.run_in_executor(self.executor, env.step, action),
                    timeout=self.tool_timeout
                )
            except asyncio.TimeoutError:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(
                        f"Warning: Trajectory {application_id} completed due to: {termination_reason} "
                        f"before able to perform 1 complete action. "
                        f"This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n",
                        "red"
                    )
                reward = 0
                done = True
                info = {}
                next_observation = {
                    "tool_outputs": {
                        "tool_timeout_call_id": f"timeout for tool call: {action}"
                    }
                }

            logger.info(f"call tool step: {step_idx} end ...")

            if stream_queue:
                stream_queue.put_nowait({
                    "event": "run_item_stream_event",
                    "data": {
                        "name": 'tool_output',
                        "item": next_observation,
                        "type": "run_item_stream_event"
                    }
                })

            delta_time = time.time() - start_time
            env_step_times.append(delta_time)
            env_time += delta_time
            total_time += delta_time
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len
            logger.info(
                f"trajectory performance status, appID:{application_id}, "
                f"step_idx:{step_idx}, start_time:{strftime(start_time)}, "
                f"end_time:{strftime(start_time + delta_time)}, env_time: {delta_time}"
            )
            app_stats.stat_env_step(application_id, step_idx, start_time, start_time + delta_time, termination_reason)

            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )

            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)
            cur_step.step_id = step_idx

            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

            if assistant_message is None and mode == "Token":
                raise RuntimeError("Assistant messages is none when accumulating token trajectories which should be conversations. This should not happen.")
            if env_messages is None and mode == "Token":
                raise RuntimeError("Environment messages is none when accumulating token trajectories which should be conversations. This should not happen.")
            env_msg_tokens, env_msg_masks = [], []
            if env_messages:
                env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(
                    env_messages,
                    tokenizer=self.tokenizer,
                    parser=self.chat_parser,
                    contains_first_msg=False,
                    contains_generation_msg=True
                )

            logger.info(
                f"trajectory performance status, appID:{application_id}, step_idx:{step_idx}, "
                f"prompt_length:{curr_step_prompt_length}, "
                f"response_length:{len(vllm_response_tokens)}, env_length:{len(env_msg_tokens)}"
            )

            response_token_len += len(vllm_response_tokens) + len(env_msg_tokens)

            curr_step_prompt_length = len(self.tokenizer.encode(
                self.chat_parser.parse(
                    agent.chat_completions,
                    add_generation_prompt=True,
                    is_first_msg=True
                ),
                add_special_tokens=False
            ))
            trajectory_token_length = prompt_token_len + response_token_len
            if (not self.enforce_max_prompt_length and
                    (curr_step_prompt_length >= max_model_len or trajectory_token_length >= max_model_len)):
                truncation_length = max_model_len - trajectory_token_length
                single_turn_logprobs += [0] * len(env_msg_tokens)

                if truncation_length < 0:
                    truncated_response_tokens = (vllm_response_tokens + env_msg_tokens)[:truncation_length]
                    truncated_response_masks = (vllm_response_masks + env_msg_masks)[:truncation_length]
                    truncated_response_logprobs = single_turn_logprobs[:truncation_length]
                else:
                    truncated_response_tokens = vllm_response_tokens + env_msg_tokens
                    truncated_response_masks = vllm_response_masks + env_msg_masks
                    truncated_response_logprobs = single_turn_logprobs

                response_tokens.extend(truncated_response_tokens)
                response_masks.extend(truncated_response_masks)
                logprobs_list.extend(truncated_response_logprobs)

                cur_step = agent.get_current_state()
                if curr_step_prompt_length - len(env_msg_tokens) > max_model_len:
                    cur_step.reward = 0.0

                cur_step.done = True
                termination_reason = "TRUNCATION"
                break

            response_tokens.extend(vllm_response_tokens)
            response_masks.extend(vllm_response_masks)
            logprobs_list.extend(single_turn_logprobs)
            observation = next_observation

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            if done:
                termination_reason = "ENV_DONE"
                break

            response_tokens.extend(env_msg_tokens)
            response_masks.extend(env_msg_masks)
            logprobs_list.extend([0] * len(env_msg_tokens))

            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        app_stats.stat_env_state(application_id, step_idx_last, termination_reason)
        masked_out = False
        if self.overlong_filter:
            if (termination_reason == "TRUNCATION" or
                    termination_reason == "MAX_STEPS" or termination_reason == "TIMEOUT"):
                response_masks = [0] * len(response_masks)
                masked_out = True

        if termination_reason == "ENV_TIMEOUT":
            response_masks = [0] * len(response_masks)
            masked_out = True

        if hasattr(env, "compute_final_reward") and not masked_out:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            cur_step.reward = reward

        await loop.run_in_executor(self.executor, env.close)

        if termination_reason:
            if reward > 0:
                color = "green"
            else:
                color = "yellow"
            colorful_print(
                f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}. \n",
                color,
            )
            if masked_out:
                colorful_print(f"Trajectory {idx} is masked out due to overlong filter.", "red")

        trajectory: Trajectory = agent.trajectory
        trajectory.data_id = id_32
        trajectory.training_id = self.train_id
        trajectory.epoch_id = 0
        trajectory.iteration_id = self.iteration
        trajectory.sample_id = env.sample_id
        trajectory.application_id = application_id
        trajectory.trajectory_id = (
                trajectory.data_id + "-" + trajectory.training_id + "-" + str(trajectory.epoch_id)
                + "-" + str(trajectory.iteration_id) + "-" + str(trajectory.sample_id) + "-" + "0"
        )
        app_stats.stat_trajectory(application_id, trajectory.trajectory_id)
        trajectory.termination_reason = termination_reason if termination_reason is not None else ""

        self.compute_trajectory_reward_fn(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)

        if self.episode is not None:
            self.episode.set_termination_reason.remote(termination_reason)
            self.episode.add_trajectory.remote("aee", trajectory)
            if hasattr(env, "task"):
                self.episode.set_task.remote(env.task)

        prompt_id = application_id.split('-', 1)[0]
        trajectory.prompt_id = prompt_id
        logger.info(
            f"trajectory performance status, appID:{application_id}, total_llm_time:{llm_time}, "
            f"llm_step_times:{llm_step_times}, total_env_time:{env_time}, env_step_times:{env_step_times},"
            f" total_prompt_tokens:{len(prompt_tokens)}, total_response_tokens:{len(response_tokens)}"
        )
        trajectory.task = env.task

        if mode == "Text":
            return trajectory
        elif mode == "Token":
            prompt_tokens, response_tokens, response_masks, is_valid_trajectory = self.assemble_steps(episode_steps, masked_out)
            logger.info(f"tool call reward: {trajectory.toolcall_reward}")
            logger.info(f"res reward: {trajectory.res_reward}")
            logger.info(f"final reward: {trajectory.reward}")
            token_result = {
                "prompt_tokens": torch.tensor(vllm_prompt_tokens, dtype=torch.long),
                "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                "response_masks": torch.tensor(response_masks, dtype=torch.long),
                "logprobs": logprobs_list,
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "prompt_id": trajectory.prompt_id,
                "chat_completions": agent.chat_completions,
                "trajectory": trajectory.to_info_dict(),
                "metrics": {
                    # Total number of steps taken in the trajectory
                    "steps": len(trajectory.steps),
                    # Time to calculate reward
                    "reward_time": reward_time,
                    # Total time spent in environment execution (env.step)
                    "env_time": env_time,
                    # Time to calculate response tokens
                    "llm_time": llm_time,
                    # Total time spent in the trajectory
                    "total_time": total_time,
                    # Average reward for call tools within a traj
                    "toolcall_reward": trajectory.toolcall_reward,
                    # Result reward when a traj done
                    "res_reward": trajectory.res_reward,
                    # env step performance in the trajectory
                    "env_step_times": env_step_times,
                    # llm step performance in the trajectory
                    "llm_step_times": llm_step_times
                },
            }
            return token_result
        elif mode == "Conversation":
            return agent.chat_completions
        elif mode == "Step":
            steps_result = {
                "steps": episode_steps,
                "trajectory": trajectory.to_info_dict(),
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "mc_returns": [step.mc_return for step in trajectory.steps][: len(episode_steps)],
            }
            return steps_result

    async def trajectory_generator(self, task, stream_queue=None, reset_seed=0, mode="Text", **kwargs):
        if not all(env.is_multithread_safe() for env in self.env_dict.values()):
            raise TypeError("All environments must be multithread safe for async engine")

        async def launch_one_trajectory_task(task_id: str):
            try:
                prompt_id = kwargs['prompt_id'] if 'prompt_id' in kwargs else 0
                application_id = create_application_id(prompt_id)
                self.store_application_id(task, application_id)
                res = await self.run_agent_trajectory_async(
                    idx=task_id,
                    application_id=application_id,
                    seed=reset_seed,
                    mode=mode,
                    stream_queue=stream_queue,
                    **kwargs,
                )
            except Exception as exp:
                traceback.print_exc()
                logger.error(f"run trajectory failed, error: {exp}")
                raise exp
            return res

        tasks_to_run = [launch_one_trajectory_task(task['task_id'])]

        tasks_completed = 0
        for co in asyncio.as_completed(tasks_to_run):
            try:
                result = await co
                tasks_completed += 1
                yield result
            except Exception as e:
                logger.error(f"Exception {e}")
                raise e

    async def execute_tasks(self, tasks: list[dict]):
        """
        Run asynchronous interactions between the agent and environment where each agent
        has its own environment instance and can proceed independently.

        Args:
            tasks: List of tasks to process

        Returns:
            A list of trajectories, one for each task.
        """
        max_concurrent = self.n_parallel_agents

        all_trajectories = {}

        task_queue = list(enumerate(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
        for i in range(max_concurrent):
            index_queue.put_nowait(i)

        completed = 0
        total = len(tasks)

        async def sem_wrapper(task_id, task):
            nonlocal completed
            async with (semaphore):
                index = await index_queue.get()
                try:
                    self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
                    self.agents[index] = self.agent_class(**self.agent_args)
                    if not (self.agents[index] is not None and isinstance(self.agents[index], BaseAgent)):
                        raise TypeError("Agent is not initialized or not inheriting from BaseAgent")
                    self.agents[index].trajectory.task = task
                    res = await self.run_agent_trajectory_async(index, application_id=task_id)
                    res.task = task
                    completed += 1
                    colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
                    return task_id, res
                finally:
                    await index_queue.put(index)

        results = await asyncio.gather(*[sem_wrapper(task_id, task) for task_id, task in task_queue])

        all_trajectories = {task_id: trajectory for task_id, trajectory in results}
        ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]
        return ordered_trajectories

    def assemble_steps(self, steps: list[dict], masked_out: bool):
        """
        Transform step-by-step results into trajectory format for training.
        The assemble is aggresive, if steps is not cumulative, the response_masks is set to all 0s.

        Each step_result contains:
        - steps: List of {"prompt": str, "response": str, "prompt_ids": list, "completion_ids": list}

        For training, we need to assemble the full conversation sequence where:
        - prompt_tokens: Initial prompt (first step's prompt_ids)
        - response_tokens: All subsequent conversation (completion_ids + next step's prompt_ids)
        - response_masks: Mask indicating which tokens contribute to loss (only completion_ids)
        """
        initial_prompt_ids = steps[0]["prompt_ids"]
        accumulated_sequence = initial_prompt_ids.copy()
        response_tokens = []
        response_masks = []
        log_probs = []
        is_valid_trajectory = True

        for i, step in enumerate(steps):
            current_prompt_ids = step["prompt_ids"]
            current_completion_ids = step["completion_ids"]
            current_log_probs = step["logprobs"]

            if i == 0:
                response_tokens.extend(current_completion_ids)
                response_masks.extend([1] * len(current_completion_ids))
                log_probs.extend(current_log_probs)
                accumulated_sequence.extend(current_completion_ids)
            else:
                if current_prompt_ids[: len(accumulated_sequence)] != accumulated_sequence:
                    prefix = current_prompt_ids[: len(accumulated_sequence)]
                    diff_pos = None
                    for i, (expected, actual) in enumerate(zip(accumulated_sequence, prefix, strict=False)):
                        if expected != actual:
                            diff_pos = i
                            break

                    if diff_pos is not None:
                        logger.warning(
                            f"When assemble steps, detect the trajectory not accumulative at position "
                            f"{diff_pos}. Expected: {accumulated_sequence[diff_pos : diff_pos + 5]}, "
                            f"Got: {prefix[diff_pos : diff_pos + 5]}. Setting response_masks to all 0s. "
                            f"This is likely due to retokenization."
                        )
                    else:
                        logger.warning(
                            f"When assemble steps, detect length mismatch. Expected length: "
                            f"{len(accumulated_sequence)}, Got length: {len(prefix)}. "
                            f"Setting response_masks to all 0s."
                        )

                    is_valid_trajectory = False
                    if not is_valid_trajectory:
                        raise Exception("Detected invalid trajectory. Abort.")
                    break

                response_tokens.extend(
                    current_prompt_ids[len(accumulated_sequence) :] + current_completion_ids
                )
                response_masks.extend(
                    [0] * (len(current_prompt_ids) - len(accumulated_sequence))
                    + [1] * len(current_completion_ids)
                )
                log_probs.extend(
                    [0] * (len(current_prompt_ids) - len(accumulated_sequence))
                    + [1] * len(current_completion_ids)
                )
                accumulated_sequence = current_prompt_ids + current_completion_ids

        if len(response_masks) != len(response_tokens):
            raise Exception(f"response_masks length ({len(response_masks)}) does not match response_tokens length ({len(response_tokens)})")

        prompt_tokens = torch.tensor(initial_prompt_ids, dtype=torch.long)
        response_tokens = torch.tensor(response_tokens, dtype=torch.long)
        response_masks = torch.tensor(response_masks, dtype=torch.long)

        if masked_out:
            response_masks = [0] * len(response_masks)

        return prompt_tokens, response_tokens, response_masks, is_valid_trajectory


class AsyncAgentExecutionEngine(AgentExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)