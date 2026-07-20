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
import math
import os
import re
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch

from aura.base.log.loggers import Loggers
from aura.base.misc.misc import app_stats, colorful_print
from aura.base.utils.utils import strftime
from aura.runner.agent_engine_wrapper.base.agent.base_agent import Action, BaseAgent, Trajectory
from aura.runner.agent_engine_wrapper.base.environment.env_utils import compute_mc_return, compute_trajectory_reward
from aura.runner.agent_engine_wrapper.base.parser.chat_template import ChatTemplateParser
from aura.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask
from aura.runner.agent_engine_wrapper.rllm.msg_handler import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)

logger = Loggers(__name__).get_logger()

GLOBAL_INDEX = 0
DEFAULT_DATA_ID = "000000000000000000000000000000000"
FULL_REWARD_TOP_SCORE = 999.0
TERMINATION_MODEL_ERROR = "MODEL_ERROR"
TRAJECTORY_GENERATION_METHOD_DEFAULT = "default"
TRAJECTORY_GENERATION_METHOD_CHAIN = "chain"
TRAJECTORY_GENERATION_METHOD_TREE = "tree"
VALID_TRAJECTORY_GENERATION_METHODS = {
    TRAJECTORY_GENERATION_METHOD_DEFAULT,
    TRAJECTORY_GENERATION_METHOD_CHAIN,
    TRAJECTORY_GENERATION_METHOD_TREE,
}


@dataclass
class Beam:
    agent: Any = None
    env: Any = None
    beam_id: str | None = None
    parent_beam_id: str | None = None
    steps_length: int = 0
    response_tokens: list = field(default_factory=list)
    response_masks: list = field(default_factory=list)
    response_token_len: int = 0
    cum_reward: float = 0.0
    final_reward: float = 0.0
    done: bool = False
    termination_reason: str | None = None
    llm_time: float = 0.0
    env_time: float = 0.0
    total_time: float = 0.0
    collect_reward: bool = False
    is_last_step: bool = True
    step_depth: int = 0
    node_signature: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


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

def _recover_initial_user_message(agent, env, observation, info):
    """Rebuild the initial user message when memory only has system."""
    task = getattr(env, "task", None)
    if not isinstance(task, dict) and hasattr(task, "model_dump"):
        task = task.model_dump()
    task = dict(task) if isinstance(task, dict) else {}

    merged_obs = {}
    merged_obs.update(task)
    if isinstance(observation, dict):
        merged_obs.update(observation)
    elif observation:
        merged_obs["initial_observation"] = observation

    extra_args = merged_obs.pop("extra_args", None) or {}
    if isinstance(extra_args, dict):
        merged_obs.update(extra_args)
    if merged_obs.get("problem") and not merged_obs.get("question"):
        merged_obs["question"] = merged_obs["problem"]

    question = merged_obs.get("question") or merged_obs.get("query") or merged_obs.get("problem")
    if not question:
        return False

    formatter = getattr(agent, "_format_observation_as_messages", None)
    messages = formatter(merged_obs, info=info) if callable(formatter) else []
    if not messages:
        content = f"Question: {question}"
        root_url = merged_obs.get("root_url") or merged_obs.get("website")
        if root_url:
            content += f"\nURL: {root_url}"
        initial_observation = merged_obs.get("initial_observation")
        if initial_observation:
            content += f"\n\nObservation: {initial_observation}"
        messages = [{"role": "user", "content": content}]

    memory = getattr(agent, "memory", None)
    if memory is not None and hasattr(memory, "add_message"):
        memory.add_message(messages, metadata=[{"reward": 0.0}])
        logger.warning("Recovered missing initial user message from env.task/reset observation.")
        return True
    return False


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
        trajectory_generation_method="default",
        beam_size=3,
        per_beam_expand=2,
        max_model_len=16384,
        compute_trajectory_reward_fn=compute_trajectory_reward,
        tokenizer_name_or_path=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        env_args=None,
        max_workers=64,
        enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
        overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
        **kwargs,
    ):
        if agent_args is None:
            agent_args = {}
        if env_args is None:
            env_args = {}

        self.simplify_think_content = simplify_think_content
        if trajectory_generation_method not in VALID_TRAJECTORY_GENERATION_METHODS:
            raise ValueError(
                f"trajectory_generation_method must be one of {sorted(VALID_TRAJECTORY_GENERATION_METHODS)}, "
                f"got {trajectory_generation_method}"
            )
        self.trajectory_generation_method = trajectory_generation_method
        self.beam_size = beam_size
        self.per_beam_expand = per_beam_expand
        self.max_model_len = max_model_len

        self.tokenizer = tokenizer
        self.n_parallel_agents = n_parallel_agents
        self.overlong_filter = overlong_filter

        # For interaction
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
        self.token_in_token_out = kwargs.get("token_in_token_out", False)

        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.server_addresses = None
        self.router = None
        self.init_router(server_addresses)

        # Create a thread pool executor for environment interactions (i.e. step, reset, close)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer,
                                                             disable_thinking=kwargs.get("disable_thinking", False))
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
        logger.debug(f"addresses: {addresses}, router: {self.router}")
        if addresses is None or addresses == [None]:
            return
        if self.router is not None:
            self.router.update_address(addresses)
            logger.debug(f"router update_address, addresses: {addresses}")
            return
        logger.info(f"create router, addresses: {addresses} token_in_token_out={self.token_in_token_out}")
        self.server_addresses = addresses
        from aura.runner.scheduler.router import Router
        self.router = Router.create(
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            tokenizer=self.tokenizer,
            addresses=self.server_addresses,
            token_in_token_out=self.token_in_token_out,
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

    async def get_model_response(self, prompt, application_id, stream_queue=None, server_handles=None, **kwargs):
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
        if server_handles is not None:
            from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
            logger.info(f"use internal vLLM server handles: {server_handles}")
            server_manager = AsyncLLMServerManager({}, server_handles=server_handles)
            default_kwargs = dict(
                n=1,
                request_id=application_id,
            )

            merged_kwargs = {**default_kwargs, **self.sampling_params, **kwargs}
            logger.info(f"default simpling: {self.sampling_params}")
            logger.info(f"kwargs: {kwargs}")
            logger.info(f"default merged_kwargs: {merged_kwargs}")

            sample_params = {"n": merged_kwargs["n"],
                             "top_p": merged_kwargs["top_p"],
                             "temperature": merged_kwargs["temperature"],
                             "max_tokens": merged_kwargs["max_tokens"],
                             "logprobs": 1}

            token_output = await server_manager.generate(
                request_id=application_id, prompt_ids=prompt, sampling_params=sample_params)

            completion_text = self.tokenizer.decode(token_output.token_ids, skip_special_tokens=True)

            http_response = {
                "message": completion_text,
                "logprobs": token_output.log_probs,
                "response_tokens": token_output.token_ids,
                "prompt_tokens": prompt
            }

            return http_response
        else:
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
            raise ValueError(
                f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
            )
        self.envs = envs
        # For keeping track of the environment index in the batch.
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
        # If prompt is in chat format, convert it to text format
        prompt_text = prompt
        messages = prompt
        # if isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt):
        #     prompt_text = self.chat_parser.parse(prompt, add_generation_prompt=True, is_first_msg=True)
        response = await self.router.chat(
            messages, application_id, self.sampling_params, stream_queue=stream_queue, **kwargs)
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

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, stream_queue=None, mode="Text", server_handles=None, **kwargs):
        """Run a single agent's trajectory asynchronously"""
        agent = self.agent_dict[idx] if isinstance(idx, str) else self.agents[idx]
        env = self.env_dict[idx] if isinstance(idx, str) else self.envs[idx]
        env.application_id = application_id

        termination_reason = None
        response_token_len = 0
        response_tokens = []
        response_masks = []
        logprobs_list = []
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0

        # for step return
        episode_steps = []

        # for step perf
        llm_step_times = []
        env_step_times = []
        llm_step_input_lengths = []
        llm_step_output_lengths = []

        # Reset environment with the task using the executor
        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps

        # Reset agent
        agent.reset()
        # Update agent internal state from environment.
        agent.update_from_env(
            observation=observation,  # Raw observation from environment
            reward=0.0,
            done=False,
            info=info,
        )
        messages = agent.chat_completions
        if len(messages) <= 1 and _recover_initial_user_message(agent, env, observation, info):
            messages = agent.chat_completions

        # Use a hash of the initial user message as the original prompt id.
        if len(messages) > 1:
            original_prompt = messages[1]["content"]
            hash_obj = hashlib.sha256(original_prompt.encode('utf-8'))
            id_32 = hash_obj.hexdigest()[:32]
        else:
            logger.warning("The user content is missing; only the system content is provided.")
            id_32 = DEFAULT_DATA_ID

        prompt_tokens, _ = convert_messages_to_tokens_and_masks(messages, tokenizer=self.tokenizer,
                                                                parser=self.chat_parser, contains_first_msg=True,
                                                                contains_generation_msg=True)
        prompt_token_len = len(prompt_tokens)
        # Note, this should never happen!
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(
                f"Trajectory {idx}: initial prompt length {prompt_token_len} "
                f"already exceeded max_prompt_length {self.max_prompt_length}, retrying")

        step_idx_last = 0
        max_model_len = self.max_model_len
        max_tokens_old = self.sampling_params.get("max_tokens", 8192)
        for step_idx in range(self.max_steps):
            step_idx_last = step_idx
            if self.stop:
                logger.warning(f"trajectory canceled, appID:{application_id}, step_idx:{step_idx}")
                return None

            assistant_msg_tokens = []
            assistant_msg_masks = []
            single_turn_logprobs = []
            # Get action from agent
            prompt_messages = agent.chat_completions.copy()

            if self.simplify_think_content:
                assistant_indices = [i for i, item in enumerate(prompt_messages) if item.get("role") == "assistant"]
                if assistant_indices:
                    for ass_idx in assistant_indices:
                        content = '<think>' + prompt_messages[ass_idx]["content"]
                        modified_content = re.sub(r'\<think>.*?\</think>',
                                                  '<think>thinking omitted</think>', content,
                                                  flags=re.DOTALL)
                        prompt_messages[ass_idx]["content"] = modified_content
            # Max remaining tokens left for the response
            # For enforced max prompt at each step, no need to deduct here
            # curr step prompt_len
            curr_step_prompt_length = len(self.tokenizer.encode(
                self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                add_special_tokens=False))
            if not self.enforce_max_prompt_length:
                # add redundant design
                max_tokens = max_model_len - curr_step_prompt_length
                if max_tokens <= 0:
                    max_tokens = 128
                logger.info(
                    f"===appID:{application_id}, step_idx:{step_idx}, max_model_len:{max_model_len},"
                    f" history_response_token_len:{response_token_len}, "
                    f"curr_step_prompt_length:{curr_step_prompt_length}, residual_max_tokens, {max_tokens}")
            else:
                max_tokens = max_tokens_old

                # since max prompt is enforced, we filter out too long prompts.
                prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
                prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    break
                # handle exceed max model length error
                if prompt_len + max_tokens > max_model_len:
                    logger.warning("exit for exceed max model length error...")
                    termination_reason = "EXCEED_MODEL_LENGTH"
                    break
            kwargs["max_tokens"] = max_tokens
            kwargs["step_idx"] = step_idx

            start_time = time.time()
            if self.token_in_token_out:
                prompt_for_vllm = prompt_tokens if step_idx == 0 else prompt_tokens + response_tokens
                llm_step_input_lengths.append(len(prompt_for_vllm))
                model_response_error = False
                try:
                    http_response = await self.get_model_response(
                        prompt_for_vllm, application_id, stream_queue=stream_queue, server_handles=server_handles, **kwargs)
                except Exception as e:
                    logger.error(f"appId:{application_id} get_model_response exception! e:{e}")
                    return None
                if isinstance(http_response, str):
                    logger.error(
                        f"appId:{application_id} get_model_response returned an error string: {http_response}"
                    )
                    response = http_response
                    assistant_msg_tokens = self.tokenizer.encode(response, add_special_tokens=False)
                    assistant_msg_masks = [0] * len(assistant_msg_tokens)
                    single_turn_logprobs = [0.0] * len(assistant_msg_tokens)
                    termination_reason = TERMINATION_MODEL_ERROR
                    model_response_error = True
                    llm_step_output_lengths.append(len(assistant_msg_tokens))
                    if step_idx == 0:
                        prompt_token_len = len(prompt_tokens)
                else:
                    if step_idx == 0:
                        prompt_tokens = http_response["prompt_tokens"] or prompt_for_vllm
                        prompt_token_len = len(prompt_tokens)

                    response = http_response["message"]
                    assistant_msg_tokens = http_response["response_tokens"]
                    if assistant_msg_tokens is None:
                        assistant_msg_tokens = self.tokenizer.encode(response, add_special_tokens=False)
                    assistant_msg_masks = [1] * len(assistant_msg_tokens)

                    single_turn_logprobs = http_response["logprobs"]
                    if single_turn_logprobs is None:
                        single_turn_logprobs = [0.0] * len(assistant_msg_tokens)
                    llm_step_output_lengths.append(len(assistant_msg_tokens))
            else:
                response = await self.get_model_response(prompt_messages, application_id, stream_queue=stream_queue,
                                                         **kwargs)
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
                f"end_time:{strftime(start_time + delta_time)}, llm_time: {delta_time}")
            app_stats.stat_vllm_step(application_id, step_idx, start_time, start_time + delta_time)
            # Update steps
            prompt_response_pair = {
                "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                "response": response,
            }
            if self.token_in_token_out:
                if isinstance(http_response, str):
                    prompt_response_pair["prompt_ids"] = prompt_for_vllm
                    prompt_response_pair["completion_ids"] = assistant_msg_tokens
                else:
                    prompt_response_pair["prompt_ids"] = prompt_tokens if step_idx == 0 else prompt_for_vllm
                    prompt_response_pair["completion_ids"] = assistant_msg_tokens
                prompt_response_pair["logprobs"] = single_turn_logprobs
            episode_steps.append(prompt_response_pair)
            if self.token_in_token_out and model_response_error:
                response_tokens.extend(assistant_msg_tokens)
                response_masks.extend(assistant_msg_masks)
                logprobs_list.extend(single_turn_logprobs)
                response_token_len += len(assistant_msg_tokens)
                break
            if stream_queue:
                stream_queue.put_nowait({
                    "event": "run_item_stream_event",
                    "data": {
                        "name": 'message_output_created',
                        "item": response,
                        "type": "run_item_stream_event"
                    }
                })

            # Update agent with model response
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

            # Take step in environment using the executor
            start_time = time.time()

            logger.info(f"call tool step: {step_idx} start ...")
            try:
                next_observation, reward, done, info = await asyncio.wait_for(
                    loop.run_in_executor(self.executor, env.step, action),
                    timeout=self.tool_timeout)
            except asyncio.TimeoutError:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(
                        f"Warning: Trajectory {application_id} completed due to: {termination_reason} "
                        f"before able to perform 1 complete action. "
                        f"This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n",
                        "red")
                reward = 0
                done = True
                info = {}
                next_observation = {"tool_outputs": {"tool_timeout_call_id": f"timeout for tool call: {action}"}}

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
                f"end_time:{strftime(start_time + delta_time)}, env_time: {delta_time}")
            app_stats.stat_env_step(application_id, step_idx, start_time, start_time + delta_time, termination_reason)

            # Update agent internal state.
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
            # info: assistant_message is dict, env_messages is list(dict),
            # info: each env_messages list contains one dict, for example {"role": "tool", "content": "..."}.
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

            # Check and convert to tokens if necessary
            if assistant_message is None and mode == "Token":
                raise RuntimeError(
                    "Assistant messages is none when accumulating token trajectories "
                    "which should be conversations. This should not happen."
                )
            if env_messages is None and mode == "Token":
                raise RuntimeError(
                    "Environment messages is none when accumulating token trajectories "
                    "which should be conversations. This should not happen."
                )
            env_msg_tokens, env_msg_masks = [], []
            if assistant_message and not self.token_in_token_out:
                assistant_msg_tokens, assistant_msg_masks = (
                    convert_messages_to_tokens_and_masks(
                        [assistant_message], tokenizer=self.tokenizer, parser=self.chat_parser,
                        contains_first_msg=False, contains_generation_msg=False))
            if env_messages:
                env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(
                    env_messages, tokenizer=self.tokenizer, parser=self.chat_parser,
                    contains_first_msg=False, contains_generation_msg=True)

            logger.info(f"trajectory performance status, appID:{application_id}, step_idx:{step_idx}, "
                        f"prompt_length:{curr_step_prompt_length}, "
                        f"response_length:{len(assistant_msg_tokens)}, env_length:{len(env_msg_tokens)}")
            # Update response token length
            response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
            # Reached maximum number of tokens for the trajectory
            curr_step_prompt_length = len(self.tokenizer.encode(
                self.chat_parser.parse(agent.chat_completions, add_generation_prompt=True, is_first_msg=True),
                add_special_tokens=False))
            logger.info(f"trajectory performance status, prompt truncation judge, "
                        f"appID:{application_id}, step_idx:{step_idx}, "
                        f"current step_prompt_length: {curr_step_prompt_length}, "
                        f"max_model_len: {max_model_len}")
            if not self.enforce_max_prompt_length and curr_step_prompt_length >= max_model_len:
                logger.info(f"exceed max_model_len, current step_prompt_length: {curr_step_prompt_length}, "
                            f"max_model_len: {max_model_len}")
                # Truncation length
                truncation_length = max_model_len - curr_step_prompt_length
                # Truncate the response and masks
                if truncation_length < 0:
                    truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                    truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                    truncated_response_logprobs = single_turn_logprobs[:truncation_length]
                else:
                    # Edge case where the response is exactly the max response length.
                    truncated_response_tokens = assistant_msg_tokens + env_msg_tokens
                    truncated_response_masks = assistant_msg_masks + env_msg_masks
                    truncated_response_logprobs = single_turn_logprobs
                # Update token collections
                response_tokens.extend(truncated_response_tokens)
                response_masks.extend(truncated_response_masks)
                logprobs_list.extend(truncated_response_logprobs)

                logger.info(f"===========truncated response tokens, appID:{application_id}, step_idx:{step_idx}, "
                            f"truncation_length:{truncation_length}, "
                            f"truncated_response_tokens len: {len(truncated_response_tokens)}, "
                            f"response_tokens len: {len(response_tokens)}")

                cur_step = agent.get_current_state()
                if curr_step_prompt_length - len(env_msg_tokens) > max_model_len:
                    cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "TRUNCATION"
                break

            # Update the token version of trajectory
            response_tokens.extend(assistant_msg_tokens)
            response_masks.extend(assistant_msg_masks)
            logprobs_list.extend(single_turn_logprobs)
            observation = next_observation

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            # Check if episode is done
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
        # info: self.overlong_filter=false
        if self.overlong_filter:
            if (termination_reason == "TRUNCATION" or
                    termination_reason == "MAX_STEPS" or termination_reason == "TIMEOUT"):
                # Mask out the entire response for overlong trajectories if the reward is 0.
                response_masks = [0] * len(response_masks)
                masked_out = True

        # add by ts: env timeout, mask out
        if termination_reason == "ENV_TIMEOUT":
            response_masks = [0] * len(response_masks)
            masked_out = True

        # info: hasattr(env, "compute_final_reward")=False
        if hasattr(env, "compute_final_reward") and not masked_out:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            cur_step.reward = reward
        # Closing environment using the executor.
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
        trajectory.trajectory_id = (trajectory.data_id + "-" + trajectory.training_id + "-" + str(trajectory.epoch_id)
                                    + "-" + str(trajectory.iteration_id) + "-" + str(trajectory.sample_id) + "-" + "0")
        app_stats.stat_trajectory(application_id, trajectory.trajectory_id)
        trajectory.termination_reason = termination_reason if termination_reason is not None else ""
        # Tag generation method so compute_trajectory_reward can dispatch chain-mode
        # (0/1 source_url reward) without affecting tree mode (handled separately).
        trajectory.trajectory_generation_method = self.trajectory_generation_method
        # Aggregate final trajectory statistics
        self.compute_trajectory_reward_fn(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)

        if self.episode is not None:
            self.episode.set_termination_reason.remote(termination_reason)
            self.episode.add_trajectory.remote("aee", trajectory)
            if hasattr(env, "task"):
                self.episode.set_task.remote(env.task)

        prompt_id = application_id.split('-', 1)[0]
        trajectory.prompt_id = prompt_id
        logger.info(f"trajectory performance status, appID:{application_id}, "
                    f"total_llm_time:{llm_time}, llm_step_times:{llm_step_times}, "
                    f"llm_step_input_lengths: {llm_step_input_lengths}, llm_step_output_lengths: {llm_step_output_lengths}, "
                    f"total_env_time:{env_time}, env_step_times:{env_step_times}, "
                    f" total_prompt_tokens:{len(prompt_tokens)}, total_response_tokens:{len(response_tokens)}")
        trajectory.task = env.task
        if mode == "Text":
            return trajectory
        elif mode == "Token":
            logger.info(f"trajectory reward, appID:{application_id} tool call reward: {trajectory.toolcall_reward} res reward: {trajectory.res_reward} final reward: {trajectory.reward}")
            token_result = {
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                "response_masks": torch.tensor(response_masks, dtype=torch.long),
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
            if self.token_in_token_out:
                token_result["logprobs"] = torch.tensor(logprobs_list)
            return token_result
        elif mode == "Conversation":
            return agent.chat_completions
        elif mode == "Step":
            steps_result = {
                "steps": episode_steps,
                "trajectory": trajectory.to_info_dict(),
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "prompt_id": trajectory.prompt_id,
                "mc_returns": [step.mc_return for step in trajectory.steps][: len(episode_steps)],
                "metrics": {
                    "steps": len(trajectory.steps),
                    "reward_time": reward_time,
                    "env_time": env_time,
                    "llm_time": llm_time,
                    "total_time": total_time,
                    "toolcall_reward": trajectory.toolcall_reward,
                    "res_reward": trajectory.res_reward,
                    "env_step_times": env_step_times,
                    "llm_step_times": llm_step_times
                },
            }
            return steps_result

    def _messages_to_prompt_tokens(self, messages):
        prompt_text = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
        return self.tokenizer.encode(prompt_text, add_special_tokens=False)

    def _validate_tree_snapshot_support(self, agent, env):
        missing = [
            name
            for name, obj in (("agent", agent), ("env", env))
            if not callable(getattr(obj, "snapshot", None))
        ]
        if missing:
            raise TypeError(f"{' and '.join(missing)} must implement snapshot() for tree mode")

    async def _beam_model_response(self, agent, application_id, stream_queue=None, **kwargs):
        prompt_messages = agent.chat_completions.copy()
        prompt = self._messages_to_prompt_tokens(prompt_messages) if self.token_in_token_out else prompt_messages
        http_response = await self.get_model_response(
            prompt,
            application_id,
            stream_queue=stream_queue,
            **kwargs,
        )
        if isinstance(http_response, str):
            return http_response
        return http_response["message"]

    async def run_episode_with_dynamic_beam_search(
        self,
        agent,
        env,
        application_id: str,
        kwargs: dict,
        *,
        beam_size: int,
        per_beam_expand: int,
        score_fn: str = "sum_reward",
        mode: str = "Step",
        stream_queue=None,
    ):
        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps

        agent.reset()
        agent.update_from_env(observation=observation, reward=0.0, done=False, info=info)
        if len(agent.chat_completions) <= 1:
            _recover_initial_user_message(agent, env, observation, info)

        self._validate_tree_snapshot_support(agent, env)
        root_beam = Beam(agent=agent.snapshot(), env=env.snapshot(), beam_id=str(uuid.uuid1()))
        beams = [root_beam]
        all_candidates = []
        target_num = max(1, beam_size * per_beam_expand)

        def score_beam(beam: Beam) -> float:
            if score_fn == "avg_reward":
                return beam.cum_reward / max(1, beam.steps_length)
            if beam.steps_length > 0 and math.isclose(beam.cum_reward, float(beam.steps_length)):
                # Every explored step reached the maximum reward, so keep it at the front.
                return FULL_REWARD_TOP_SCORE
            return beam.cum_reward

        def beam_metadata_bool(beam: Beam, key: str) -> bool:
            return bool(beam.metadata.get(key, False))

        for step_idx in range(self.max_steps):
            active_beams = [beam for beam in beams if not beam.done]
            if not active_beams:
                break

            expand_tasks = []
            n_active = len(active_beams)
            base_expand, extra_expand = divmod(target_num, n_active)
            for beam_idx, beam in enumerate(active_beams):
                expand_count = base_expand + (1 if beam_idx < extra_expand else 0)
                if expand_count == 0:
                    continue
                prompt_len = len(self._messages_to_prompt_tokens(beam.agent.chat_completions))
                max_tokens = self.sampling_params.get("max_tokens", 8192)
                if not self.enforce_max_prompt_length:
                    max_tokens = max(128, self.max_model_len - prompt_len)
                elif prompt_len > self.max_prompt_length:
                    beam.done = True
                    beam.termination_reason = "PROMPT_TRUNCATION"
                    beam.collect_reward = True
                    continue

                for _ in range(expand_count):
                    gen_kwargs = dict(kwargs)
                    gen_kwargs["max_tokens"] = max_tokens
                    gen_kwargs["step_idx"] = step_idx

                    async def one_candidate(parent_beam: Beam, gen_kwargs_local: dict):
                        temp_agent = parent_beam.agent.snapshot()
                        temp_env = parent_beam.env.snapshot()

                        start_llm = time.time()
                        response = await self._beam_model_response(
                            temp_agent,
                            application_id,
                            stream_queue=stream_queue,
                            **gen_kwargs_local,
                        )
                        delta_llm = time.time() - start_llm

                        action: Action = temp_agent.update_from_model(response)
                        action = action.action

                        remain_timeout = max(0.0, self.trajectory_timeout - parent_beam.total_time - delta_llm)
                        try:
                            start_env = time.time()
                            next_obs, reward, done, info = await asyncio.wait_for(
                                loop.run_in_executor(self.executor, temp_env.step, action),
                                timeout=remain_timeout,
                            )
                            delta_env = time.time() - start_env
                            term = None
                        except asyncio.TimeoutError:
                            next_obs, reward, done, info = {"tool_outputs": {}}, 0.0, True, {}
                            delta_env = 0.0
                            term = "ENV_TIMEOUT"

                        info["max_steps"] = self.max_steps
                        info["cur_tokens"] = parent_beam.response_token_len
                        temp_agent.update_from_env(observation=next_obs, reward=reward, done=done, info=info)

                        cur_step = temp_agent.get_current_state()
                        cur_step.reward = reward
                        cur_step.done = done
                        cur_step.info.update(info)
                        cur_step.step_id = step_idx

                        child = Beam(
                            agent=temp_agent,
                            env=temp_env,
                            beam_id=str(uuid.uuid1()),
                            parent_beam_id=parent_beam.beam_id,
                            steps_length=parent_beam.steps_length + 1,
                            response_tokens=list(parent_beam.response_tokens),
                            response_masks=list(parent_beam.response_masks),
                            response_token_len=parent_beam.response_token_len,
                            cum_reward=parent_beam.cum_reward + float(reward),
                            final_reward=float(reward),
                            llm_time=parent_beam.llm_time + delta_llm,
                            env_time=parent_beam.env_time + delta_env,
                            total_time=parent_beam.total_time + delta_llm + delta_env,
                            collect_reward=False,
                            is_last_step=True,
                            step_depth=step_idx,
                            node_signature=parent_beam.node_signature,
                            metadata=dict(parent_beam.metadata),
                        )

                        metadata = info.get("metadata", {}) if isinstance(info, dict) else {}
                        child.metadata.update(
                            {
                                "on_golden_path": bool(metadata.get("on_golden_path", False)),
                                "golden_path_configured": bool(metadata.get("golden_path_configured", False)),
                                "clicked_button": str(metadata.get("clicked_button", "") or ""),
                                "reached_golden_path_end": bool(metadata.get("reached_golden_path_end", False)),
                            }
                        )

                        if beam_metadata_bool(child, "reached_golden_path_end"):
                            done = True
                            child.termination_reason = "GOLDEN_PATH_COMPLETE"

                        if done:
                            child.done = True
                            child.collect_reward = True
                            child.termination_reason = child.termination_reason or term or "ENV_DONE"
                        elif step_idx == self.max_steps - 1:
                            child.done = True
                            child.collect_reward = True
                            child.termination_reason = "MAX_STEPS"

                        return child

                    expand_tasks.append(one_candidate(beam, gen_kwargs))

            if not expand_tasks:
                break

            new_children = await asyncio.gather(*expand_tasks)
            all_candidates.extend(new_children)
            candidates = sorted(new_children, key=score_beam, reverse=True)

            active_candidates = [beam for beam in candidates if not beam.done]
            done_candidates = [beam for beam in candidates if beam.done]
            if active_candidates and any(beam_metadata_bool(beam, "golden_path_configured") for beam in active_candidates):
                golden_beams = [beam for beam in active_candidates if beam_metadata_bool(beam, "on_golden_path")]
                deviated_beams = [beam for beam in active_candidates if not beam_metadata_bool(beam, "on_golden_path")]
                if golden_beams:
                    for beam in deviated_beams:
                        beam.done = True
                        beam.collect_reward = True
                        beam.termination_reason = "GOLDEN_PATH_DEVIATED"
                    # Dynamic expansion: keep ALL nodes that hit the
                    # golden path into next layer, do NOT truncate to beam_size.
                    beams = done_candidates + golden_beams
                    logger.info(
                        f"[beam_golden_select] step={step_idx} "
                        f"kept_golden={len(golden_beams)} stopped_deviated={len(deviated_beams)}"
                    )
                else:
                    for beam in active_candidates:
                        beam.done = True
                        beam.collect_reward = True
                        beam.termination_reason = beam.termination_reason or "GOLDEN_PATH_STOP_NO_MATCH"
                    beams = done_candidates + active_candidates
                    break
            else:
                # No golden-path task: keep all beams sharing the top score,
                # rather than a fixed beam_size cut.
                if active_candidates:
                    top_score = score_beam(active_candidates[0])
                    top_beams = [b for b in active_candidates if math.isclose(score_beam(b), top_score)]
                else:
                    top_beams = []
                beams = done_candidates + top_beams

            for beam in beams:
                if not beam.done:
                    beam.is_last_step = False
            if all(beam.done for beam in beams):
                break

        for beam in all_candidates:
            if beam.done and beam.termination_reason is None:
                beam.termination_reason = "FINISHED"
        return all_candidates or beams

    async def convert_beam_to_trajectory(self, beam_path: Beam, idx, mode="Step", trajectory_generation_method=None):
        loop = asyncio.get_event_loop()
        env = beam_path.env
        agent = beam_path.agent
        reward = beam_path.final_reward
        termination_reason = beam_path.termination_reason or "FINISHED"
        reward_time = None

        await loop.run_in_executor(self.executor, env.close)
        trajectory: Trajectory = agent.trajectory
        messages = agent.chat_completions
        if isinstance(messages, list) and len(messages) > 1 and isinstance(messages[1], dict):
            original_prompt = messages[1].get("content", "")
            id_32 = hashlib.sha256(original_prompt.encode("utf-8")).hexdigest()[:32]
        else:
            id_32 = DEFAULT_DATA_ID

        trajectory.data_id = id_32
        trajectory.training_id = self.train_id
        trajectory.epoch_id = 0
        trajectory.iteration_id = self.iteration
        prompt_id = (
            getattr(env, "prompt_id", None)
            or (env.task.get("prompt_id") if isinstance(getattr(env, "task", None), dict) else None)
            or 0
        )
        trajectory.sample_id = str(idx)
        trajectory.application_id = f"{prompt_id}-{beam_path.beam_id}"
        trajectory.trajectory_id = (
            f"{trajectory.data_id}-{trajectory.training_id}-{trajectory.epoch_id}-"
            f"{trajectory.iteration_id}-{trajectory.sample_id}-{idx}"
        )
        trajectory.termination_reason = termination_reason
        trajectory.task = env.task
        trajectory.prompt_id = str(prompt_id)
        trajectory.prompt_index = getattr(env, "idx", idx)
        trajectory.idx = getattr(env, "idx", idx)
        trajectory.trajectory_generation_method = trajectory_generation_method or self.trajectory_generation_method
        trajectory.is_last_step = beam_path.is_last_step
        trajectory.step_depth = beam_path.step_depth
        trajectory.collect_reward = beam_path.collect_reward

        self.compute_trajectory_reward_fn(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)

        if mode == "Text":
            return trajectory

        steps = []
        for step in trajectory.steps:
            chat_completions = step.chat_completions or []
            if len(chat_completions) >= 1 and chat_completions[-1].get("role") == "assistant":
                prompt_messages = chat_completions[:-1]
                response_text = chat_completions[-1].get("content", "")
            else:
                prompt_messages = chat_completions
                response_text = step.model_response
            prompt_text = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
            steps.append({
                "prompt": prompt_text,
                "response": response_text,
            })

        return {
            "steps": steps,
            "trajectory": trajectory.to_info_dict(),
            "trajectory_reward": trajectory.reward,
            "idx": trajectory.idx,
            "prompt_id": trajectory.prompt_id,
            "prompt_index": trajectory.prompt_index,
            "step_depth": trajectory.step_depth,
            "is_last_step": trajectory.is_last_step,
            "collect_reward": trajectory.collect_reward,
            "mc_returns": [step.mc_return for step in trajectory.steps][:len(steps)],
            "metrics": {
                "steps": len(trajectory.steps),
                "reward_time": reward_time,
                "env_time": beam_path.env_time,
                "llm_time": beam_path.llm_time,
                "total_time": beam_path.total_time,
                "toolcall_reward": trajectory.toolcall_reward,
                "res_reward": trajectory.res_reward,
            },
        }

    async def run_agent_trajectory_async_tree(
        self,
        idx,
        application_id,
        seed=0,
        stream_queue=None,
        mode="Step",
        server_handles=None,
        trajectory_generation_method=TRAJECTORY_GENERATION_METHOD_TREE,
        **kwargs,
    ):
        agent = self.agent_dict[idx] if isinstance(idx, str) else self.agents[idx]
        env = self.env_dict[idx] if isinstance(idx, str) else self.envs[idx]
        env.application_id = application_id

        beams_path = await self.run_episode_with_dynamic_beam_search(
            agent,
            env,
            application_id,
            kwargs,
            beam_size=self.beam_size,
            per_beam_expand=self.per_beam_expand,
            score_fn="sum_reward",
            mode=mode,
            stream_queue=stream_queue,
        )

        results = [
            self.convert_beam_to_trajectory(
                path,
                idx=f"{idx}_{path_idx}",
                mode=mode,
                trajectory_generation_method=trajectory_generation_method,
            )
            for path_idx, path in enumerate(beams_path)
        ]
        return await asyncio.gather(*results, return_exceptions=False)

    async def trajectory_generator(self, task, stream_queue=None, reset_seed=0, mode="Text", server_handles=None, **kwargs):
        if not all(env.is_multithread_safe() for env in self.env_dict.values()):
            raise TypeError("All environments must be multithread safe for async engine")

        async def launch_one_trajectory_task(task_id: str):
            try:
                prompt_id = kwargs['prompt_id'] if 'prompt_id' in kwargs else 0
                application_id = create_application_id(prompt_id)
                self.store_application_id(task, application_id)
                generation_method = (
                    task.get("trajectory_generation_method")
                    if isinstance(task, dict)
                    else None
                ) or self.trajectory_generation_method
                if generation_method == TRAJECTORY_GENERATION_METHOD_TREE:
                    res = await self.run_agent_trajectory_async_tree(
                        idx=task_id,
                        application_id=application_id,
                        seed=reset_seed,
                        mode=mode,
                        stream_queue=stream_queue,
                        server_handles=server_handles,
                        trajectory_generation_method=generation_method,
                        **kwargs,
                    )
                else:
                    res = await self.run_agent_trajectory_async(
                        idx=task_id,
                        application_id=application_id,
                        seed=reset_seed,
                        mode=mode,
                        stream_queue=stream_queue,
                        server_handles=server_handles,
                        **kwargs,
                    )
            except Exception as exp:
                import traceback
                traceback.print_exc()
                logger.error(f"run trajectory failed, error: {exp}")
                raise exp
            return res

        # Create all N conceptual tasks. Their execution will be throttled by the semaphore
        # and the availability of agent/env indices.
        # One idx corresponds to one agent task.
        tasks_to_run = [launch_one_trajectory_task(task['task_id'])]

        tasks_completed = 0
        for co in asyncio.as_completed(tasks_to_run):
            try:
                result = await co
                tasks_completed += 1
                if isinstance(result, list):
                    for item in result:
                        yield item
                else:
                    yield result
                # Yield each trajectory result as soon as it completes.
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

        # Initialize results list to store trajectories for all tasks
        all_trajectories = {}

        # Create a queue of tasks to process
        task_queue = list(enumerate(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
        for i in range(max_concurrent):
            index_queue.put_nowait(i)

        # Track completed trajectories
        completed = 0
        total = len(tasks)

        async def sem_wrapper(task_id, task):
            nonlocal completed
            async with (semaphore):
                # Get an available index
                index = await index_queue.get()
                try:
                    self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
                    self.agents[index] = self.agent_class(**self.agent_args)
                    assert self.agents[index] is not None and isinstance(self.agents[index], BaseAgent), (
                        "Agent is not initialized or not inheriting from BaseAgent"
                    )
                    self.agents[index].trajectory.task = task  # type: ignore
                    res = await self.run_agent_trajectory_async(index, application_id=task_id)
                    res.task = task
                    completed += 1
                    colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
                    return task_id, res
                finally:
                    # Put the index back in the queue when done
                    await index_queue.put(index)

        # Run all tasks concurrently
        results = await asyncio.gather(*[sem_wrapper(task_id, task) for task_id, task in task_queue])

        all_trajectories = {task_id: trajectory for task_id, trajectory in results}
        ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]
        return ordered_trajectories


class AsyncAgentExecutionEngine(AgentExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
