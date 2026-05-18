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
import hashlib
import time

import torch

from aura.base.misc.misc import colorful_print
from aura.runner.agent_engine_wrapper.base.agent.base_agent import Action, Trajectory
from aura.runner.agent_engine_wrapper.base.environment.env_utils import compute_mc_return

from aura.runner.agent_engine_wrapper.rllm.msg_handler import parse_whole_response_token_ids
from aura.runner.agent_engine_wrapper.base.parser.chat_template import ChatTemplateParser
from aura.runner.agent_engine_wrapper.rllm.agent_execution_engine import AgentExecutionEngine, DEFAULT_DATA_ID
from aura.runner.agent_engine_wrapper.proxy_client.agent_proxy_client import AgentProxyClient
from aura.runner.agent_engine_wrapper.proxy_client.traj_proxy_client import TrajProxyClient

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class ExternAgentWrapper(AgentExecutionEngine):
    def __init__(
        self,
        tokenizer=None,
        server_addresses=None,
        chat_parser=None,
        n_parallel_agents=1,
        gamma=0.2,
        api_retries=3,  # to delete
        retry_limit=3,
        max_steps=5,
        max_prompt_length=1024,
        simplify_think_content=False,
        max_model_len=16384,
        compute_trajectory_reward_fn=None,
        tokenizer_name_or_path=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        env_args=None,
        max_workers=64,
        enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
        overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
        trajectory_timeout=None,
        max_response_length=8192,
        agent_proxy_url="",
        traj_proxy_url="",
        traj_proxy_run_id="",
        agent_engine_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            server_addresses=server_addresses,
            chat_parser=chat_parser,
            n_parallel_agents=n_parallel_agents,
            compute_trajectory_reward_fn=compute_trajectory_reward_fn,
            max_workers=max_workers,
            gamma=gamma,
            max_steps=max_steps,
            max_prompt_length=max_prompt_length,
            simplify_think_content=simplify_think_content,
            max_model_len=max_model_len,
            tokenizer_name_or_path=tokenizer_name_or_path,
            enforce_max_prompt_length=enforce_max_prompt_length,
            overlong_filter=overlong_filter,
            trajectory_timeout=trajectory_timeout,
            max_response_length=max_response_length,
            agent_args=agent_args,
            **kwargs,
        )

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(
                self.tokenizer,
                disable_thinking=kwargs.get("disable_thinking", False),
                toolcall_parser=agent_args.get("toolcall_parser", "qwen3_coder"),
            )
        else:
            self.chat_parser = chat_parser
        model_name = self.sampling_params.get("model", "")
        params = {}
        params["infer_params"] = self.sampling_params
        params["extra_params"] = agent_engine_kwargs
        agent_engine_kwargs["max_model_len"] = max_model_len
        agent_proxy_kwargs = agent_engine_kwargs.get("agent_proxy_args", {})
        agent_proxy_kwargs["model_name"] = model_name
        agent_proxy_kwargs["params"] = params
        agent_proxy_kwargs["timeout"] = self.trajectory_timeout
        agent_proxy_kwargs["max_retries"] = retry_limit
        agent_proxy_kwargs["agent_addr"] = agent_proxy_url
        agent_proxy_kwargs["traj_addr"] = traj_proxy_url
        agent_proxy_kwargs["run_id"] = traj_proxy_run_id
        self.agent_proxy_client = AgentProxyClient(**agent_proxy_kwargs)

        traj_proxy_kwargs = agent_engine_kwargs.get("traj_proxy_args", {})
        traj_proxy_kwargs["model_name"] = model_name
        traj_proxy_kwargs["infer_url"] = traj_proxy_url
        self.traj_proxy_client = TrajProxyClient(**traj_proxy_kwargs)

        # For step perf tracking
        self.llm_step_times = []
        self.env_step_times = []

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, stream_queue=None, mode="Text", **kwargs):
        """Run a single agent's trajectory asynchronously"""
        agent = self.agent_dict[idx] if isinstance(idx, str) else self.agents[idx]
        env = self.env_dict[idx] if isinstance(idx, str) else self.envs[idx]
        env.application_id = application_id

        tools = None
        termination_reason = None
        response_token_len = 0
        response_tokens = []
        response_masks = []
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0

        # Step performance tracking
        llm_step_times = []
        env_step_times = []

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

        # Get the original prompt_id using a hash
        if len(messages) > 1:
            original_prompt = messages[1]["content"]
            hash_obj = hashlib.sha256(original_prompt.encode('utf-8'))
            id_32 = hash_obj.hexdigest()[:32]
        else:
            logger.warning("The user content is missing; only the system content is provided.")
            id_32 = DEFAULT_DATA_ID

        prompt_tokens = self.tokenizer.encode(
            self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True), add_special_tokens=False
        )
        prompt_token_len = len(prompt_tokens)

        # Note, this should never happen!
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(
                f"Trajectory {idx}: initial prompt length {prompt_token_len} "
                f"already exceeded max_prompt_length {self.max_prompt_length}, retrying"
            )

        max_tokens_old = self.sampling_params.get("max_tokens", 8192)
        max_model_len = self.max_model_len

        ret, session_id = await self.agent_proxy_client.get_agent_response(messages, application_id, env.sample_id)
        if ret != 0:
            raise Exception(f"proxy_worker get_agent_response failed for application_id: {application_id}")

        # Get trajectory response from proxy worker (async call)
        traj_res = await self.traj_proxy_client.get_agent_trajectory(session_id)
        if not traj_res:
            logger.error(f"Failed to get trajectory for application_id: {application_id}")
            return None

        # Parse and sort step information by step_id
        step_info_list = traj_res.get("step_info", [])
        step_info_list = sorted(step_info_list, key=lambda x: x.get("step_id", 0))

        # Process each step in the trajectory
        for step_idx in range(min(self.max_steps, len(step_info_list))):
            step_info = step_info_list[step_idx]
            tools = step_info.get("tools", None)
            # Max remaining tokens left for the response
            # For enforced max prompt at each step, no need to deduct here
            # curr step prompt_len
            prompt_messages = agent.chat_completions
            curr_step_prompt_length = len(
                self.tokenizer.encode(
                    self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True, tools=tools),
                    add_special_tokens=False,
                )
            )
            if not self.enforce_max_prompt_length:
                # add redundant design
                max_tokens = max_model_len - curr_step_prompt_length
                logger.info(
                    f"===appID:{application_id}, step_idx:{step_idx}, max_model_len:{max_model_len},"
                    f" history_response_token_len:{response_token_len}, "
                    f"curr_step_prompt_length:{curr_step_prompt_length}, residual_max_tokens, {max_tokens}"
                )
            else:
                # TODO: never rechead but bugs
                max_tokens = max_tokens_old

                # since max prompt is enforced, we filter out too long prompts.
                if curr_step_prompt_length > self.max_prompt_length:
                    logger.warning("exit for prompt_len exceed max model length error...")
                    termination_reason = "PROMPT_TRUNCATION"
                    break
                # handle exceed max model length error
                if curr_step_prompt_length + max_tokens > max_model_len:
                    logger.warning("exit for prompt_len + max_tokens exceed max model length error...")
                    termination_reason = "EXCEED_MODEL_LENGTH"
                    break

            system_prompt = step_info.get("system_prompt", None)
            if system_prompt is not None:
                truncated_prompt = system_prompt if len(system_prompt) <= 50 else system_prompt[:50] + "......"
                logger.info(f"system prompt change: session_id={session_id}, step_idx={step_idx}: {truncated_prompt}")
                agent.update_system_prompt(system_prompt)
            user_prompt = step_info.get("user_prompt", None)
            if user_prompt is not None:
                truncated_prompt = user_prompt if len(user_prompt) <= 50 else user_prompt[:50] + "......"
                logger.info(f"user prompt change: session_id={session_id}, step_idx={step_idx}: {truncated_prompt}")
                agent.update_user_prompt(user_prompt)

            # Update agent with model response
            model_response = step_info["model_response"]
            action: Action = agent.update_from_model(model_response)
            action = action.action

            raw_reward = step_info.get("reward", None)
            tool_response = step_info.get("env_response", None)
            end = False

            if step_idx == min(env.max_steps, len(step_info_list)) - 1:
                end = True
            next_observation, reward, done, info = env.step(action, end, tool_response, raw_reward)

            # metrics
            step_start_time = step_info.get("start_time", 0)
            llm_step_time = step_info.get("llm_time", 0)
            env_step_time = step_info.get("env_time", 0)
            llm_step_times.append(llm_step_time)
            env_step_times.append(env_step_time)
            delta_time = llm_step_time + env_step_time
            llm_time += llm_step_time
            env_time += env_step_time
            total_time += delta_time

            termination_reason = traj_res.get("termination_reason", None)

            info = {}
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len
            logger.info(
                f"trajectory performance status, appID:{application_id}, "
                f"step_idx:{step_idx}, start_time:{step_start_time}, "
                f"end_time:{step_start_time} + {delta_time}, env_time: {env_time}"
            )

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

            if step_idx == min(env.max_steps, len(step_info_list)) - 1:
                prompt_token_ids, response_token_ids, response_mask_ids = parse_whole_response_token_ids(
                    self.chat_parser, agent.chat_completions, tools, self.chat_parser.get_assistant_token()
                )
                prompt_tokens = prompt_token_ids
                logger.info(f"session_id:{session_id} messages:{agent.chat_completions}")
                logger.info(
                    f"trajectory performance status, appID:{application_id}, step_idx:{step_idx}, "
                    f"prompt_length:{len(prompt_token_ids)}, response_length:{len(response_token_ids)} "
                    f"curr_step_prompt_length：{curr_step_prompt_length}"
                )
                # Update response token length
                response_token_len += len(response_token_ids)
                # Reached maximum number of tokens for the trajectory
                curr_prompt_length = len(prompt_token_ids)
                if not self.enforce_max_prompt_length and (curr_prompt_length + response_token_len) >= max_model_len:
                    # Update token collections
                    response_tokens = response_token_ids[:max_model_len]
                    response_masks = response_mask_ids[:max_model_len]
                    termination_reason = "TRUNCATION"
                    # handle returning
                    break
                else:
                    # Update the token version of trajectory
                    response_tokens = response_token_ids
                    response_masks = response_mask_ids

            # Check if episode is done
            if termination_reason == "ENV_DONE":
                break

            if step_idx >= self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        masked_out = False
        # info: self.overlong_filter=false
        if self.overlong_filter:
            if termination_reason == "TRUNCATION" or termination_reason == "TIMEOUT":
                # Mask out the entire response for overlong trajectories if the reward is 0.
                response_masks = [0] * len(response_masks)
                masked_out = True

        # add by ts: env timeout, mask out
        if termination_reason == "ENV_TIMEOUT":
            response_masks = [0] * len(response_masks)

        if hasattr(env, "compute_final_reward") and not masked_out:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = env.compute_final_reward()
            reward_time = time.time() - start_time
            cur_step.reward = reward

        env.close()

        if termination_reason:
            if reward > 0:
                color = "green"
            else:
                color = "yellow"
            colorful_print(
                f"Trajectory {idx} / Application {application_id} completed due to: {termination_reason}. Reward is {reward}. \n",
                color,
            )
            if masked_out:
                colorful_print(
                    f"Trajectory {idx} / Application {application_id} is masked out due to overlong filter.", "red"
                )

        trajectory: Trajectory = agent.trajectory
        trajectory.data_id = id_32
        trajectory.training_id = self.train_id
        trajectory.epoch_id = 0
        trajectory.iteration_id = self.iteration
        trajectory.sample_id = env.sample_id
        trajectory.trajectory_id = (
            trajectory.data_id
            + "-"
            + trajectory.training_id
            + "-"
            + str(trajectory.epoch_id)
            + "-"
            + str(trajectory.iteration_id)
            + "-"
            + str(trajectory.sample_id)
            + "-"
            + "0"
            + str(application_id)
        )
        # Aggregate final trajectory statistics
        self.compute_trajectory_reward_fn(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)

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
            logger.info(f"tool call reward: {trajectory.toolcall_reward}")
            logger.info(f"res reward: {trajectory.res_reward}")
            token_result = {
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                "response_masks": torch.tensor(response_masks, dtype=torch.long),
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "prompt_id": trajectory.prompt_id,
                "tools": tools,
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
                    "llm_step_times": llm_step_times,
                    "prompt_len": len(prompt_tokens),
                    "response_len": len(response_tokens),
                },
            }
            return token_result
