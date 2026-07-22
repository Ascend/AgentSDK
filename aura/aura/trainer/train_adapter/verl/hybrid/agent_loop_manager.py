# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0 OR MulanPSL-2.0
#
# This file is part of the AgentSDK project.
# Adapted from rucaibox/swe-master/DeepSWE_RL/rllm/rllm/trainer/verl/agent_ppo_trainer.py
# Copyright (c) 2026 Huatong Song
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
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import asyncio
import time
import json
import os
from typing import Any, List

import numpy as np
import torch
from omegaconf import DictConfig

from aura.base.log.loggers import Loggers
from aura.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask
from aura.runner.agent_engine_wrapper.vaee_v2.vaee_types import Trajectory, Episode
from aura.runner.infer_router import InferRouter
from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.utils import hf_tokenizer

logger = Loggers(__name__).get_logger()


async def launch_server(infer_service: str, model_name: str, chat_server_list: list[str]) -> None:
    """Launch inference servers for the given chat server addresses.

    Args:
        infer_service: Name of the inference service to launch.
        model_name: Model identifier to serve.
        chat_server_list: List of host:port strings for chat servers.
    """
    chat_server_list = [f"http://{chat_server}" for chat_server in chat_server_list]

    logger.info(f"======launch_server chat_server={chat_server_list}")
    infer_router = await InferRouter.create()
    await infer_router.launch_server(
        model_name=infer_service, kwargs_list=[{"model_name": model_name, "chat_server": chat_server_list}]
    )


async def create_tasks(agent_service: str, prompts: DataProto, n_samples_per_prompt: int) -> list[AgentTask]:
    """Build AgentTask objects from a DataProto prompt batch.

    Args:
        agent_service: Name of the agent service to invoke.
        prompts: DataProto containing prompt data and metadata.
        n_samples_per_prompt: Number of samples to generate per prompt.

    Returns:
        List of AgentTask instances ready for trajectory generation.
    """
    agent_tasks = []
    for idx in range(len(prompts)):
        known_fields = ["index", "global_steps", "raw_prompt", "reward_model", "extra_info"]
        global_steps = prompts.meta_info["global_steps"]
        problem = prompts.non_tensor_batch["raw_prompt"][idx][0]["content"]

        agent_task = AgentTask(
            task_id=str(idx),
            sample_id=idx % n_samples_per_prompt,
            iteration=global_steps,
            agent_name=agent_service,
            problem=problem,
            prompt_id=idx // n_samples_per_prompt,
            content="",
            extra_args={},
        )

        if (
            "reward_model" in prompts.non_tensor_batch
            and "ground_truth" in prompts.non_tensor_batch["reward_model"][idx]
        ):
            ground_truth = prompts.non_tensor_batch["reward_model"][idx]["ground_truth"]
            agent_task.ground_truth = ground_truth

        if "extra_info" in prompts.non_tensor_batch:
            extra_info = prompts.non_tensor_batch["extra_info"][idx]
            if isinstance(extra_info, dict):
                for key, value in extra_info.items():
                    agent_task.extra_args[key] = value

        for field in [key for key in prompts.non_tensor_batch.keys() if key not in known_fields]:
            agent_task.extra_args[field] = prompts.non_tensor_batch[field][idx]

        agent_tasks.append(agent_task)

    return agent_tasks


async def generate_trajectory(
    agent_task: AgentTask, addresses, server_handles, mode: str = "Token"
) -> dict | list | Episode:
    """Generate a single trajectory via the AgentRouter.

    Args:
        agent_task: The task specification for trajectory generation.
        mode: ``Token`` for standard rollout, ``Step`` for dynamic-beam stepwise rollout.

    Returns:
        Trajectory dict, list of beam step dicts (stepwise), or Episode.
    """
    from aura.runner.agent_router import AgentRouter

    router = await AgentRouter.create()
    trajectory = await router.generate_trajectory(
        agent_task, mode=mode, addresses=addresses, server_handles=server_handles
    )
    return trajectory


async def transform_beam_steps_to_batch(config: Any, tokenizer: Any, beam_trajectories: List) -> DataProto:
    """Convert dynamic-beam step candidates into a padded DataProto batch."""
    def _sort_key(t):
        return (
            int(t.get("prompt_index", t.get("idx", 0)) if isinstance(t, dict) else 0),
            int(t.get("step_depth", 0) if isinstance(t, dict) else 0),
        )

    beam_trajectories = [t for t in beam_trajectories if isinstance(t, dict) and t.get("steps")]
    beam_trajectories.sort(key=_sort_key)

    all_initial_tokens_list = []
    all_response_tokens_list = []
    all_masks_list = []
    traj_scores = []
    group_keys = []
    index_in_batch = []
    index_in_steps = []
    index_in_group = []
    is_last_step = []

    for traj in beam_trajectories:
        steps = traj["steps"]
        step = steps[-1]
        prompt_index = int(traj.get("prompt_index", traj.get("idx", 0)))
        step_idx = int(traj.get("step_depth", len(steps) - 1))
        group_source = traj.get("idx", traj.get("prompt_id", traj.get("task_id", traj.get("application_id", prompt_index))))
        step_scores = traj.get("mc_returns", []) or []
        trajectory_reward = traj.get("trajectory_reward", traj.get("reward", 0.0))
        score = step_scores[-1] if len(step_scores) > 0 else trajectory_reward

        prompt = torch.tensor(tokenizer.encode(step["prompt"], add_special_tokens=False), dtype=torch.long)
        response = torch.tensor(tokenizer.encode(step["response"], add_special_tokens=False), dtype=torch.long)
        if prompt.numel() == 0 or response.numel() == 0:
            continue
        all_initial_tokens_list.append(prompt)
        all_response_tokens_list.append(response)
        all_masks_list.append(torch.ones_like(response, dtype=torch.long))
        traj_scores.append(float(score))
        group_keys.append(f"{group_source}_{prompt_index}_{step_idx}")
        index_in_batch.append(prompt_index)
        index_in_steps.append(step_idx)
        index_in_group.append(str(group_source))
        is_last_step.append(bool(traj.get("is_last_step", True)))

    if not all_initial_tokens_list:
        raise ValueError("transform_beam_steps_to_batch: no valid beam candidate")

    prompts_batch = torch.nn.utils.rnn.pad_sequence(
        [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    ).flip(dims=[1])
    response_batch = torch.nn.utils.rnn.pad_sequence(
        all_response_tokens_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    input_ids_list = torch.concat([prompts_batch, response_batch], dim=1)

    traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
    attention_mask = torch.where(input_ids_list != tokenizer.pad_token_id, 1, 0)
    position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
    prompt_length = prompts_batch.shape[1]
    valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)
    for i, traj_score in enumerate(traj_scores):
        last_valid_idx = valid_response_length_sequences[i] - 1
        if 0 <= last_valid_idx < score_batch.shape[1]:
            score_batch[i, last_valid_idx] = traj_score

    batch_tensors = {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": response_batch,
        "prompts": prompts_batch,
        "token_level_rewards": score_batch,
        "response_mask": traj_mask,
        "rm_scores": score_batch,
    }
    batch = DataProto.from_dict(tensors=batch_tensors)
    batch.non_tensor_batch["uid"] = np.array(group_keys, dtype=object)
    batch.non_tensor_batch["index_in_batch"] = np.array(index_in_batch, dtype=object)
    batch.non_tensor_batch["index_in_steps"] = np.array(index_in_steps, dtype=object)
    batch.non_tensor_batch["index_in_group"] = np.array(index_in_group, dtype=object)
    batch.non_tensor_batch["is_last_step"] = np.array(is_last_step, dtype=object)
    batch.meta_info["timing"] = {}
    return batch


def _is_mock_value(value: Any) -> bool:
    return type(value).__module__ == "unittest.mock"


def _get_stepwise_advantage_enabled(config: Any) -> bool:
    try:
        algorithm_cfg = config.get("algorithm", {})
    except Exception:
        algorithm_cfg = {}
    if _is_mock_value(algorithm_cfg):
        return False
    if isinstance(algorithm_cfg, dict):
        value = algorithm_cfg.get("use_stepwise_advantage", False)
    else:
        value = getattr(algorithm_cfg, "use_stepwise_advantage", False)
    if _is_mock_value(value):
        return False
    return bool(value)


async def transform_trajectories_to_batch(config: Any, tokenizer: Any, trajectories: list[dict]) -> DataProto:
    """Convert raw trajectory dicts into a padded DataProto batch.

    Tensor layout:
      - prompt_ids:    [Pad, Pad, ..., Token, Token]  (left-padded)
      - response_ids:  [Token, Token, ..., Pad, Pad]  (right-padded)
      - input_ids:     [prompt_ids, response_ids]
      - response_mask: [1, 1, ..., 0, 0]  (1 = LLM-generated, 0 = tool/padding)

    Args:
        config: Training configuration.
        tokenizer: HuggingFace tokenizer instance.
        trajectories: List of trajectory dicts from rollout.

    Returns:
        DataProto with padded tensors and non-tensor metadata.
    """
    trajectories.sort(key=lambda x: x["idx"])

    all_prompt_ids = []
    all_initial_tokens_list = []
    all_response_tokens_list = []
    all_masks_list = []
    all_logprobs_list = []
    traj_scores = []
    chat_completions = []
    cancel_logprobs = False

    for traj in trajectories:
        prompt_id = traj["prompt_id"]
        prompt_tokens = traj["prompt_tokens"]
        response_tokens = traj["response_tokens"]
        if prompt_tokens.numel() == 0 or response_tokens.numel() == 0:
            raise ValueError(
                f"Both prompt {prompt_tokens.numel()} and response {response_tokens.numel()} "
                f"of trajectory shouldn't be empty. Please check make sure environment is working and the config"
            )
        all_initial_tokens_list.append(prompt_tokens)
        all_response_tokens_list.append(response_tokens)
        if "logprobs" in traj and len(traj["logprobs"]) != 0 and not cancel_logprobs:
            all_logprobs_list.append(torch.tensor(traj["logprobs"]))
        else:
            cancel_logprobs = True
        all_masks_list.append(traj["response_masks"])
        traj_scores.append(traj["trajectory_reward"])
        chat_completions.append(traj["chat_completions"])
        all_prompt_ids.append(prompt_id)

    # Reverse, pad, then flip to achieve left-padding for prompts
    prompts_batch = torch.nn.utils.rnn.pad_sequence(
        [torch.flip(token, dims=[0]) for token in all_initial_tokens_list],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    ).flip(dims=[1])

    response_batch = torch.nn.utils.rnn.pad_sequence(
        all_response_tokens_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    input_ids_list = torch.concat([prompts_batch, response_batch], dim=1)

    traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
    trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)
    attention_mask = torch.where(trajectory_batch != tokenizer.pad_token_id, 1, 0)

    position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    score_batch = torch.zeros_like(response_batch, dtype=torch.float32)

    prompt_length = prompts_batch.shape[1]
    valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

    for idx, traj_score in enumerate(traj_scores):
        last_valid_idx = valid_response_length_sequences[idx] - 1
        if 0 <= last_valid_idx < score_batch.shape[1]:
            score_batch[idx, last_valid_idx] = traj_score

    rollout_log_probs_batch = None
    if not cancel_logprobs:
        rollout_log_probs_batch = torch.nn.utils.rnn.pad_sequence(
            all_logprobs_list,
            batch_first=True,
            padding_value=0.0,
        )

    batch_tensors = {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": response_batch,
        "prompts": prompts_batch,
        "token_level_rewards": score_batch,
        "response_mask": traj_mask,
        "rm_scores": score_batch,
    }

    if not cancel_logprobs:
        batch_tensors["rollout_log_probs"] = rollout_log_probs_batch
    batch = DataProto.from_dict(tensors=batch_tensors)
    batch.non_tensor_batch["uid"] = np.array(all_prompt_ids)
    batch.meta_info["timing"] = {}

    return batch


async def transform_episodes_to_batch(tokenizer, trajectories: List[Trajectory], task_id_list) -> DataProto:
    combined = sorted(zip(task_id_list, trajectories))
    task_id_list, trajectories = map(list, zip(*combined))

    all_prompt_ids = []
    all_prompt_tokens_list = []
    all_response_tokens_list = []
    all_masks_list = []
    all_logprobs_list = []
    traj_scores = []
    step_nums = []
    is_last_step = []  # steps中的最后一个step是否是 整个trajectory中的最后一个step
    cancel_logprobs = False

    for i in range(0, len(trajectories)):
        task_id = task_id_list[i]  # TODO: 调整不确定待调整，目标是保证每个prompt的唯一性
        traj = trajectories[i]
        steps = traj.steps

        # step mode
        if False:  # TODO: 这里先写成False强制token模式，待后续完善可改为traj.is_cumulative
            for step in steps:
                all_prompt_ids.append(task_id)
                all_prompt_tokens_list.append(torch.tensor(step.prompt_ids, dtype=torch.long))
                all_response_tokens_list.append(torch.tensor(step.response_ids, dtype=torch.long))
                all_masks_list.append(torch.tensor([1] * len(step.response_ids), dtype=torch.long))
                if step.logprobs is not None and len(step.logprobs) != 0 and not cancel_logprobs:
                    all_logprobs_list.append(torch.tensor(step.logprobs))
                else:
                    cancel_logprobs = True
            step_nums.append(len(steps))
            all_prompt_ids.extend([task_id for _ in range(len(steps))])
            is_last_step.extend([False for _ in range(len(steps))])
            is_last_step[-1] = True
        # token mode
        else:
            all_prompt_ids.append(task_id)
            all_prompt_tokens_list.append(torch.tensor(steps[0].prompt_ids, dtype=torch.long))

            # 计算response_masks: response部分填充1，tool部分填充0，tool由本轮prompt减上一轮的（prompt+response)形成
            # 计算logprobs: response部分保持原有logprobs，tool部分填充0，如果有logprobs值为None的就放弃计算，直接返回[]
            response_tokens = []
            response_masks = []
            logprobs = []
            for j in range(len(steps)):
                response_tokens.extend(steps[j].response_ids)
                response_masks.extend([1] * len(steps[j].response_ids))
                if steps[j].logprobs is not None and len(steps[j].logprobs) != 0 and not cancel_logprobs:
                    logprobs.extend(steps[j].logprobs)
                else:
                    cancel_logprobs = True
                if j < len(steps) - 1:
                    prefix_len = len(steps[j].prompt_ids) + len(steps[j].response_ids)
                    tool_tokens = steps[j + 1].prompt_ids[prefix_len:]
                    response_tokens.extend(tool_tokens)
                    response_masks.extend([0] * len(tool_tokens))
                    if not cancel_logprobs:
                        logprobs.extend([0] * len(tool_tokens))
            all_response_tokens_list.append(torch.tensor(response_tokens, dtype=torch.long))
            all_masks_list.append(torch.tensor(response_masks, dtype=torch.long))
            if not cancel_logprobs:
                all_logprobs_list.append(torch.tensor(logprobs))

        traj_scores.append(traj.reward)

    # reverse the list and create tensors, pad, then flip to achieve left padding
    prompts_batch = torch.nn.utils.rnn.pad_sequence(
        [torch.flip(i, dims=[0]) for i in all_prompt_tokens_list],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    ).flip(dims=[1])

    response_batch = torch.nn.utils.rnn.pad_sequence(
        all_response_tokens_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    input_ids_list = torch.concat([prompts_batch, response_batch], dim=1)

    prompt_length_list = []
    for prompt in all_prompt_tokens_list:
        prompt_length_list.append(torch.tensor([len(prompt)]))

    traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
    trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)
    attention_mask = torch.where(trajectory_batch != tokenizer.pad_token_id, 1, 0)

    # Compute position_ids
    position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    # Place all rewards to last response token
    score_batch = torch.zeros_like(response_batch, dtype=torch.float32)

    prompt_length = prompts_batch.shape[1]
    valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

    if False:  # TODO: 这里先写成False强制token模式，待后续完善可改为traj.is_cumulative或者其它
        step_index = 0
        for i, traj_score in enumerate(traj_scores):
            step_num = step_nums[i]
            for _ in range(step_num):
                last_valid_idx = valid_response_length_sequences[step_index] - 1
                if 0 <= last_valid_idx < score_batch.shape[1]:
                    score_batch[step_index, last_valid_idx] = traj_score
                step_index += 1
    else:
        for i, traj_score in enumerate(traj_scores):
            last_valid_idx = valid_response_length_sequences[i] - 1
            if 0 <= last_valid_idx < score_batch.shape[1]:
                score_batch[i, last_valid_idx] = traj_score
    if all_logprobs_list:
        rollout_log_probs_batch = torch.nn.utils.rnn.pad_sequence(
            all_logprobs_list,
            batch_first=True,
            padding_value=0.0,
        )

    batch_tensors = {
        "input_ids": input_ids_list,  # 无pad，长短不一
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": response_batch,  # 右pad
        "prompts": prompts_batch,  # 左pad
        "token_level_rewards": score_batch,  # 右pad，只有在长度那一位有值为score其余为0
        "response_mask": traj_mask,  # 形状与responses一样，右pad 0
        "rm_scores": score_batch,
    }
    if not cancel_logprobs:
        batch_tensors["rollout_log_probs"] = rollout_log_probs_batch

    batch = DataProto.from_dict(tensors=batch_tensors)
    batch.non_tensor_batch["uid"] = np.array(all_prompt_ids)  # idxs
    if False:  # TODO: 这里先写成False强制token模式，待后续完善可改为traj.is_cumulative或者其它
        batch.non_tensor_batch["is_last_step"] = np.array(is_last_step)

    batch.meta_info["timing"] = {}
    logger.info(f"transform_episodes_to_batch: {batch=}")
    return batch


class HybridAgentLoopManager(AgentLoopManager):
    """Agent loop manager for hybrid (sync rollout + async agent) training mode."""

    def __init__(self, config: DictConfig, *args, **kwargs):
        kwargs.pop('config', None)
        super().__init__(config, *args, **kwargs)
        self.tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=True)

    async def _initialize_llm_servers(self) -> None:
        """重写此方法，在父类设置 server_addresses 后执行 hybrid 初始化"""
        await super()._initialize_llm_servers()  # 先让父类设置 server_addresses

        # hybrid 特有初始化
        self.chat_server_list = self.server_addresses
        self.tokenizer = hf_tokenizer(self.config.actor_rollout_ref.model.path, trust_remote_code=True)
        self.iteration = 0
        self.traj_output_path = self.config.extras.traj_output_path
        self.perf_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        await launch_server(
            infer_service=self.config.extras.infer_service,
            model_name=self.config.actor_rollout_ref.model.path,
            chat_server_list=self.chat_server_list,
        )

    async def _init_agent_loop_workers(self) -> None:
        """Override: no separate agent loop workers needed in hybrid mode."""
        pass

    async def async_generate_sequences(
        self,
        config: Any,
        prompts: DataProto,
        tokenizer: Any,
    ) -> DataProto:
        """Asynchronously generate trajectories for all prompts.

        Args:
            config: Training configuration.
            prompts: DataProto containing prompt data.
            tokenizer: HuggingFace tokenizer instance.

        Returns:
            DataProto batch assembled from generated trajectories.
        """
        agent_tasks = await create_tasks(config.extras.agent_service, prompts, config.actor_rollout_ref.rollout.n)
        use_stepwise = _get_stepwise_advantage_enabled(config)
        if hasattr(config.extras, "chat_interface") and config.extras.chat_interface == "generate":
            futures = [
                asyncio.create_task(
                    generate_trajectory(task, self.chat_server_list, self.server_handles, "Step")
                    if use_stepwise
                    else generate_trajectory(task, self.chat_server_list, self.server_handles)
                )
                for task in agent_tasks
            ]
        else:
            futures = [
                asyncio.create_task(
                    generate_trajectory(task, self.chat_server_list, None, "Step")
                    if use_stepwise
                    else generate_trajectory(task, self.chat_server_list, None)
                )
                for task in agent_tasks
            ]
        episode_list = []
        task_id_list = []
        trajectory_list = []
        for f in futures:
            episode_list.append(await f)

        if use_stepwise:
            beam_trajectories = []
            for item in episode_list:
                if isinstance(item, list):
                    beam_trajectories.extend(item)
                elif item is not None:
                    beam_trajectories.append(item)
            if self.traj_output_path is not None:
                self.write_file(beam_trajectories, prefix="trajectories")
            return await transform_beam_steps_to_batch(config, tokenizer, beam_trajectories)

        if isinstance(episode_list[0], Episode):
            for episode in episode_list:
                task_id_list.append(int(episode.task_id.split("-")[0]))
                trajectory_list.extend(episode.trajectories)  # FIXME: 假设每个episode只有一个trajectories
            if self.traj_output_path is not None:
                self.write_file(trajectory_list, prefix="trajectories")
            result = await transform_episodes_to_batch(tokenizer, trajectory_list, task_id_list)
        else:
            if self.traj_output_path is not None:
                self.write_file(episode_list, prefix="trajectories")
            result = await transform_trajectories_to_batch(config, tokenizer, episode_list)
        return result

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Synchronous entry point: forward prompts to the agent service and return results.

        Args:
            prompts: DataProto containing prompt data with global_steps in meta_info.

        Returns:
            DataProto batch with generated trajectories.
        """
        output = asyncio.run(self.async_generate_sequences(self.config, prompts, self.tokenizer))

        logger.info(f"generate_sequences: {len(output)=}, {output=}")
        return output

    def wake_up(self) -> None:
        """Wake up all rollout replicas for weight synchronization."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self) -> None:
        """Put all rollout replicas to sleep after generation."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def clear_kv_cache(self) -> None:
        """Clear KV cache on all rollout replicas."""
        self._run_all([replica.clear_kv_cache() for replica in self.rollout_replicas])

    def write_file(self, data_dict: Any, prefix: str) -> None:
        """Serialize trajectory data to a JSON file.

        Args:
            data_dict: Data to serialize (may contain Tensors).
            prefix: Filename prefix for the output JSON file.
        """

        def convert_to_string(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return str(value.tolist())
            elif isinstance(value, list):
                return [convert_to_string(v) for v in value]
            elif isinstance(value, dict):
                return {key: convert_to_string(v) for key, v in value.items()}
            else:
                return str(value)

        add_iter = {"iteration": self.iteration, f"{prefix}": data_dict}
        data_str = convert_to_string(add_iter)

        output_file = f'rollout_{prefix}_{int(self.perf_timestamp)}.json'
        os.makedirs(self.traj_output_path, exist_ok=True)
        with open(os.path.join(self.traj_output_path, output_file), 'a') as f:
            json.dump(data_str, f, indent=4, ensure_ascii=False)
            f.write('\n')

        logger.info(f'write_file {output_file} in iteration {self.iteration} done')
