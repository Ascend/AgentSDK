# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0 OR MulanPSL-2.0
#
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
# 
import asyncio
import datetime
import json
import os
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask
from agentic_rl.runner.infer_router import InferRouter
from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.utils import hf_tokenizer

logger = Loggers(__name__).get_logger()


async def launch_server(
    infer_service: str, model_name: str, chat_server_list: list[str]
) -> None:
    """Launch inference servers for the given chat server addresses.

    Args:
        infer_service: Name of the inference service to launch.
        model_name: Model identifier to serve.
        chat_server_list: List of host:port strings for chat servers.
    """
    chat_server_list = [
        f"http://{chat_server}"
        for chat_server in chat_server_list
    ]

    infer_router = await InferRouter.create()
    await infer_router.launch_server(
        model_name=infer_service,
        kwargs_list=[{
            "model_name": model_name,
            "chat_server": chat_server_list
        }]
    )


async def create_tasks(
    agent_service: str, prompts: DataProto, n_samples_per_prompt: int
) -> list[AgentTask]:
    """Build AgentTask objects from a DataProto prompt batch.

    Args:
        agent_service: Name of the agent service to invoke.
        prompts: DataProto containing prompt data and metadata.
        n_samples_per_prompt: Number of samples to generate per prompt.

    Returns:
        List of AgentTask instances ready for trajectory generation.
    """
    agent_tasks = []
    known_fields = ["index", "global_steps", "raw_prompt", "reward_model", "extra_info"]

    for idx in range(len(prompts)):
        index = prompts.non_tensor_batch["index"][idx] if "index" in prompts.non_tensor_batch else idx
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


async def generate_trajectory(agent_task: AgentTask) -> dict:
    """Generate a single trajectory via the AgentRouter.

    Args:
        agent_task: The task specification for trajectory generation.

    Returns:
        Dictionary containing trajectory data (tokens, rewards, etc.).
    """
    from agentic_rl.runner.agent_router import AgentRouter

    router = await AgentRouter.create()
    trajectory = await router.generate_trajectory(agent_task, mode='Token')
    return trajectory


async def transform_trajectories_to_batch(
    config: Any, tokenizer: Any, trajectories: list[dict]
) -> DataProto:
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
        all_logprobs_list.append(torch.tensor(traj["logprobs"]))
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

    prompt_length_list = [torch.tensor([len(prompt)]) for prompt in all_initial_tokens_list]

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
        "rollout_log_probs": rollout_log_probs_batch,
        "rm_scores": score_batch,
    }

    batch = DataProto.from_dict(tensors=batch_tensors)
    batch.non_tensor_batch["uid"] = np.array(all_prompt_ids)
    batch.meta_info["timing"] = {}

    return batch


class HybridAgentLoopManager(AgentLoopManager):
    """Agent loop manager for hybrid (sync rollout + async agent) training mode."""

    def __init__(self, config: DictConfig, *args, **kwargs) -> None:
        super().__init__(config=config, *args, **kwargs)
        self.chat_server_list = self.server_addresses
        self.tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=True)
        self.iteration = 0
        self.traj_output_path = config.extras.traj_output_path

        asyncio.run(launch_server(
            infer_service=self.config.extras.infer_service,
            model_name=self.config.actor_rollout_ref.model.path,
            chat_server_list=self.chat_server_list
        ))

    def _init_agent_loop_workers(self) -> None:
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
        agent_tasks = await create_tasks(
            config.extras.agent_service, prompts, config.actor_rollout_ref.rollout.n
        )
        futures = [
            asyncio.create_task(generate_trajectory(task))
            for task in agent_tasks
        ]
        trajectory_list = [await future for future in futures]

        if self.traj_output_path is not None:
            self.write_file(trajectory_list, prefix="trajectories")

        result = await transform_trajectories_to_batch(config, tokenizer, trajectory_list)
        return result

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Synchronous entry point: forward prompts to the agent service and return results.

        Args:
            prompts: DataProto containing prompt data with global_steps in meta_info.

        Returns:
            DataProto batch with generated trajectories.
        """
        self.iteration = prompts.meta_info["global_steps"]
        logger.info(f"iteration: {self.iteration}, generate_sequences: {len(prompts)=}, {prompts=}")
        self.wake_up()

        output = asyncio.run(self.async_generate_sequences(self.config, prompts, self.tokenizer))

        self.sleep()

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

        output_file = f'rollout_{prefix}_{int(datetime.datetime.now().timestamp())}.json'
        output_path = os.path.realpath(os.path.join(self.traj_output_path, output_file))

        file_descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(file_descriptor, 'a') as file_handle:
            json.dump(data_str, file_handle, indent=4, ensure_ascii=False)
            file_handle.write('\n')

        logger.info(f'write_file {output_file} in iteration {self.iteration} done')
