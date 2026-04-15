# -*- coding: utf-8 -*-
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
from typing import List

import numpy as np
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.utils import hf_tokenizer
from verl.experimental.agent_loop import AgentLoopManager

from agentic_rl.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask
from verl import DataProto

from agentic_rl.runner.infer_router import InferRouter
from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


async def launch_server(infer_service: str, model_name: str, server_addresses: List[str]) -> None:
    """
    Launch the inference server by registering chat server endpoints.

    Args:
        infer_service: Name of the inference service to register.
        model_name: Model identifier used by the chat servers.
        server_addresses: List of host:port addresses for the chat servers.
    """
    chat_server_urls = [f"http://{address}" for address in server_addresses]

    infer_router = await InferRouter.create()
    await infer_router.launch_server(
        model_name=infer_service,
        kwargs_list=[{
            "model_name": model_name,
            "chat_server": chat_server_urls
        }]
    )


async def create_tasks(agent_service: str, prompts: DataProto, n_samples_per_prompt: int) -> List[AgentTask]:
    """
    Build a list of AgentTask objects from the given prompts.

    Args:
        agent_service: Name of the agent service to route tasks to.
        prompts: DataProto containing prompt data and metadata.
        n_samples_per_prompt: Number of samples generated per unique prompt.

    Returns:
        List of AgentTask instances ready for trajectory generation.
    """
    agent_tasks = []
    for i in range(len(prompts)):
        index = (
            prompts.non_tensor_batch["index"][i]
            if "index" in prompts.non_tensor_batch
            else i
        )
        global_steps = prompts.meta_info["global_steps"]
        problem = prompts.non_tensor_batch["raw_prompt"][i][0]["content"]
        ground_truth = prompts.non_tensor_batch["reward_model"][i]["ground_truth"]

        agent_task = AgentTask(
            task_id=f"{global_steps}_{index}_{i % n_samples_per_prompt}",
            sample_id=index,
            iteration=global_steps,
            agent_name=agent_service,
            problem=problem,
            ground_truth=ground_truth,
            prompt_id=index,
            content="",
        )
        agent_tasks.append(agent_task)
    return agent_tasks


async def generate_trajectory(agent_task: AgentTask) -> dict:
    """
    Generate a single trajectory by routing the agent task.

    Args:
        agent_task: The task describing the problem for trajectory generation.

    Returns:
        A dictionary containing prompt tokens, response tokens, masks, and rewards.
    """
    from agentic_rl.runner.agent_router import AgentRouter

    router = await AgentRouter.create()
    trajectory = await router.generate_trajectory(agent_task, mode='Token')
    return trajectory


async def transform_trajectories_to_batch(config: DictConfig, tokenizer, trajectory_list: List[dict]) -> DataProto:
    """
    Transform a list of trajectory dicts into a batched DataProto.

    Pads prompts (left) and responses (right) to configured lengths, assembles
    tensor and non-tensor batches, and computes reward scores.

    Args:
        config: OmegaConf configuration with rollout length settings.
        tokenizer: HuggingFace tokenizer used for padding.
        trajectory_list: List of trajectory dictionaries, each containing
            prompt_tokens, response_tokens, response_masks, and trajectory_reward.

    Returns:
        A DataProto with batched tensors and non-tensor metadata.

    Raises:
        ValueError: If any prompt or response exceeds the configured max length.
        RuntimeError: If concatenated length does not match expected total.
    """

    import torch
    processed_inputs = []

    max_prompt_length = config.actor_rollout_ref.rollout.prompt_length
    max_response_length = config.actor_rollout_ref.rollout.response_length
    expected_total_length = max_prompt_length + max_response_length

    for i, traj in enumerate(trajectory_list):
        p_tokens = traj["prompt_tokens"]
        r_tokens = traj["response_tokens"]

        if len(p_tokens) > max_prompt_length:
            raise ValueError(f"Entry {i} Prompt too long: {len(p_tokens)} > {max_prompt_length}")
        if len(r_tokens) > max_response_length:
            raise ValueError(f"Entry {i} Response too long: {len(r_tokens)} > {max_response_length}")

        tokenizer.padding_side = "left"
        prompt_output = tokenizer.pad(
            {"input_ids": p_tokens},
            padding="max_length",
            max_length=max_prompt_length,
            return_tensors="pt",
        )

        tokenizer.padding_side = "right"
        response_output = tokenizer.pad(
            {"input_ids": r_tokens},
            padding="max_length",
            max_length=max_response_length,
            return_tensors="pt",
        )

        # Normalize to [1, L] regardless of tokenizer output shape
        p_ids = prompt_output["input_ids"].view(1, -1)
        p_mask = prompt_output["attention_mask"].view(1, -1)
        r_ids = response_output["input_ids"].view(1, -1)
        r_mask = response_output["attention_mask"].view(1, -1)

        response_mask_output = tokenizer.pad(
            {"input_ids": traj["response_masks"]},
            padding="max_length",
            max_length=max_response_length,
            return_tensors="pt",
        )
        res_mask_ids = response_mask_output["input_ids"].view(1, -1)
        # Zero out padding positions by intersecting with attention_mask
        response_mask = res_mask_ids * r_mask

        input_ids = torch.cat([p_ids, r_ids], dim=1)
        attention_mask = torch.cat([p_mask, r_mask], dim=1)

        if input_ids.shape[1] != expected_total_length:
            raise RuntimeError(f"Entry {i} total length {input_ids.shape[1]} != {expected_total_length}")

        # Cumulative count of valid tokens; padding positions stay at 0
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
        position_ids = torch.clamp(position_ids, min=0)

        # Place reward score at the last valid response token
        rm_scores = torch.zeros([1, max_response_length], dtype=torch.float32)
        valid_response_len = r_mask.sum().item()
        if valid_response_len > 0:
            rm_scores[0, int(valid_response_len - 1)] = traj["trajectory_reward"]

        processed_inputs.append({
            "tensors": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": p_ids,
                "responses": r_ids,
                "response_mask": response_mask,
                "rm_scores": rm_scores,
            },
            "non_tensors": {
                "__num_turns__": traj.get("num_turns", 1),
                "raw_prompt": traj.get("raw_prompt", ""),
                "reward_extra_info": traj.get("reward_extra_info", {}),
                "multi_modal_inputs": traj.get("multi_modal_inputs", None),
            }
        })

    # Stack individual [1, L] tensors into [Batch, L]
    batch_tensors = {}
    for key in processed_inputs[0]["tensors"].keys():
        batch_tensors[key] = torch.cat([item["tensors"][key] for item in processed_inputs], dim=0)

    batch = TensorDict(batch_tensors, batch_size=len(trajectory_list))

    non_tensor_batch = {}
    for key in processed_inputs[0]["non_tensors"].keys():
        non_tensor_batch[key] = np.array([item["non_tensors"][key] for item in processed_inputs], dtype=object)

    # Flatten reward_extra_info metrics (e.g. 'acc') into top-level columns
    reward_extra_info_sample = processed_inputs[0]["non_tensors"]["reward_extra_info"]
    reward_keys = reward_extra_info_sample.keys() if isinstance(reward_extra_info_sample, dict) else []

    for r_key in reward_keys:
        non_tensor_batch[r_key] = np.array(
            [item["non_tensors"]["reward_extra_info"].get(r_key, 0) for item in processed_inputs]
        )

    return DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info={
            "timing": {},
            "reward_extra_keys": list(reward_keys)
        }
    )


async def async_generate_sequences(
        config: DictConfig,
        prompts: DataProto,
        tokenizer
) -> DataProto:
    """
    Create agent tasks from prompts, generate trajectories concurrently, and batch the results.

    Args:
        config: OmegaConf configuration containing agent service and rollout settings.
        prompts: DataProto with prompt data to generate trajectories for.
        tokenizer: HuggingFace tokenizer used for padding during batching.

    Returns:
        A DataProto containing the batched trajectory results.
    """
    agent_tasks = await create_tasks(
        config.extras.agent_service, prompts, config.actor_rollout_ref.rollout.n
    )
    futures = [
        asyncio.create_task(generate_trajectory(task))
        for task in agent_tasks
    ]
    trajectory_list = []
    for future in futures:
        trajectory_list.append(await future)
    result = await transform_trajectories_to_batch(config, tokenizer, trajectory_list)
    return result


class HybridAgentLoopManager(AgentLoopManager):
    """Agent loop manager that delegates inference to external chat servers in hybrid mode."""

    def __init__(self, config: DictConfig, *args, **kwargs):
        """
        Initialize the hybrid agent loop manager and launch inference servers.

        Args:
            config: OmegaConf configuration for actor, rollout, and inference settings.
            *args: Positional arguments forwarded to the parent AgentLoopManager.
            **kwargs: Keyword arguments forwarded to the parent AgentLoopManager.
        """
        super().__init__(config=config, *args, **kwargs)
        self.chat_server_list = self.server_addresses
        self.tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=True)

        asyncio.run(launch_server(
            infer_service=self.config.extras.infer_service,
            model_name=self.config.actor_rollout_ref.model.path,
            server_addresses=self.chat_server_list
        ))

    def _init_agent_loop_workers(self):
        pass

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Forward prompts to the agent service and return batched trajectory output.

        Args:
            prompts: DataProto containing the input prompts.

        Returns:
            A DataProto with generated trajectory tensors and metadata.
        """
        # Fix for Issue #4147: Always call wake_up() to ensure weight sync
        # The wake_up()/sleep() methods internally check free_cache_engine
        self.wake_up()

        output = asyncio.run(async_generate_sequences(self.config, prompts, self.tokenizer))

        # Fix for Issue #4147: Always call sleep() to ensure proper cleanup
        self.sleep()

        return output

    def wake_up(self) -> None:
        """Activate all rollout replicas to ensure model weights are synced."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self) -> None:
        """Deactivate all rollout replicas and release inference resources."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def clear_kv_cache(self) -> None:
        """Clear the KV cache on all rollout replicas."""
        self._run_all([replica.clear_kv_cache() for replica in self.rollout_replicas])
