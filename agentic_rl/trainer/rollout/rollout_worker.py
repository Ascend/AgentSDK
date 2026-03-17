#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import time
from typing import Callable

import numpy as np
import ray
import torch
from transformers import AutoTokenizer

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.file_utils import FileCheck
from agentic_rl.configs.agentic_rl_config import AgenticRLConfig
from agentic_rl.configs.agentic_rl_config import GenConfig
from agentic_rl.data_manager.data_manager import DataManager
from agentic_rl.runner.agent_engine_wrapper.base import Trajectory, StepTrajectory
from agentic_rl.runner.infer_adapter.async_server import AsyncServerManager
from agentic_rl.runner.runner_worker import RunnerWorker

logger = Loggers(__name__)


@ray.remote
class RolloutWorker:
    def __init__(
        self,
        tokenizer_name_or_path,
        generate_config: GenConfig,
        agentic_rl_config: AgenticRLConfig,
        remove_padding_tensor_dict_to_dict,
        remove_padding_and_split_to_list,
        n_parallel_agents=8,
        max_prompt_length=8192,
        actor_rollout_dispatch_size=0,
        simplify_think_content=False,
        dataset_additional_keys=None,
        worker_group=None,
        global_batch_size=2,
    ):
        self.actor_rollout_dispatch_size = actor_rollout_dispatch_size
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.generate_config = generate_config
        self.agentic_rl_config = agentic_rl_config
        self.parallel_state = None
        self.iteration = 0
        self.dataset_additional_keys = dataset_additional_keys
        self.remove_padding_tensor_dict_to_dict = remove_padding_tensor_dict_to_dict
        self.remove_padding_and_split_to_list = remove_padding_and_split_to_list
        self.n_parallel_agents = n_parallel_agents
        self.max_prompt_length = max_prompt_length
        self.simplify_think_content = simplify_think_content
        self.worker_group = worker_group
        self.use_stepwise_advantage = agentic_rl_config.use_stepwise_advantage
        self.global_batch_size = global_batch_size
        self._param_check()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path, local_files_only=True, weights_only=True
        )
        self.train_backend = self.agentic_rl_config.train_backend
        self.data_manager = DataManager(self.agentic_rl_config.train_backend)
        self.rollout_engine = AsyncServerManager(
            config=self.generate_config,
            agentic_rl_config=self.agentic_rl_config,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            worker_group=worker_group,
        )

        sampling_params = {
            "temperature": self.generate_config.sampling_config.temperature,
            "top_k": self.generate_config.sampling_config.top_k,
            "logprobs": self.generate_config.sampling_config.logprobs,
            "max_tokens": self.generate_config.sampling_config.max_tokens,
            "top_p": self.generate_config.sampling_config.top_p,
            "min_p": self.generate_config.sampling_config.min_p,
            "detokenize": self.generate_config.sampling_config.detokenize,
            "model_name": self.tokenizer_name_or_path,
        }
        servers = self.rollout_engine.async_servers
        addresses = self.rollout_engine.server_addresses

        self.runner_worker = RunnerWorker.remote(
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            sampling_params=sampling_params,
            n_parallel_agents=n_parallel_agents,
            max_prompt_length=max_prompt_length,
            max_model_len=self.generate_config.max_model_len,
            agentic_rl_config=self.agentic_rl_config,
            servers=servers,
            addresses=addresses,
            agent_engine_wrapper_path=self.agentic_rl_config.agent_engine_wrapper_path,
        )

    @staticmethod
    def _validate_param(param_value, param_name, expected_type, *, min_val=None, max_val=None, path_check=False):
        """Universal parameter validation factory function"""
        err_msg = ""
        if not isinstance(param_value, expected_type):
            err_msg = f"{param_name}: {param_value} type error, should {expected_type.__name__}."
        elif min_val is not None and param_value < min_val:
            err_msg = f"{param_name}: {param_value}, should be ≥ {min_val}."
        elif max_val is not None and param_value > max_val:
            err_msg = f"{param_name}: {param_value}, should be ≤ {max_val}."
        elif path_check:
            try:
                FileCheck.check_data_path_is_valid(param_value)
            except (ValueError, TypeError):
                err_msg = f"{param_name}: {param_value} is invalid, should be valid path."

        if err_msg:
            raise ValueError(err_msg)

    def _param_check(self):
        RolloutWorker._validate_param(self.tokenizer_name_or_path, "tokenizer_name_or_path", str, path_check=True)
        RolloutWorker._validate_param(self.generate_config, "generate_config", GenConfig)
        RolloutWorker._validate_param(self.agentic_rl_config, "agentic_rl_config", AgenticRLConfig)
        RolloutWorker._validate_param(
            self.remove_padding_tensor_dict_to_dict, "remove_padding_tensor_dict_to_dict", Callable
        )
        RolloutWorker._validate_param(
            self.remove_padding_and_split_to_list, "remove_padding_and_split_to_list", Callable
        )
        RolloutWorker._validate_param(self.n_parallel_agents, "n_parallel_agents", int, min_val=1)
        RolloutWorker._validate_param(self.max_prompt_length, "max_prompt_length", int, min_val=1)
        RolloutWorker._validate_param(self.actor_rollout_dispatch_size, "actor_rollout_dispatch_size", int, min_val=0)
        RolloutWorker._validate_param(self.simplify_think_content, "simplify_think_content", bool)
        RolloutWorker._validate_param(self.use_stepwise_advantage, "use_stepwise_advantage", bool)
        RolloutWorker._validate_param(self.global_batch_size, "global_batch_size", int, min_val=1)

    def wait_init_finished(self):
        pass

    def init_data_manager(self, data_manager):
        self.data_manager.sync_init_data_manager(data_manager)

    async def _get_batch_data(self, experience_consumer_stage, experience_columns, experience_count):
        while self.data_manager.all_consumed(experience_consumer_stage) > 0:
            batch_data, index = self.data_manager.get_data(
                experience_consumer_stage, experience_columns, experience_count
            )
            if not index:
                continue
            return batch_data, index
        return None, None

    async def _generate_trajectories(self, tasks):
        self.rollout_engine.wake_up()
        try:
            trajectories = ray.get(self.runner_worker.generate_agent_trajectories_async.remote(tasks))
            if self.use_stepwise_advantage:
                if not isinstance(trajectories, list) or not all(
                        (isinstance(traj, StepTrajectory) for traj in trajectories)):
                    raise TypeError("Trajectories must be a list of StepTrajectory objects")
            else:
                if not isinstance(trajectories, list) or not all(
                        (isinstance(traj, Trajectory) for traj in trajectories)):
                    raise TypeError("Trajectories must be a list of Trajectory objects")
        except (RuntimeError, TypeError) as e:
            raise RuntimeError(f"Failed to get response from API: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during get response from API: {e}") from e

        self.rollout_engine.sleep()
        return trajectories

    def _process_trajectories(self, trajectories):
        trajectories.sort(key=lambda x: x.idx)
        if self.use_stepwise_advantage:
            traj_reward_list = []
            for traj in trajectories:
                traj_reward_list.append(traj.trajectory_reward)
            max_traj_reward = max(traj_reward_list)
            min_traj_reward = min(traj_reward_list)
            mean_traj_reward = sum(traj_reward_list) / len(traj_reward_list)
            self.data_manager.update_metrics("traj_reward/max", value=[float(max_traj_reward)], cumulate=True)
            self.data_manager.update_metrics("traj_reward/min", value=[float(min_traj_reward)], cumulate=True)
            self.data_manager.update_metrics("traj_reward/mean", value=[float(mean_traj_reward)], cumulate=True)
            self._normalize_trajectory_reward(trajectories)
        final_gen_batch_output, metrics = self._transform_agent_trajectories(trajectories)
        responses = final_gen_batch_output["responses"]
        input_ids = final_gen_batch_output["input_ids"]
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
        responses = self.remove_padding_and_split_to_list(responses, pad_token_id, pad_token_id)
        responses_length = [torch.tensor([len(response)]) for response in responses]

        outputs = {
            "responses": responses,
            "input_ids": input_ids,
            "response_length": responses_length,
            "prompt_length": final_gen_batch_output["prompt_length"],
            "rm_scores": final_gen_batch_output["token_level_scores"],
            "token_level_rewards": final_gen_batch_output["token_level_scores"],
            "response_mask": final_gen_batch_output["traj_mask"],
        }
        return outputs, metrics

    async def generate_sequences(self):
        experience_consumer_stage = "actor_rollout"
        experience_columns = []
        experience_columns.extend(self.dataset_additional_keys)
        experience_count = self.actor_rollout_dispatch_size

        tasks = []
        indexes = []
        start_time_defined = False

        while True:
            batch_data, index = await self._get_batch_data(
                experience_consumer_stage, experience_columns, experience_count
            )
            if not index:
                break

            if not start_time_defined:
                start_time = time.time()
                start_time_defined = True

            _tasks = [dict() for _ in range(len(index))]
            batch_data = self.remove_padding_tensor_dict_to_dict(batch_data)

            for i, index_value in enumerate(index):
                _tasks[i]["id"] = index_value
                for key in self.dataset_additional_keys:
                    _tasks[i][key] = self.tokenizer.decode(batch_data[key][i])

            tasks.extend(_tasks)
            indexes.extend(index)

        trajectories = await self._generate_trajectories(tasks)
        outputs, metrics = self._process_trajectories(trajectories)

        self.iteration += 1
        if self.use_stepwise_advantage:
            outputs = RolloutWorker._pad_batch_to_divisor(outputs, self.global_batch_size)
            indexes = [i for i in range(len(outputs["input_ids"]))]
            # reset experience length to padding length
            self.data_manager.reset_experience_len(len(indexes))
        self.data_manager.put_data(outputs, indexes)
        end_time = time.time()
        for k, value in metrics.items():
            if "res_reward" in k or "toolcall_reward" in k:
                self.data_manager.update_metrics(k, value=[float(value)], cumulate=True)

        self.data_manager.update_metrics(
            "timing/rollout", value=[round(end_time, 4), round(start_time, 4)], cumulate=True
        )
        # number of n_samples_per_prompt after padding. batch_len = global_batch_size x n_samples_per_prompt
        return len(indexes) // self.global_batch_size

    @staticmethod
    def _pad_batch_to_divisor(tensor_batch: dict, size_divisor: int):
        """
        Align output tensor batch with global_batch_size before put into data_manager.

        If the tensor is not an integer multiple of global_batch_size, then the beginning of the tensor will be
        padded to the end of the tensor to align it.
        Exception: 'token_level_scores' padding data not considered to compute advantage
        """
        current_len = len(tensor_batch["input_ids"])
        if current_len % size_divisor == 0:
            return tensor_batch

        remaining_pad = size_divisor - current_len % size_divisor
        for key, value in tensor_batch.items():
            if isinstance(value, list):
                tensor_batch[key] = value + value[:remaining_pad]
            else:
                if key == "token_level_scores":
                    size = value[0].size()[0]
                    pad_tensor = torch.zeros(remaining_pad, size)
                else:
                    pad_tensor = value[:remaining_pad]
                tensor_batch[key] = torch.concat([value, pad_tensor], dim=0)
        return tensor_batch

    def _normalize_trajectory_reward(self, trajectories):
        """Normalize trajectory_reward for compute advantages for each agent"""
        scores = torch.tensor(
            [trajectory.trajectory_reward for trajectory in trajectories],
            dtype=torch.float64
        )
        scores = scores.reshape(-1, self.n_parallel_agents)
        scores = (scores - scores.mean(dim=-1, keepdim=True)) / (scores.std(dim=-1, keepdim=True) + 1e-6)
        scores = scores.reshape(-1)
        for i, trajectory in zip(range(len(trajectories)), trajectories):
            trajectory.trajectory_reward = scores[i]

    async def generate_sequences_verl(self, batch=None):
        tasks = []
        reward_model = batch.non_tensor_batch["reward_model"]
        problems = batch.non_tensor_batch["extra_info"]
        idx = len(batch.non_tensor_batch["extra_info"])

        def launch_one_traj_task(idx: int):
            additional_keys_dict = {"question": problems[idx]["question"]}
            additional_keys_dict["ground_truth"] = reward_model[idx]["ground_truth"]
            additional_keys_dict["id"] = problems[idx]["index"]
            return additional_keys_dict

        tasks = [launch_one_traj_task(i) for i in range(idx)]
        trajectories = await self._generate_trajectories(tasks)
        trajectories.sort(key=lambda x: x.idx)

        outputs, metrics = self._transform_agent_trajectories(trajectories)
        from verl import DataProto
        return DataProto.from_dict(tensors=outputs), metrics

    async def generate_validation(self, batch_data, index):
        tasks = [dict() for _ in range(len(index))]
        batch_data = self.remove_padding_tensor_dict_to_dict(batch_data)
        for i, index_value in enumerate(index):
            tasks[i]["id"] = index_value
            for key in self.dataset_additional_keys:
                tasks[i][key] = self.tokenizer.decode(batch_data[key][i])

        trajectories = await self._generate_trajectories(tasks)
        outputs, _ = self._process_trajectories(trajectories)
        batch_reward_tensor = outputs["token_level_rewards"]
        return batch_reward_tensor.sum(-1).detach().cpu()

    def _transform_agent_trajectories(self, trajectories):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.

        Args:
            trajectories (list of dict): List of trajectories to process.

        Returns:
            DataProto: A structured dataset containing input tokens, masks, and rewards.
        """
        all_initial_tokens, all_response_tokens, all_masks, traj_scores, step_nums, chat_completions = \
            self._extract_trajectory_data(trajectories)
        metrics = self.run_trajectories_perf_metric(trajectories)

        prompts_batch = self._pad_sequences(all_initial_tokens, left_pad=True)
        response_batch = self._pad_sequences(all_response_tokens, left_pad=False)
        input_ids_list, prompt_length_list = self._create_input_ids(all_initial_tokens, all_response_tokens)
        if not all_masks:
            traj_mask = torch.where(response_batch != self.tokenizer.pad_token_id, 1, 0)
        else:
            traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks, batch_first=True, padding_value=0)
        if len(response_batch.shape) < 2 or len(prompts_batch.shape) < 2:
            raise ValueError("response_batch and prompts_batch must have at least two dimensions")
        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)
        attention_mask = torch.where(trajectory_batch != self.tokenizer.pad_token_id, 1, 0)
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

        if self.use_stepwise_advantage:
            RolloutWorker._assign_stepwise_scores(score_batch, traj_scores, step_nums, valid_response_length_sequences)
        else:
            for i, traj_score in enumerate(traj_scores):
                last_valid_idx = valid_response_length_sequences[i] - 1
                if RolloutWorker._is_valid_index(score_batch, last_valid_idx):
                    score_batch[i, last_valid_idx] = traj_score

        if self.train_backend == "verl":
            tensor_batch = {
                "input_ids": trajectory_batch,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": response_batch,
                "prompts": prompts_batch,
                "token_level_scores": score_batch,
                "response_mask": traj_mask,
            }
        elif self.train_backend == "mindspeed_rl":
            tensor_batch = {
                "input_ids": input_ids_list,
                "prompt_length": prompt_length_list,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": response_batch,
                "prompts": prompts_batch,
                "token_level_scores": score_batch,
                "traj_mask": traj_mask,
            }
        else:
            raise ValueError(f"Unsupported train backend: {self.train_backend}")

        return tensor_batch, metrics

    @staticmethod
    def _assign_stepwise_scores(score_batch, traj_scores, step_nums, valid_response_length_sequences):
        """Assign trajectory reward at the end of the corresponding step batch of data"""
        step_index = 0
        for i, traj_score in enumerate(traj_scores):
            step_num = step_nums[i]
            for _ in range(step_num):
                last_valid_idx = valid_response_length_sequences[step_index] - 1
                if RolloutWorker._is_valid_index(score_batch, last_valid_idx):
                    score_batch[step_index, last_valid_idx] = traj_score
                step_index += 1

    @staticmethod
    def _is_valid_index(score_batch, index):
        """Check if the index is valid"""
        return 0 <= index < score_batch.shape[1]

    def _extract_trajectory_data(self, trajectories):
        step_nums = []
        all_initial_tokens = []
        all_response_tokens = []
        all_masks = []
        traj_scores = []
        chat_completions = []

        for traj in trajectories:
            if self.use_stepwise_advantage:
                # step mode extract data and convert to tokens
                self._extract_trajectory_step_data(traj, all_initial_tokens, all_response_tokens, step_nums)
            else:
                prompt_tokens = traj.prompt_tokens
                response_tokens = traj.response_tokens

                if prompt_tokens.numel() == 0 or response_tokens.numel() == 0:
                    raise ValueError(
                        f"Both prompt {prompt_tokens.numel()} and response {response_tokens.numel()} "
                        f"of trajectory shouldn't be empty. "
                        f"Please check to make sure the environment is working and the config is correct."
                    )

                all_initial_tokens.append(prompt_tokens)
                all_response_tokens.append(response_tokens)
                all_masks.append(traj.response_masks)
            traj_scores.append(traj.trajectory_reward)
            chat_completions.append(traj.chat_completions)

        return all_initial_tokens, all_response_tokens, all_masks, traj_scores, step_nums, chat_completions

    def _extract_trajectory_step_data(self, trajectory, all_initial_tokens, all_response_tokens, step_nums):
        """Extract chat_completions of trajectory step data and convert to tokens"""
        for step in trajectory.steps:
            chat_completions = step.chat_completions
            prompt = self._parse_messages(chat_completions[:-1], add_generation_prompt=True)
            prompt = torch.tensor(self.tokenizer.encode(prompt, add_special_tokens=False), dtype=torch.long)
            if "content" not in chat_completions[-1]:
                raise ValueError("The response message must have a 'content' attribute.")
            response = chat_completions[-1]["content"]
            response = torch.tensor(self.tokenizer.encode(response, add_special_tokens=False), dtype=torch.long)
            if prompt.numel() == 0 or response.numel() == 0:
                raise ValueError(
                    f"Both prompt {prompt.numel()} and response {response.numel()} "
                    f"of trajectory shouldn't be empty. Please check to make sure the environment is working and the "
                    f"config is correct."
                )
            all_initial_tokens.append(prompt)
            all_response_tokens.append(response)

        step_nums.append(len(trajectory.steps))

    def _parse_messages(self, messages, add_generation_prompt=False):
        """Convert text data to tokens"""
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def _pad_sequences(self, sequences, left_pad=False):
        if left_pad:
            sequences = [torch.flip(seq, dims=[0]) for seq in sequences]
            padded = torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            return padded.flip(dims=[1])
        else:
            return torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

    @staticmethod
    def _create_input_ids(all_initial_tokens, all_response_tokens):
        input_ids_list = []
        prompt_length_list = []
        for prompt, response in zip(all_initial_tokens, all_response_tokens):
            input_ids_list.append(torch.cat((prompt, response), dim=0))
            prompt_length_list.append(torch.tensor([len(prompt)]))
        return input_ids_list, prompt_length_list

    @staticmethod
    def run_trajectories_perf_metric(trajectories):
        if not trajectories:
            raise ValueError("Parameter trajectories cannot be empty")

        if not isinstance(trajectories, list) or len(trajectories) == 0:
            raise TypeError("Parameter trajectories must be a not empty list")

        traj_metrics = []
        for traj in trajectories:
            if not isinstance(traj, Trajectory) or not getattr(traj, "metrics", None):
                raise TypeError("Each trajectory must be a Trajectory and contain 'metrics' key")
            traj_metrics.append(traj.metrics)

        all_keys = set(traj_metrics[0].keys())
        metrics = {}

        # Define a helper function to log and aggregate metrics
        def update_and_aggregate(key, values):
            if not values or len(values) == 0:
                return
            values = np.array(values)
            mean, min_val, max_val = values.mean(), values.min(), values.max()

            metrics.update(
                {
                    f"traj/{key}_mean": mean,
                    f"traj/{key}_min": min_val,
                    f"traj/{key}_max": max_val,
                }
            )

        # Aggregate and log metrics
        for key in all_keys:
            if key == "traj_start_time":
                continue

            raw_values = [d.get(key) for d in traj_metrics]

            flattened = []
            for v in raw_values:
                if v is None:
                    continue
                if isinstance(v, (list, tuple)):
                    flattened.extend(v)
                else:
                    flattened.append(v)

            if flattened:
                update_and_aggregate(key, flattened)

        return metrics