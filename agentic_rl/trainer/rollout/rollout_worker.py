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

from agentic_rl.base.utils.file_utils import FileCheck
from agentic_rl.configs.agentic_rl_config import AgenticRLConfig
from agentic_rl.configs.agentic_rl_config import GenConfig
from agentic_rl.data_manager.data_manager import DataManager
from agentic_rl.runner.agent_engine_wrapper.base import Trajectory
from agentic_rl.runner.infer_adapter.async_server import AsyncServerManager
from agentic_rl.runner.runner_worker import RunnerWorker


def _parse_text(text):
    start_tag = "<|im_start|>"
    end_tag = "<|im_end|>"

    i = 0
    results = []

    while True:
        start = text.find(start_tag, i)
        if start == -1:
            break

        role_start = start + len(start_tag)

        content_end = text.find(end_tag, role_start)
        if content_end == -1:
            break

        before_content_start = text.find('\n', role_start, content_end)
        if before_content_start == -1:
            i = content_end + len(end_tag)
            continue

        lt_pos = text.find('<', role_start, content_end)
        if lt_pos != -1 and lt_pos < before_content_start:
            i = role_start
            continue

        role = text[role_start:before_content_start]
        content = text[before_content_start + 1:content_end]

        results.append((role, content))
        i = content_end + len(end_tag)

    return results


def parse_qwen_messages(prompt, max_length=100000):
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string")

    if len(prompt) > max_length:
        raise ValueError(f"The length of prompt is too long: {len(prompt)} > {max_length}")

    if "<|im_start|>" not in prompt or "<|im_end|>" not in prompt:
        return []

    matches = _parse_text(prompt)

    extracted_messages = []
    for role, content in matches:
        role_clean = role.strip()
        content_clean = content.strip()

        if role_clean in ["system", "user", "assistant"]:
            extracted_messages.append({
                "role": role_clean,
                "content": content_clean
            })

    return extracted_messages


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
        self._param_check()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path, local_files_only=True, weights_only=True
        )
        self.data_manager = DataManager(self.agentic_rl_config.train_backend)

        self.rollout_engine = AsyncServerManager(
            config=self.generate_config,
            agentic_rl_config=self.agentic_rl_config,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            worker_group=worker_group
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
        addresses = self.rollout_engine.server_addresses

        self.runner_worker = RunnerWorker.remote(
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            sampling_params=sampling_params,
            n_parallel_agents=n_parallel_agents,
            max_prompt_length=max_prompt_length,
            max_model_len=self.generate_config.max_model_len,
            agentic_rl_config=self.agentic_rl_config,
            addresses=addresses,
            agent_engine_wrapper_path=self.agentic_rl_config.agent_engine_wrapper_path
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
            self.remove_padding_tensor_dict_to_dict, "remove_padding_tensor_dict_to_dict", Callable)
        RolloutWorker._validate_param(
            self.remove_padding_and_split_to_list, "remove_padding_and_split_to_list", Callable)
        RolloutWorker._validate_param(self.n_parallel_agents, "n_parallel_agents", int, min_val=1)
        RolloutWorker._validate_param(self.max_prompt_length, "max_prompt_length", int, min_val=1)
        RolloutWorker._validate_param(self.actor_rollout_dispatch_size, "actor_rollout_dispatch_size", int, min_val=0)
        RolloutWorker._validate_param(self.simplify_think_content, "simplify_think_content", bool)

    def wait_init_finished(self):
        pass

    def init_data_manager(self, data_manager):
        self.data_manager.sync_init_data_manager(data_manager)

    async def _get_batch_data(self, experience_consumer_stage, experience_columns, experience_count):
        while self.data_manager.all_consumed(experience_consumer_stage) > 0:
            batch_data, index = self.data_manager.get_data(
                experience_consumer_stage,
                experience_columns,
                experience_count
            )
            if not index:
                continue
            return batch_data, index
        return None, None

    def _preprocess_batch_data(self, batch_data):
        batch_data = self.remove_padding_tensor_dict_to_dict(batch_data)
        prompts = [parse_qwen_messages(self.tokenizer.decode(s)) for s in batch_data['prompts']]
        problems = []
        for messages in prompts:
            for content in messages:
                if content['role'] == 'user':
                    problems.append(content['content'])
        return problems

    def _generate_tasks(self, batch_data, problems, index):
        additional_keys_dict = {"question": problems}
        for key in self.dataset_additional_keys:
            decode_list = [self.tokenizer.decode(s) for s in batch_data[key]]
            if "labels" == key:
                additional_keys_dict["ground_truth"] = decode_list
            else:
                additional_keys_dict[key] = decode_list

        tasks = []
        for idx, index_value in enumerate(index):
            task = {
                "id": index_value
            }
            for key in additional_keys_dict.keys():
                if len(additional_keys_dict[key]) <= idx:
                    raise IndexError(
                        f"Data length of key {key} mismatch for idx {idx}, "
                        f"actual length is {len(additional_keys_dict[key])}")
                task[key] = additional_keys_dict[key][idx]
            tasks.append(task)
        return tasks

    async def _generate_trajectories(self, tasks):
        self.rollout_engine.wake_up()
        try:
            trajectories = ray.get(self.runner_worker.generate_agent_trajectories_async.remote(tasks))
            if not isinstance(trajectories, list) or not all((isinstance(traj, Trajectory) for traj in trajectories)):
                raise TypeError("Trajectories must be a list of Trajectory objects")
        except (RuntimeError, TypeError) as e:
            raise RuntimeError(f"Failed to get response from API: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during get response from API: {e}") from e

        self.rollout_engine.sleep()
        return trajectories

    def _process_trajectories(self, trajectories):
        trajectories.sort(key=lambda x: x.idx)
        final_gen_batch_output, metrics = self._transform_agent_trajectories(trajectories)
        responses = final_gen_batch_output['responses']
        input_ids = final_gen_batch_output['input_ids']
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
        responses = self.remove_padding_and_split_to_list(responses, pad_token_id, pad_token_id)
        responses_length = [torch.tensor([len(response)]) for response in responses]

        outputs = {
            'responses': responses,
            'input_ids': input_ids,
            'response_length': responses_length,
            'prompt_length': final_gen_batch_output['prompt_length'],
            'rm_scores': final_gen_batch_output["token_level_scores"],
            'token_level_rewards': final_gen_batch_output["token_level_scores"],
            'response_mask': final_gen_batch_output['traj_mask']
        }
        return outputs, metrics

    async def generate_sequences(self):
        experience_consumer_stage = 'actor_rollout'
        experience_columns = ['prompts', 'prompt_length']
        experience_columns.extend(self.dataset_additional_keys)
        experience_count = self.actor_rollout_dispatch_size

        tasks = []
        indexes = []
        start_time_defined = False

        while True:
            batch_data, index = await self._get_batch_data(experience_consumer_stage, experience_columns,
                                                           experience_count)
            if not index:
                break

            if not start_time_defined:
                start_time = time.time()
                start_time_defined = True

            problems = self._preprocess_batch_data(batch_data)
            tasks.extend(self._generate_tasks(batch_data, problems, index))
            indexes.extend(index)

        trajectories = await self._generate_trajectories(tasks)
        outputs, metrics = self._process_trajectories(trajectories)

        self.iteration += 1
        self.data_manager.put_data(outputs, indexes)
        end_time = time.time()
        for k, value in metrics.items():
            if "res_reward" in k or "toolcall_reward" in k:
                self.data_manager.update_metrics(k, value=[float(value)], cumulate=True)

        self.data_manager.update_metrics("timing/rollout",
                                         value=[round(end_time, 4), round(start_time, 4)],
                                         cumulate=True
                                         )

    def _transform_agent_trajectories(self, trajectories):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.

        Args:
            trajectories (list of dict): List of trajectories to process.

        Returns:
            DataProto: A structured dataset containing input tokens, masks, and rewards.
        """
        all_initial_tokens, all_response_tokens, all_masks, traj_scores, chat_completions = \
            self._extract_trajectory_data(trajectories)
        metrics = self.run_trajectories_perf_metric(trajectories)

        prompts_batch = self._pad_sequences(all_initial_tokens, left_pad=True)
        response_batch = self._pad_sequences(all_response_tokens, left_pad=False)
        input_ids_list, prompt_length_list = self._create_input_ids(all_initial_tokens, all_response_tokens)
        traj_mask = self._pad_sequences(all_masks, left_pad=False)
        if len(response_batch.shape) < 2 or len(prompts_batch.shape) < 2:
            raise ValueError("response_batch and prompts_batch must have at least two dimensions")
        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)
        attention_mask = torch.where(trajectory_batch != self.tokenizer.pad_token_id, 1, 0)
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        prompt_length = response_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

        for i, traj_score in enumerate(traj_scores):
            last_valid_idx = valid_response_length_sequences[i] - 1
            if 0 <= last_valid_idx < score_batch.shape[1]:
                score_batch[i, last_valid_idx] = traj_score

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

        return tensor_batch, metrics

    @staticmethod
    def _extract_trajectory_data(trajectories):
        all_initial_tokens = []
        all_response_tokens = []
        all_masks = []
        traj_scores = []
        chat_completions = []

        for traj in trajectories:
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

        return all_initial_tokens, all_response_tokens, all_masks, traj_scores, chat_completions

    def _pad_sequences(self, sequences, left_pad=False):
        if left_pad:
            sequences = [torch.flip(seq, dims=[0]) for seq in sequences]
            padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True,
                                                     padding_value=self.tokenizer.pad_token_id)
            return padded.flip(dims=[1])
        else:
            return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True,
                                                   padding_value=self.tokenizer.pad_token_id)

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

        flattened_metrics = {k: [d[k] for d in traj_metrics] for k in traj_metrics[0]}
        metrics = {}

        # Define a helper function to log and aggregate metrics
        def update_and_aggregate(key, values):
            if not values or len(values) == 0:
                return
            values = np.array(values)
            mean, min_val, max_val = values.mean(), values.min(), values.max()

            metrics.update({
                f"traj/{key}_mean": mean,
                f"traj/{key}_min": min_val,
                f"traj/{key}_max": max_val,
            })

        # Aggregate and log metrics
        for key, values in flattened_metrics.items():
            if key == "traj_start_time":
                continue

            if key in ["llm_step_times", "env_step_times", "step_reward"]:
                values = [
                    item
                    for sublist in values
                    for item in sublist
                ]
            else:
                values = [v for v in values if v is not None]
            update_and_aggregate(key, values)

        return metrics
