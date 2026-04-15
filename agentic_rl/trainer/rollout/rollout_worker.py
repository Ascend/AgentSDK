#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import gc
import json
import math
import os
import time
from collections import defaultdict

from omegaconf import DictConfig
import numpy as np
import ray
import torch
from transformers import AutoTokenizer

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.misc.misc import app_stats
from agentic_rl.base.utils.globals import ROLLOUT_WEIGHTS_PREFIX
from agentic_rl.controllers.rollout_controller.rollout_queue import get_rollout_queue_actor
from agentic_rl.controllers.utils.utils import DEFAULT_SLEEP_TIME
from agentic_rl.data_manager.data_manager import DataManager
from agentic_rl.runner.agent_router import AgentRouter
from agentic_rl.runner.infer_adapter.async_server import AsyncServerManager, AsyncServerProxyManager

logger = Loggers(__name__).get_logger()

UNAVAILABLE_WEIGHT_VERSION = -1

def get_least_common_multiple(num_1: int, num_2: int):
    return abs(num_1 * num_2) // math.gcd(num_1, num_2)


def generate_dummy_trajectory(prompt_id):
    trajectory = {
        "prompt_tokens": torch.tensor([0]),
        "response_tokens": torch.tensor([0]),
        "response_masks": torch.tensor([1]),
        "trajectory_reward": 0.0,
        "idx": "0",
        "prompt_id": str(prompt_id),
        "chat_completions": [
            {
                "role": "system",
                "content": "0"
            }
        ],
        "trajectory": {
            "task": {},
            "data_id": "000000000000000000000000000000000",
            "training_id": "20251218230427",
            "epoch_id": 0,
            "iteration_id": 0,
            "sample_id": 1,
            "trajectory_id": "000000000000000000000000000000000-20251218230427-0-0-1-0"
        },
        "metrics": {
            "steps": 1,
            "reward_time": None,
            "env_time": 0.0,
            "llm_time": 0.0,
            "total_time": 0.0,
            "toolcall_reward": 0.0,
            "res_reward": 0.0,
            "env_step_times": [
                0.0
            ],
            "llm_step_times": [
                0.0
            ]
        }
    }
    return trajectory

def parse_messages(prompt, model_name="qwen"):
    import re

    # Match Qwen ChatML format
    if "qwen" in model_name:
        pattern = r"<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>"
    elif "deepseek" in model_name:
        pattern = r"<｜(.*?)｜>(.*?)(?=<｜|$)"
    else:
        raise NotImplementedError(f"{model_name} is not supported!")
    matches = re.findall(pattern, prompt, re.DOTALL)

    # Extract roles and content
    extracted_messages = []
    for role, content in matches:
        extracted_messages.append({
            "role": role.strip().lower(),
            "content": content.strip()
        })

    return extracted_messages


def _stat_rollout_metrics(rollout_cost, resharding_to_infer, metrics):
    rollout_metrics = {
        "rollout_cost": rollout_cost,
        "resharding_to_infer": resharding_to_infer
    }
    for k, value in metrics.items():
        if "res_reward" in k or "toolcall_reward" in k:
            actual_key = k.split("/")[1]
            rollout_metrics[f"{actual_key}"] = value
    return rollout_metrics

def clean_traj_groups(traj_groups, all_prompt_ids, trajectories):
    for traj in trajectories:
        try:
            traj_groups[traj['prompt_id']].remove(traj)
            all_prompt_ids.discard(int(traj['prompt_id']))
        except ValueError:
            pass


def get_all_prompt_ids(agent_tasks):
    all_prompt_ids = {task.prompt_id for task in agent_tasks}
    return all_prompt_ids


@ray.remote
class RolloutWorker:
    def __init__(
        self,
        train_backend,
        weight_save_dir,
        trajectory_timeout,
        hybrid_batch_num,
        use_on_policy,
        n_parallel_agents=8,
        max_prompt_length=8192,
        actor_rollout_dispatch_size=0,
        simplify_think_content=False,
        validate_n_samples=1,
        traj_output_path=None,
        tokenizer_name_or_path=None,
        dataset_additional_keys=None,
        global_batch_size=None,
        worker_group=None,
        remove_padding_tensor_dict_to_dict=None,
        remove_padding_and_split_to_list=None,
        service_mode="train",
        agent_service=None,
        infer_service=None
    ):
        # ------------------------------------------------
        import signal
        import threading

        # backup original signal handler
        _original_signal = signal.signal

        def _noop_signal(*args, **kwargs):
            if threading.current_thread() is not threading.main_thread():
                return
            return _original_signal(*args, **kwargs)

        signal.signal = _noop_signal
        # ------------------------------------------------

        self.weight_save_dir = weight_save_dir
        self.actor_rollout_dispatch_size = actor_rollout_dispatch_size
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.validate_n_samples = validate_n_samples
        self.traj_output_path = traj_output_path
        logger.info(f"traj_output_path={self.traj_output_path}")

        self.parallel_state = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        self.iteration = 0
        self.dataset_additional_keys = dataset_additional_keys
        self.global_batch_size = global_batch_size

        self.service_mode = service_mode
        self.data_manager = DataManager(train_backend, service_mode)

        self.remove_padding_tensor_dict_to_dict = remove_padding_tensor_dict_to_dict
        self.remove_padding_and_split_to_list = remove_padding_and_split_to_list
        self.n_samples_per_prompt = n_parallel_agents

        logger.info(f"in rollout worker, n_samples_per_prompt={self.n_samples_per_prompt}")

        # one step off weight update state
        self.rollout_weight_manager = None
        self.current_weights_version = 0

        self.agent_service = agent_service
        self.infer_service = infer_service

        # Timestamp for performance data file naming
        self.perf_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        self.worker_group = worker_group
        self.rollout_engine = None

        self.trajectory_timeout = trajectory_timeout
        self.retry_limit = 3
        self.prompt_ids: dict[str, int] = {}
        self.prompt_count: dict[str, int] = {}
        self.hybrid_batch_num = hybrid_batch_num
        self.use_on_policy = use_on_policy
        if self.use_on_policy and self.hybrid_batch_num > 1:
            raise AssertionError(
                f"Configuration error: hybrid_batch_num={self.hybrid_batch_num} "
                f"must be 1 when use_on_policy={self.use_on_policy}.")
        self.wait_timeout = float(os.getenv("WAIT_AVAILABLE_WEIGHT_TIMEOUT", -1)) if not self.use_on_policy else -1
        self.terminate_trajectories = 0

        logger.info(f"trajectory_timeout: {self.trajectory_timeout}")

    async def wait_init_finished(self, is_proxy_mode=True):
        if is_proxy_mode:
            # Training and inference separated deployment mode, controller managed via ray cluster co-deployment
            self.rollout_engine = AsyncServerProxyManager(
                # config=self.generate_config,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
                worker_group=self.worker_group,
                infer_service=self.infer_service
            )
            await self.rollout_engine.init()
            return
        # 1. Training and inference separated deployment mode, controller deployed separately; 2. Training and inference on same card mode
        self.rollout_engine = AsyncServerManager(
            config=self.generate_config,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            worker_group=self.worker_group
        )

    def init_data_manager(self, data_manager):
        return self.data_manager.sync_init_data_manager(data_manager)

    def data_manager_put_experience(self, batch_dict, index):
        return self.data_manager.put_experience(batch_dict, index)

    def init_weight_manager(self, rollout_weight_manager):
        self.rollout_weight_manager = rollout_weight_manager
        logger.info(f"init rollout_weight_manager")

    async def _do_update_model_weights(self, actual_batch_num=1):
        start_time = time.time()
        if self.service_mode == "train" or self.rollout_engine.get_weight_offloaded():
            logger.info(f"first generation sequence, wake up ori weights ===")
            ray.get(self.rollout_weight_manager.update_max_version.remote(
                add_version_num=actual_batch_num))
            await self.rollout_engine.wake_up()
        else:
            logger.info("update model weights from train ===")
            await self.update_model_weights(actual_batch_num)
        cost_time = time.time() - start_time
        logger.info(f"infer update weights done, e2e cost: {cost_time}, "
                    f"current version: {self.current_weights_version} ===")
        return cost_time

    async def _do_offload_model_weights(self):
        if self.service_mode == "train":
            await self.rollout_engine.sleep()

    def get_data_for_generation(self):
        experience_consumer_stage = 'actor_rollout'
        experience_columns = ['prompts', 'prompt_length']
        if self.dataset_additional_keys is not None:
            experience_columns.extend(self.dataset_additional_keys)
        experience_count = self.actor_rollout_dispatch_size

        start_time_defined = False
        start_time = time.time()
        tasks = []
        indexes = []
        while self.data_manager.all_consumed(experience_consumer_stage) > 0:
            batch_data, index = self.data_manager.get_data(
                experience_consumer_stage,
                experience_columns,
                experience_count
            )
            if not index:
                continue

            # remove pad
            batch_data = self.remove_padding_tensor_dict_to_dict(batch_data)
            if not start_time_defined:
                start_time = time.time()
                start_time_defined = True
            model_name = self.tokenizer.name_or_path.lower()
            prompts = [parse_messages(self.tokenizer.decode(s), model_name=model_name) for s in batch_data['prompts']]
            problems = []
            for messages in prompts:
                for content in messages:
                    if content['role'] == 'user':
                        problems.append(content['content'])

            additional_keys_dict = {"question": problems}
            if self.dataset_additional_keys is not None:
                for key in self.dataset_additional_keys:
                    decode_list = [self.tokenizer.decode(s) for s in batch_data[key]]
                    if "labels" == key:
                        additional_keys_dict["ground_truth"] = decode_list
                    else:
                        additional_keys_dict[key] = decode_list

            for i in range(len(index)):
                task = {
                    "id": index[i]
                }
                for key in additional_keys_dict.keys():
                    task[key] = additional_keys_dict[key][i]
                tasks.append(task)
            indexes.extend(index)

        for task in tasks:
            question = task["question"]
            self.prompt_count[question] = self.prompt_count.get(question, 0) + 1
            if question not in self.prompt_ids.keys():
                self.prompt_ids[question] = len(self.prompt_ids)
            else:
                # If there are duplicate question data, additional processing is needed
                if self.prompt_count[question] > self.n_samples_per_prompt:
                    tmp_idx = (self.prompt_count[question] - 1) // self.n_samples_per_prompt
                    question = question + str(tmp_idx)
                    if question not in self.prompt_ids.keys():
                        self.prompt_ids[question] = len(self.prompt_ids)
            task["prompt_id"] = self.prompt_ids[question]

        logger.info(f'generate_sequences with experience consumer stage: '
                    f'{experience_consumer_stage}, and tasks: {tasks}')
        return tasks, indexes, start_time

    async def get_agents(self, tasks):
        from agentic_rl.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask
        agent_tasks = [
            AgentTask(
                task_id=str(task["id"]),
                sample_id=task["id"] % self.n_samples_per_prompt,
                iteration=self.iteration,
                agent_name=self.agent_service,
                problem=task["question"],
                ground_truth=task["ground_truth"] if "ground_truth" in task else "",
                prompt_id=task["prompt_id"],
                content=task["content"] if "content" in task else "", # Required for dtn_code scenario
                extra_args={key: value for key, value in task.items() if key not in ["id", "question", "ground_truth", "prompt_id", "content"]}
            )
            for task in tasks
        ]
        agent_router = await AgentRouter.create()
        return agent_tasks, agent_router

    async def early_termination_requests(self, task, agent_router):
        logger.warning(f">>> long trajectory, early termination: {task}")
        await agent_router.cancel_request(task)
        self.terminate_trajectories += 1
        rollout_queue_actor = get_rollout_queue_actor()
        rollout_queue_actor.add_abort_queue.remote(task)

    async def stream_generate_trajectories(self, agent_tasks, agent_router, mode="Text", concurrency=64):
        """Return completed task results in real-time streaming mode"""
        semaphore = asyncio.Semaphore(concurrency)
        async def worker_with_retry(task):
            retry_count = 0
            while retry_count < self.retry_limit:
                try:
                    async with semaphore:
                        task_result = await asyncio.wait_for(
                            agent_router.generate_trajectory(
                                task=task, mode=mode, addresses=self.rollout_engine.server_addresses),
                            timeout=self.trajectory_timeout
                        )
                    return task_result
                except asyncio.TimeoutError:
                    logger.warning(f"generate trajectory timeout, task id: {task.task_id}, prompt id: {task.prompt_id} "
                                   f"after {self.trajectory_timeout}s, early termination.")
                    await self.early_termination_requests(task, agent_router)
                    return None
                except Exception as exp:
                    retry_count += 1
                    logger.warning(f"generate trajectory failed task: {task.task_id} prompt_id: {task.prompt_id}, "
                                   f"retrying ({retry_count}/{self.retry_limit}), exp: {exp}")
            raise Exception(f"generate_agent_trajectory Task failed after {self.retry_limit} retries.")

        futures = [asyncio.create_task(worker_with_retry(task)) for task in agent_tasks]
        for future in asyncio.as_completed(futures):
            try:
                result = await future
                logger.info(f">>> get worker future")
                yield result
            except Exception as e:
                logger.error(f"Task failed: {e}")

    def handle_full_batch_trajectories(
        self,
        indexes,
        start_time,
        resharding_to_infer,
        trajectories
    ):
        trajectories.sort(key=lambda x: x["idx"])
        final_gen_batch_output, metrics = self._transform_agent_trajectories(trajectories)

        responses = final_gen_batch_output['responses']
        input_ids = final_gen_batch_output['input_ids']
        prompt_ids = final_gen_batch_output['prompt_ids']

        outputs = {
            'responses': responses,  # list with varying lengths, actual length (all subsequent rounds)
            'input_ids': input_ids,  # no padding, varying lengths, contains prompt (initial) and responses (all subsequent rounds)
            "prompt_ids": prompt_ids,
            'prompt_length': final_gen_batch_output['prompt_length'],
            'rm_scores': final_gen_batch_output["token_level_scores"],
            'token_level_rewards': final_gen_batch_output["token_level_scores"],
            'position_ids': final_gen_batch_output["position_ids"],
            'prompts': final_gen_batch_output["prompts"],
            'rollout_log_probs': final_gen_batch_output["rollout_log_probs"],
            'attention_mask': final_gen_batch_output["attention_mask"],
            'response_mask': final_gen_batch_output['traj_mask']  # Tool outputs are masked
        }

        self.write_file(trajectories, prefix="trajectories")
        self.write_file(outputs, prefix="outputs")
        self.iteration += 1
        end_time = time.time()

        rollout_cost = end_time - start_time
        rollout_metrics = _stat_rollout_metrics(rollout_cost, resharding_to_infer, metrics)
        self.data_manager.put_data(outputs, indexes, rollout_metrics)
        logger.info(f'|perf-stat|rollout| rollout worker put_data iteration-{self.iteration} to train')
        logger.info(f"|perf-stat|rollout| ===rollout iteration: {self.iteration}, "
                    f"timing/rollout : {time.time() - start_time:.4f}===")
        app_stats.print(self.iteration)

    def trajectories_collect_done(self, trajectories, concurrency, done_batch_count, actual_batch_num):
        if len(trajectories) < concurrency:
            if (done_batch_count + 1) == actual_batch_num:
                if len(trajectories) + self.terminate_trajectories >= concurrency:
                    return True
            return False
        return True
    
    def get_train_batch_traj(self, traj_groups, concurrency: int, n_sample: int = 8):
        trajectories = [traj for group in traj_groups.values() if len(group) == n_sample for traj in group][
                       :concurrency]
        logger.info(f"|perf-stat|rollout| ====finish trajectories: {len(trajectories)}/{concurrency}, "
                    f"terminate trajectories: {self.terminate_trajectories}")
        return trajectories

    def multi_batches_final_handle(self, traj_groups, all_prompt_ids,
                                   concurrency, indexes, start_time, resharding_to_infer):
        if not all_prompt_ids:
            logger.info(f"prompt id is empty, go to next iteration")
            return
        logger.info(f"maybe early terminated, traj_groups: {len(traj_groups)}, all_prompt_ids: {len(all_prompt_ids)}")
        trajectories = self.get_train_batch_traj(traj_groups, concurrency, self.n_samples_per_prompt)
        clean_traj_groups(traj_groups, all_prompt_ids, trajectories)
        if not trajectories:
            # No available trajectories, skip to next iteration
            logger.warning(f"skip empty trajectories, go to next iteration")
            return
        # Insufficient concurrency data, need to pad with dummy data, otherwise training will hang if it can't collect a complete td data
        if len(trajectories) < concurrency:
            for prompt_id in all_prompt_ids:
                for _ in range(self.n_samples_per_prompt):
                    traj = generate_dummy_trajectory(prompt_id)
                    trajectories.append(traj)
                if len(trajectories) == concurrency:
                    break
        logger.info(f"|perf-stat|rollout| ====finish trajectories: {len(trajectories)}/{concurrency}, "
                    f"terminate trajectories: {self.terminate_trajectories}")
        self.handle_full_batch_trajectories(indexes, start_time, resharding_to_infer, trajectories)

    async def multi_batches_generate_sequences(
        self,
        agent_tasks,
        agent_router,
        indexes,
        start_time,
        resharding_to_infer,
        actual_batch_num
    ):
        logger.info(f'|perf-stat|rollout| generate_sequences iteration: {self.iteration} begin, '
                    f'tasks: {len(agent_tasks)}, actual_batch_num: {actual_batch_num}')
        concurrency = int(len(agent_tasks) / actual_batch_num)
        result_stream = self.stream_generate_trajectories(
            agent_tasks, agent_router, mode='Token', concurrency=concurrency)
        traj_groups = defaultdict(list)
        all_prompt_ids = get_all_prompt_ids(agent_tasks)
        done_batch_count = 0
        async for trajectory in result_stream:
            if trajectory is None:
                continue
            prompt_id = trajectory['prompt_id']
            traj_groups[prompt_id].append(trajectory)
            logger.info(f"prompt_id: {prompt_id}, group len: {len(traj_groups[prompt_id])}")
            # Process immediately whenever concurrency results are collected
            trajectories = self.get_train_batch_traj(traj_groups, concurrency, self.n_samples_per_prompt)
            if not self.trajectories_collect_done(trajectories, concurrency, done_batch_count, actual_batch_num):
                continue
            # Clear groups and enter next batch collection
            clean_traj_groups(traj_groups, all_prompt_ids, trajectories)
            self.handle_full_batch_trajectories(indexes, start_time, resharding_to_infer, trajectories)
            done_batch_count += 1
            if done_batch_count < actual_batch_num:
                logger.info(f'|perf-stat|rollout| generate_sequences iteration: {self.iteration} begin')
                # Update start time for next batch
                start_time = time.time()

        # Loop may exit early due to truncation, need to process data one final time
        self.multi_batches_final_handle(traj_groups, all_prompt_ids,
                                        concurrency, indexes, start_time, resharding_to_infer)

    async def generate_sequences(self, actual_batch_num=1):
        tasks, indexes, start_time = self.get_data_for_generation()
        agent_tasks, agent_router = await self.get_agents(tasks)
        self.terminate_trajectories = 0
        resharding_to_infer = await self._do_update_model_weights(actual_batch_num)
        await self.multi_batches_generate_sequences(
            agent_tasks, agent_router, indexes, start_time, resharding_to_infer, actual_batch_num)
        await agent_router.clear_cache(self.agent_service)
        await self._do_offload_model_weights()

    def write_file(self, data_dict, prefix):
        # Convert tensor to string
        # # TODO: Store original Tensor and decoded content in file
        def convert_to_string(value):
            if isinstance(value, torch.Tensor):
                return str(value.tolist())  # Convert tensor to string
            elif isinstance(value, list):
                return [convert_to_string(v) for v in value]  # Recursively process list
            elif isinstance(value, dict):
                return {key: convert_to_string(v) for key, v in value.items()}  # Recursively process dict
            else:
                return str(value)  # Directly return if other type

        add_iter = {"iteration": self.iteration, f"{prefix}": data_dict}
        data_str = convert_to_string(add_iter)
        # Write dictionary to JSON file
        with open(os.path.join(self.traj_output_path, f'rollout_{prefix}_{self.perf_timestamp}.json'), 'a') as f:
            # Save with indent=4 formatting
            # noinspection PyTypeChecker
            json.dump(data_str, f, indent=4, ensure_ascii=False)
            f.write('\n')
            logger.info(f'write_file rollout_{prefix}_{self.perf_timestamp}.json in iteration {self.iteration} done')

    async def generate_validation(self, batch, index):
        model_name = self.tokenizer.name_or_path.lower()
        prompts = [parse_messages(self.tokenizer.decode(s), model_name=model_name) for s in batch['prompts']]
        problems = []
        for messages in prompts:
            for content in messages:
                if content['role'] == 'user':
                    problems.append(content['content'])

        additional_keys_dict = {"question": problems}
        for key in self.dataset_additional_keys:
            decode_list = [self.tokenizer.decode(s) for s in batch[key]]
            if "labels" == key:
                additional_keys_dict["ground_truth"] = decode_list
            else:
                additional_keys_dict[key] = decode_list

        tasks = []
        for i in range(len(index)):
            task = {
                "id": index[i]
            }
            for key in additional_keys_dict.keys():
                task[key] = additional_keys_dict[key][i]
            tasks.append(task)

        agent_tasks, agent_router = await self.get_agents(tasks)

        await self._do_update_model_weights()
        trajectories = await agent_router.generate_trajectories(agent_tasks, mode='Token')
        await self._do_offload_model_weights()

        trajectories.sort(key=lambda x: x["idx"])

        keys_to_remove = {"prompt_tokens", "response_tokens", "response_masks"}
        trajectories_without_remove_keys = [{k: v for k, v in traj_dict.items() if k not in keys_to_remove} for
                                            traj_dict in trajectories]
        self.write_file(trajectories_without_remove_keys, prefix="val_trajs")

        final_gen_batch_output, metrics = self._transform_agent_trajectories(trajectories)

        batch_reward_tensor = final_gen_batch_output["token_level_scores"]
        # Sum scores for each token in the last dimension
        return batch_reward_tensor.sum(-1).detach().cpu(), [item["id"] for item in tasks]

    def _transform_agent_trajectories(self, trajectories):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.

        Args:
            trajectories (list of dict): List of trajectories to process.

        Returns:
            DataProto: A structured dataset containing input tokens, masks, and rewards.
        """

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
            # test if trajectory is empty
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

        metrics = self.run_trajectories_perf_metric(trajectories)

        # reverse the list and create tensors, pad, then flip to achieve left padding
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])

        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        rollout_log_probs_batch = torch.nn.utils.rnn.pad_sequence(
            all_logprobs_list,
            batch_first=True,
            padding_value=0.0,
        )

        input_ids_list = torch.concat([prompts_batch, response_batch], dim=1)

        prompt_length_list = []
        for prompt in all_initial_tokens_list:
            prompt_length_list.append(torch.tensor([len(prompt)]))

        traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)
        attention_mask = torch.where(trajectory_batch != self.tokenizer.pad_token_id, 1, 0)

        # Compute position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

        for i, traj_score in enumerate(traj_scores):
            last_valid_idx = valid_response_length_sequences[i] - 1
            if 0 <= last_valid_idx < score_batch.shape[1]:
                score_batch[i, last_valid_idx] = traj_score
        tensor_batch = {
            "input_ids": input_ids_list,  # no padding, varying lengths
            "prompt_length": prompt_length_list,  # prompt length
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,  # right padded
            "prompts": prompts_batch,  # left padded
            "token_level_scores": score_batch,  # right padded, only the length position has a score, others are 0
            "traj_mask": traj_mask,  # same shape as responses, right padded with 0
            "rollout_log_probs": rollout_log_probs_batch,
            "prompt_ids": all_prompt_ids
        }

        self.visualize_trajectory(tensor_batch)

        return tensor_batch, metrics
    
    def visualize_trajectory(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="traj_mask"):
        """
        Visualize the trajectory from tensor_batch by de-tokenizing prompts and responses,
        and highlighting the masked parts with color.

        Args:
            tensor_batch: The tensor batch containing trajectory data
            sample_idx: Starting index of samples to visualize
            max_samples: Maximum number of samples to visualize
            mask_key: mask key
        """
        from agentic_rl.base.misc.misc import colorful_print

        # Get the relevant tensors
        prompts = tensor_batch["prompts"]
        responses = tensor_batch["responses"]
        traj_mask = tensor_batch[mask_key]
        token_level_scores = tensor_batch["token_level_scores"]

        batch_size = prompts.shape[0]
        end_idx = min(sample_idx + max_samples, batch_size)

        for i in range(sample_idx, end_idx):
            # Detokenize response with color highlighting for masked tokens
            response_tokens = responses[i]
            response_mask = traj_mask[i]

            # Get non-padding tokens
            valid_indices = response_tokens != self.tokenizer.pad_token_id
            valid_response_tokens = response_tokens[valid_indices]
            valid_response_mask = response_mask[valid_indices]

            # Then show token-by-token with masking
            # colorful_print("Response with masking:", fg="yellow", bold=True)

            for j, (token, mask) in enumerate(zip(valid_response_tokens, valid_response_mask, strict=False)):
                token_text = self.tokenizer.decode(token)

                # Check if this token has a reward
                has_reward = token_level_scores[i, j] != 0

                # Apply different colors based on mask and rewards
                if mask == 0:
                    # Masked token (not used in training)
                    colorful_print(token_text, fg="red", end="")
                elif has_reward:
                    # Token with reward
                    colorful_print(token_text, bg="green", end="")

                    reward_info = ""
                    if has_reward:
                        reward_info += f" R:{token_level_scores[i, j].item():.2f}"

                    colorful_print(reward_info, fg="magenta", end="")
                else:
                    # Normal token used in training
                    colorful_print(token_text, fg="blue", end="")

            # Print reward summary
            total_reward = token_level_scores[i].sum().item()
            colorful_print(f"Rewards: {total_reward:.2f}", fg="green", bold=True)

    # Trajectory metric and performance data statistics
    def run_trajectories_perf_metric(self, trajectories):
        traj_metrics = []
        metrics = {}
        for traj in trajectories:
            # Remove metrics for dummy trajectories
            if traj["metrics"]["total_time"] == 0.0:
                continue
            traj_metrics.append(traj["metrics"])

        # Flatten traj_metrics into a dict of lists
        traj_metrics = {k: [d[k] for d in traj_metrics]
                        for k in traj_metrics[0]}
        # Aggregate metrics (mean, min, max)
        for k, v_list in traj_metrics.items():
            if k == "traj_start_time":
                continue
            if k in ["llm_step_times", "env_step_times", "step_reward"]:
                v_list = [
                    item
                    for sublist in v_list
                    for item in sublist
                ]
                v_list = np.array(v_list)
                logger.info(
                    f"iteration {self.iteration} traj/{k}_mean: {v_list.mean()} || "
                    f"traj/{k}_min: {v_list.min()} || traj/{k}_max: {v_list.max()}")
            else:
                # fix: reward may negative
                v_list = [v for v in v_list if v is not None]
                if not v_list:
                    continue
                v_list = np.array(v_list)
                metrics.update(
                    {
                        f"traj/{k}_mean": v_list.mean(),
                        f"traj/{k}_min": v_list.min(),
                        f"traj/{k}_max": v_list.max(),
                    }
                )
                if k in ["env_time", "llm_time", "total_time"]:
                    logger.info(
                        f"iteration {self.iteration} traj/{k}_mean: {v_list.mean()} || "
                        f"traj/{k}_min: {v_list.min()} || traj/{k}_max: {v_list.max()}")
        return metrics

    def _wait_available_version(self, wait_timeout=0):
        start_time = time.time()
        logger.info(f"|perf-stat|rollout| start to detect available weights for iteration: {self.iteration}")
        while True:
            # Get the latest trainable weight version after training
            weights_version = ray.get(self.rollout_weight_manager.get_weights_version.remote())
            if self.current_weights_version < weights_version:
                break

            # Optional timeout judgment
            if 0 <= wait_timeout < (time.time() - start_time):
                weights_version = UNAVAILABLE_WEIGHT_VERSION
                logger.info(f"Waiting for weights update timed out after {wait_timeout} seconds")
                break
            time.sleep(DEFAULT_SLEEP_TIME)
        logger.info(f"|perf-stat|rollout| end waiting available weights for iteration: {self.iteration}, "
                    f"version: {weights_version}/{self.current_weights_version}")
        return weights_version

    async def update_model_weights(self, actual_batch_num=1):
        if not self.use_on_policy and self.iteration == 1:
            # After the first iteration, if entering update judgment, one_step_off weight is 0, no need to wait for weight update
            logger.info(f"|perf-stat|rollout| one_step_off skip update_weights before iteration: {self.iteration}")
            return
        weights_version = self._wait_available_version(wait_timeout=self.wait_timeout)
        logger.info(f"update_model_weights {actual_batch_num=}")
        ray.get(self.rollout_weight_manager.update_max_version.remote(add_version_num=actual_batch_num))
        if weights_version == UNAVAILABLE_WEIGHT_VERSION:
            return

        start_time = time.time()
        weights_path = (self.weight_save_dir + ROLLOUT_WEIGHTS_PREFIX + "/weights_" + str(weights_version))
        logger.info(f"|perf-stat|rollout| start update_weights from {weights_path}")

        torch.npu.empty_cache()
        gc.collect()
        # torch.npu.synchronize()

        await self.rollout_engine.update_weights(weights_path)

        gc.collect()
        # torch.npu.synchronize()
        self.current_weights_version = weights_version
        cost_time = time.time() - start_time
        logger.info(f"|perf-stat|rollout| infer update_weights done, cost: {cost_time}, "
                    f"current version: {self.current_weights_version} ===")
        