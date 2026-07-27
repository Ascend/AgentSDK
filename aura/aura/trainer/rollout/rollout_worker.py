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
import random
import time
from collections import defaultdict

import numpy as np
import ray
import torch
from transformers import AutoTokenizer

from aura.base.log.loggers import Loggers
from aura.base.misc.misc import app_stats
from aura.base.utils.globals import ROLLOUT_WEIGHTS_PREFIX
from aura.controllers.rollout_controller.rollout_queue import get_rollout_queue_actor
from aura.controllers.utils.utils import DEFAULT_SLEEP_TIME
from aura.data_manager.data_manager import DataManager
from aura.runner.agent_router import AgentRouter
from aura.runner.infer_adapter.async_server import AsyncServerManager, AsyncServerProxyManager

logger = Loggers(__name__).get_logger()

UNAVAILABLE_WEIGHT_VERSION = -1
DEFAULT_BEAM_CORRECT_THRESHOLD = 0.8
DEFAULT_BEAM_PARTIAL_THRESHOLD = 0.0


def get_least_common_multiple(num_1: int, num_2: int):
    return abs(num_1 * num_2) // math.gcd(num_1, num_2)



def generate_dummy_trajectory(prompt_id):
    trajectory = {
        "prompt_tokens": torch.tensor([0]),
        "response_tokens": torch.tensor([0]),
        "response_masks": torch.tensor([1]),
        "trajectory_reward": 0.0,
        "idx": 0,
        "prompt_id": str(prompt_id),
        "chat_completions": [{"role": "system", "content": "0"}],
        "trajectory": {
            "task": {},
            "data_id": "000000000000000000000000000000000",
            "training_id": "20251218230427",
            "epoch_id": 0,
            "iteration_id": 0,
            "sample_id": 1,
            "trajectory_id": "000000000000000000000000000000000-20251218230427-0-0-1-0",
        },
        "metrics": {
            "steps": 1,
            "reward_time": None,
            "env_time": 0.0,
            "llm_time": 0.0,
            "total_time": 0.0,
            "toolcall_reward": 0.0,
            "res_reward": 0.0,
            "env_step_times": [0.0],
            "llm_step_times": [0.0],
        },
    }
    return trajectory


def _cfg_get(config_obj, key, default=None):
    if config_obj is None:
        return default
    if isinstance(config_obj, dict):
        return config_obj.get(key, default)
    return getattr(config_obj, key, default)


def _count_trajectories_per_prompt(trajectories, global_batch_size: int) -> dict:
    counts = defaultdict(int)
    for trajectory in trajectories:
        if isinstance(trajectory, dict):
            prompt_index = trajectory.get("prompt_index", trajectory.get("prompt_id", 0))
        else:
            prompt_index = getattr(trajectory, "prompt_index", getattr(trajectory, "prompt_id", 0))
        counts[int(prompt_index)] = counts.get(int(prompt_index), 0) + 1
    return dict(counts)


def parse_messages(prompt, model_name="qwen"):
    import re

    # Match Qwen ChatML format.
    if "qwen" in model_name:
        pattern = r"<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>"
    elif "deepseek" in model_name:
        pattern = r"<｜(.*?)｜>(.*?)(?=<｜|$)"
    else:
        raise NotImplementedError(f"{model_name} is not supported!")
    matches = re.findall(pattern, prompt, re.DOTALL)

    # Extract role and content.
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


def get_trajectory_prompt_id(trajectory):
    if isinstance(trajectory, dict):
        prompt_id = trajectory.get("prompt_id")
        if prompt_id is None:
            traj_info = trajectory.get("trajectory") or {}
            application_id = traj_info.get("application_id") if isinstance(traj_info, dict) else None
            if application_id:
                prompt_id = application_id.split("-", 1)[0]
        return None if prompt_id is None else str(prompt_id)

    prompt_id = getattr(trajectory, "prompt_id", None)
    if prompt_id is None:
        application_id = getattr(trajectory, "application_id", None)
        if application_id:
            prompt_id = application_id.split("-", 1)[0]
    return None if prompt_id is None else str(prompt_id)


def get_trajectory_group_key(trajectory, fallback):
    prompt_id = get_trajectory_prompt_id(trajectory)
    if prompt_id is not None:
        return prompt_id
    if isinstance(trajectory, dict):
        prompt_index = trajectory.get("prompt_index")
        if prompt_index is not None:
            return str(prompt_index)
        idx = trajectory.get("idx")
        return str(fallback if idx is None else idx)

    prompt_index = getattr(trajectory, "prompt_index", None)
    if prompt_index is not None:
        return str(prompt_index)
    idx = getattr(trajectory, "idx", None)
    return str(fallback if idx is None else idx)


def get_trajectory_reward(trajectory):
    if isinstance(trajectory, dict):
        return trajectory.get("trajectory_reward", trajectory.get("reward", 0.0))
    return getattr(trajectory, "reward", 0.0)


def set_trajectory_reward(trajectory, reward):
    if isinstance(trajectory, dict):
        trajectory["trajectory_reward"] = reward
        return
    trajectory.reward = reward


def _synchronize_and_collect():
    torch.npu.empty_cache()
    gc.collect()
    torch.npu.synchronize()


def pad_dataproto_to_divisor(tensor_batch: dict, size_divisor: int):
    current_len = len(tensor_batch["input_ids"])
    if current_len % size_divisor != 0:
        remaining_pad = size_divisor - current_len % size_divisor
        for key in tensor_batch:
            if isinstance(tensor_batch[key], list):
                tensor_batch[key] = tensor_batch[key] + tensor_batch[key][:remaining_pad]
            else:
                if key == "token_level_scores":
                    size = tensor_batch[key][0].size()[0]
                    pad_tensor = torch.zeros(remaining_pad, size)
                else:
                    pad_tensor = tensor_batch[key][:remaining_pad]
                tensor_batch[key] = torch.concat([tensor_batch[key], pad_tensor], dim=0)
    return tensor_batch


def _batch_size_from_tensor_batch(tensor_batch: dict) -> int:
    """Return batch row count; msrl path stores input_ids as Tensor, not list."""
    input_ids = tensor_batch.get("input_ids")
    if input_ids is None:
        return 0
    if isinstance(input_ids, list):
        return len(input_ids)
    shape = getattr(input_ids, "shape", None)
    if shape:
        return int(shape[0])
    return len(input_ids)


@ray.remote
class RolloutWorker:
    def __init__(
        self,
        generate_config,
        train_backend,
        weight_save_dir,
        trajectory_timeout,
        hybrid_batch_num,
        use_on_policy,
        wait_available_weight_timeout,
        n_parallel_agents=8,
        actor_rollout_dispatch_size=0,
        use_stepwise_advantage=False,
        validate_n_samples=1,
        traj_output_path=None,
        tokenizer_name_or_path=None,
        dataset_additional_keys=None,
        global_batch_size=None,
        agentic_env_config=None,
        trajectory_generation_method="chain",
        worker_group=None,
        remove_padding_tensor_dict_to_dict=None,
        remove_padding_and_split_to_list=None,
        service_mode="train",
        agent_service=None,
        infer_service=None,
        llm_tokenizer_path=None,
    ):
        # ------------------------------------------------
        import signal
        import threading

        _original_signal = signal.signal

        def _noop_signal(*args, **kwargs):
            if threading.current_thread() is not threading.main_thread():
                return
            return _original_signal(*args, **kwargs)

        signal.signal = _noop_signal
        # ------------------------------------------------

        self.generate_config = generate_config

        self.weight_save_dir = weight_save_dir
        self.actor_rollout_dispatch_size = actor_rollout_dispatch_size
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.validate_n_samples = validate_n_samples
        self.traj_output_path = traj_output_path
        logger.info(f"traj_output_path={self.traj_output_path}, "
                    f"tokenizer_name_or_path={tokenizer_name_or_path}, "
                    f"llm_tokenizer_path={llm_tokenizer_path}")

        self.use_stepwise_advantage = use_stepwise_advantage
        self.parallel_state = None

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        self.llm_tokenizer = None
        if llm_tokenizer_path is not None:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_path)

        self.iteration = 0
        self.dataset_additional_keys = dataset_additional_keys
        self.global_batch_size = global_batch_size
        self.agentic_env_config = agentic_env_config
        self.trajectory_generation_method = trajectory_generation_method
        self.beam_train_n_samples = int(_cfg_get(agentic_env_config, "beam_train_n_samples", 0) or 0)
        self.beam_select_seed = int(_cfg_get(agentic_env_config, "beam_select_seed", 0) or 0)
        self.beam_correct_threshold = float(
            _cfg_get(agentic_env_config, "beam_correct_threshold", DEFAULT_BEAM_CORRECT_THRESHOLD)
        )
        self.beam_partial_threshold = float(
            _cfg_get(agentic_env_config, "beam_partial_threshold", DEFAULT_BEAM_PARTIAL_THRESHOLD)
        )
        self.last_new_samples_per_prompt = n_parallel_agents

        # Prefer explicit agentic config, then generate_config fallback.
        self.trajectory_generation_method = str(
            _cfg_get(agentic_env_config, "trajectory_generation_method", self.trajectory_generation_method)
            or _cfg_get(_cfg_get(generate_config, "agent_engine_kwargs", None), "trajectory_generation_method",
                        self.trajectory_generation_method)
            or self.trajectory_generation_method
        ).lower()

        self.service_mode = service_mode
        self.train_backend = train_backend
        self.data_manager = DataManager(train_backend, service_mode)

        self.remove_padding_tensor_dict_to_dict = remove_padding_tensor_dict_to_dict
        self.remove_padding_and_split_to_list = remove_padding_and_split_to_list
        self.n_samples_per_prompt = n_parallel_agents

        logger.info(f"in rollout worker, n_samples_per_prompt={self.n_samples_per_prompt}")
        logger.info(
            f"rollout trajectory_generation_method={self.trajectory_generation_method}, "
            f"beam_train_n_samples={self.beam_train_n_samples}, "
            f"stepwise_layer_pool={self._stepwise_layer_pool_size()}"
        )

        self.rollout_weight_manager = None
        self.current_weights_version = 0

        self.agent_service = agent_service
        self.infer_service = infer_service

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
        self.wait_timeout = wait_available_weight_timeout
        self.terminate_trajectories = 0

        logger.info(f"trajectory_timeout: {self.trajectory_timeout}")

    async def wait_init_finished(self, is_proxy_mode=True):
        if is_proxy_mode:
            self.rollout_engine = AsyncServerProxyManager(
                config=self.generate_config,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
                worker_group=self.worker_group,
                infer_service=self.infer_service
            )
            await self.rollout_engine.init()
            return
        # Separate controller deployment or colocated train/inference mode.
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

    def is_hybrid_mode(self):
        if self.rollout_weight_manager is not None:
            return False
        return True

    async def first_gen_update_model_weights(self, actual_batch_num=1):
        resume_iteration = int(os.getenv("RESUME_ITERATION", '-1'))  # 断点续训的iteration_id
        if resume_iteration > 0:
            if self.train_backend == "verl":
                # verl非0断点续训时，rollout直接更新续训权重版本，不再更新权重
                logger.info(f"=== update model weights version from verl resume iteration {resume_iteration} ===")
                self.current_weights_version = resume_iteration
                ray.get(self.rollout_weight_manager.update_max_version.remote(add_version_num=(actual_batch_num + resume_iteration)))
            else:
                # mindspeed_rl非0断点续训时，rollout需要更新续训权重和续训版本
                logger.info(f"=== update model weights from msrl resume iteration {resume_iteration} ===")
                await self.update_model_weights()
        else:
            # 正常启动时，无需更新权重，仅更新max version
            ray.get(self.rollout_weight_manager.update_max_version.remote(add_version_num=actual_batch_num))

    async def _do_update_model_weights(self, actual_batch_num=1):
        start_time = time.time()
        if self.service_mode == "train" or self.rollout_engine.get_weight_offloaded():
            logger.info(f"=== hybrid train mode or one step off mode first generation, wake up weights ===")
            await self.rollout_engine.wake_up()
            await self._log_actor_preflight()
            if not self.is_hybrid_mode():
                await self.first_gen_update_model_weights(actual_batch_num)
        else:
            logger.info("=== update model weights from train ===")
            await self.update_model_weights(actual_batch_num)
        cost_time = time.time() - start_time
        logger.info(f"==== infer update weights done, e2e cost: {cost_time}, "
                    f"current version: {self.current_weights_version} ===")
        return cost_time

    async def _log_actor_preflight(self):
        if os.getenv("AURA_ACTOR_PREFLIGHT_ENABLED", "").lower() not in ("1", "true", "yes", "on"):
            return
        addresses = getattr(self.rollout_engine, "server_addresses", None)
        logger.info(f"[ACTOR PREFLIGHT] rollout actor server_addresses={addresses}")
        if not addresses:
            logger.warning("[ACTOR PREFLIGHT] no actor server address is available")
            return

        import json as _json
        import urllib.error
        import urllib.request

        preflight_timeout = float(os.getenv("AURA_ACTOR_PREFLIGHT_TIMEOUT", "30"))
        preflight_retries = max(1, int(os.getenv("AURA_ACTOR_PREFLIGHT_RETRIES", "3")))
        preflight_interval_s = float(os.getenv("AURA_ACTOR_PREFLIGHT_INTERVAL_S", "5"))

        def _fetch_models(url):
            with urllib.request.urlopen(url, timeout=preflight_timeout) as response:
                return getattr(response, "status", None), response.read(4096).decode("utf-8", errors="replace")

        for raw_address in addresses:
            if not raw_address:
                logger.warning("[ACTOR PREFLIGHT] empty actor server address")
                continue
            address = str(raw_address)
            if "-" in address and not address.startswith(("http://", "https://")):
                address = address.split("-", 1)[1]
            base_url = address if address.startswith(("http://", "https://")) else f"http://{address}"
            models_url = f"{base_url.rstrip('/')}/v1/models"
            last_error = None
            for attempt in range(1, preflight_retries + 1):
                try:
                    status, payload = await asyncio.to_thread(_fetch_models, models_url)
                    model_ids = []
                    try:
                        model_ids = [
                            str(item.get("id"))
                            for item in _json.loads(payload).get("data", [])
                            if isinstance(item, dict) and item.get("id") is not None
                        ]
                    except Exception:
                        model_ids = []
                    logger.info(
                        f"[ACTOR PREFLIGHT] {models_url} OK status={status} "
                        f"models={model_ids} attempt={attempt}/{preflight_retries}"
                    )
                    last_error = None
                    break
                except (urllib.error.URLError, TimeoutError, OSError) as error:
                    last_error = error
                    if attempt < preflight_retries:
                        logger.warning(
                            f"[ACTOR PREFLIGHT] {models_url} not ready ({type(error).__name__}: {error}), "
                            f"retry {attempt}/{preflight_retries} in {preflight_interval_s}s "
                            f"(hybrid wake_up/KV init may take minutes on NPU)"
                        )
                        await asyncio.sleep(preflight_interval_s)
            if last_error is not None:
                logger.error(
                    f"[ACTOR PREFLIGHT] {models_url} failed after {preflight_retries} attempts: "
                    f"{type(last_error).__name__}: {last_error}"
                )

    async def _do_offload_model_weights(self):
        if self.service_mode == "train":
            logger.info(f"=== hybrid train mode, offload weights ===")
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
            # TODO: 使用llm语言模型解析数据
            tokenizer_handle = self.tokenizer
            if self.llm_tokenizer is not None:
                tokenizer_handle = self.llm_tokenizer
            prompts = [parse_messages(tokenizer_handle.decode(s), model_name=model_name) for s in batch_data['prompts']]
            problems = []
            for messages in prompts:
                for content in messages:
                    if content['role'] == 'user':
                        problems.append(content['content'])

            additional_keys_dict = {"question": problems}
            if self.dataset_additional_keys is not None:
                for key in self.dataset_additional_keys:
                    decode_list = [tokenizer_handle.decode(s) for s in batch_data[key]]
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

        # For non-stepwise tree training, rollout dispatch should align with beam_train_n_samples.
        beam_n = self.beam_train_n_samples
        if self.trajectory_generation_method == "tree" and beam_n > 0 and not self.use_stepwise_advantage:
            if len(indexes) % beam_n != 0:
                logger.warning(
                    f"indexes len {len(indexes)} is not divisible by beam_train_n_samples {beam_n}; "
                    "selection may be partially dropped."
                )

        for task in tasks:
            question = task["question"]
            self.prompt_count[question] = self.prompt_count.get(question, 0) + 1
            if question not in self.prompt_ids.keys():
                self.prompt_ids[question] = len(self.prompt_ids)
            else:
                # Keep duplicate questions in separate prompt groups once their sample count exceeds n_samples_per_prompt.
                if self.prompt_count[question] > self.n_samples_per_prompt:
                    tmp_idx = (self.prompt_count[question] - 1) // self.n_samples_per_prompt
                    question = question + str(tmp_idx)
                    if question not in self.prompt_ids.keys():
                        self.prompt_ids[question] = len(self.prompt_ids)
            task["prompt_id"] = self.prompt_ids[question]
            task["trajectory_generation_method"] = self.trajectory_generation_method

        logger.info(f"generate_sequences with experience consumer stage: {experience_consumer_stage}")
        return tasks, indexes, start_time

    def _is_stepwise_tree_training(self) -> bool:
        return self.use_stepwise_advantage and self.trajectory_generation_method == "tree"

    def _stepwise_layer_pool_size(self) -> int:
        """Expected beam candidates per prompt for one engine layer."""
        if not self._is_stepwise_tree_training():
            return self.n_samples_per_prompt
        beam_size = int(_cfg_get(self.agentic_env_config, "beam_size", 0) or 0)
        per_beam_expand = int(_cfg_get(self.agentic_env_config, "per_beam_expand", 0) or 0)
        if beam_size > 0 and per_beam_expand > 0:
            return beam_size * per_beam_expand
        if self.beam_train_n_samples > 0:
            return self.beam_train_n_samples
        return self.n_samples_per_prompt

    def _samples_per_prompt_for_collection(self) -> int:
        if self._is_stepwise_tree_training():
            return self._stepwise_layer_pool_size()
        return self.n_samples_per_prompt

    def _rollout_batch_concurrency(self, agent_task_count: int, actual_batch_num: int) -> int:
        prompt_count = max(1, int(agent_task_count / max(1, actual_batch_num)))
        if self._is_stepwise_tree_training():
            return prompt_count * self._stepwise_layer_pool_size()
        return prompt_count * self.n_samples_per_prompt

    async def get_agents(self, tasks):
        from aura.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask
        agent_tasks = [
            AgentTask(
                task_id=str(task["id"]),
                sample_id=task["id"] % self.n_samples_per_prompt,
                iteration=self.iteration,
                agent_name=self.agent_service,
                problem=task["question"],
                ground_truth=task["ground_truth"] if "ground_truth" in task else "",
                prompt_id=task["prompt_id"],
                content=task["content"] if "content" in task else "",
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
        """Stream completed trajectory results as they finish."""
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
                if isinstance(result, list):
                    for item in result:
                        yield item
                else:
                    yield result
            except Exception as e:
                logger.error(f"Task failed: {e}")

    def hybrid_mode_metrics_handle(self, metrics, start_time, end_time):
        if not self.is_hybrid_mode():
            return
        for k, value in metrics.items():
            if "res_reward" in k or "toolcall_reward" in k:
                self.data_manager.update_metrics(k, value=[float(value)], cumulate=True)
        self.data_manager.update_metrics("timing/rollout",
                                        value=[round(end_time, 4), round(start_time, 4)],
                                        cumulate=True)

    def reset_trajectory_reward(self, trajectories):
        if not trajectories:
            return
        grouped = defaultdict(list)
        for idx, trajectory in enumerate(trajectories):
            grouped[get_trajectory_group_key(trajectory, idx)].append((idx, trajectory))

        for _, group in grouped.items():
            scores = torch.tensor([get_trajectory_reward(traj) for _, traj in group], dtype=torch.float64)
            if scores.numel() <= 1:
                normalized = torch.zeros_like(scores)
            else:
                std = scores.std(unbiased=False)
                normalized = (
                    (scores - scores.mean()) / (std + 1e-6)
                    if not torch.isnan(std) and std.item() >= 1e-6
                    else torch.zeros_like(scores)
                )
            for local_idx, (origin_idx, _) in enumerate(group):
                set_trajectory_reward(trajectories[origin_idx], normalized[local_idx].item())
        logger.info("reset trajectory_reward finish")

    def stepwise_normalization(self, trajectories):
        if not trajectories:
            return
        if self.use_stepwise_advantage:
            # record original trajectory reward before normalization
            traj_reward_list = []
            for traj in trajectories:
                traj_reward_list.append(get_trajectory_reward(traj))
            max_traj_reward = max(traj_reward_list)
            min_traj_reward = min(traj_reward_list)
            mean_traj_reward = sum(traj_reward_list) / len(traj_reward_list)
            self.data_manager.update_metrics("traj_reward/max", value=[float(max_traj_reward)], cumulate=True)
            self.data_manager.update_metrics("traj_reward/min", value=[float(min_traj_reward)], cumulate=True)
            self.data_manager.update_metrics("traj_reward/mean", value=[float(mean_traj_reward)], cumulate=True)

            # tree + stepwise uses per-layer grouped normalization in compute_advantage;
            # keep raw rewards to preserve expansion-layer semantics.
            if self.trajectory_generation_method != "tree":
                self.reset_trajectory_reward(trajectories)
            return

        if self.trajectory_generation_method == "tree":
            self.reset_trajectory_reward(trajectories)

    def stepwise_pad_datapro(self, final_gen_batch_output):
        if self.use_stepwise_advantage:
            final_gen_batch_output = pad_dataproto_to_divisor(final_gen_batch_output, self.global_batch_size)
            experience_count = _batch_size_from_tensor_batch(final_gen_batch_output)
            if self.is_hybrid_mode():
                self.data_manager.reset_experience_len(experience_count)
            return final_gen_batch_output
        return final_gen_batch_output

    def select_beam_trajectories(self, trajectories, target_per_prompt):
        """
        For beam search: select `target_per_prompt` trajectories per prompt.
        Priority: CORRECT > PARTIALLY_CORRECT > INCORRECT.
        """
        correct_threshold = getattr(self, "beam_correct_threshold", DEFAULT_BEAM_CORRECT_THRESHOLD)
        partial_threshold = getattr(self, "beam_partial_threshold", DEFAULT_BEAM_PARTIAL_THRESHOLD)

        def _stable_trajectory_key(traj):
            if isinstance(traj, dict):
                trajectory = traj.get("trajectory", {})
                application_id = trajectory.get("application_id") if isinstance(trajectory, dict) else None
                return (
                    str(application_id or traj.get("application_id", "")),
                    int(traj.get("prompt_index", traj.get("idx", 0)) or 0),
                    int(traj.get("idx", 0) or 0),
                )
            return (
                str(getattr(traj, "application_id", "")),
                int(getattr(traj, "prompt_index", getattr(traj, "idx", 0)) or 0),
                int(getattr(traj, "idx", 0) or 0),
            )

        groups = defaultdict(list)
        for traj in trajectories:
            pid = int(traj.get("prompt_index", traj.get("idx", 0)) if isinstance(traj, dict) else getattr(traj, "prompt_index", 0))
            groups[pid].append(traj)

        selected = []
        for pid, group in sorted(groups.items()):
            correct, partial, incorrect = [], [], []
            for t in group:
                if isinstance(t, dict):
                    res_r = t.get("metrics", {}).get("res_reward")
                    if res_r is None:
                        res_r = t.get("trajectory_reward", t.get("reward", 0.0))
                else:
                    res_r = getattr(t, "res_reward", getattr(t, "reward", 0.0))
                if res_r >= correct_threshold:
                    correct.append(t)
                elif res_r > partial_threshold:
                    partial.append(t)
                else:
                    incorrect.append(t)

            correct.sort(key=_stable_trajectory_key)
            partial.sort(key=_stable_trajectory_key)
            incorrect.sort(key=_stable_trajectory_key)
            rng = random.Random(getattr(self, "beam_select_seed", 0) + pid)
            rng.shuffle(correct)
            rng.shuffle(partial)
            rng.shuffle(incorrect)

            picked = []
            remaining = target_per_prompt
            for tier in [correct, partial, incorrect]:
                take = min(len(tier), remaining)
                picked.extend(tier[:take])
                remaining -= take
                if remaining <= 0:
                    break
            logger.info(
                f"[beam_select] prompt={pid}: {len(correct)} correct, {len(partial)} partial, "
                f"{len(incorrect)} incorrect -> picked {len(picked)}"
            )
            selected.extend(picked)
        return selected

    def add_output_for_verl(self, final_gen_batch_output, responses, outputs):
        if self.remove_padding_and_split_to_list is not None:
            return
        outputs["responses"] = responses  # verl: tensor, 长短一样
        outputs["input_ids"] = final_gen_batch_output['input_ids']  # verl: pad, 长短一样
        outputs["prompt_ids"] = final_gen_batch_output['prompt_ids']
        outputs["position_ids"] = final_gen_batch_output["position_ids"]
        outputs["attention_mask"] = final_gen_batch_output["attention_mask"]
        outputs["prompts"] = final_gen_batch_output["prompts"]
        outputs["prompt_length"] = final_gen_batch_output["prompt_length"]
        outputs["rm_scores"] = final_gen_batch_output["token_level_scores"]
        outputs["token_level_rewards"] = final_gen_batch_output["token_level_scores"]
        outputs["position_ids"] = final_gen_batch_output["position_ids"]
        outputs["response_mask"] = final_gen_batch_output["traj_mask"]
        if "rollout_log_probs" in final_gen_batch_output:
            outputs["rollout_log_probs"] = final_gen_batch_output["rollout_log_probs"]

    def add_output_for_msrl(self, responses, input_ids, outputs):
        if self.remove_padding_and_split_to_list is None:
            return
        responses_length = [torch.tensor([len(response)]) for response in responses]
        outputs["responses"] = responses
        outputs["input_ids"] = input_ids
        outputs["response_length"] = responses_length

    def handle_full_batch_trajectories(
        self,
        indexes,
        start_time,
        resharding_to_infer,
        trajectories
    ):
        ## 3. 保存数据
        from aura.base.analysis.data_analysis import json_save_data
        json_save_data(trajectories, "trajectories_before_sort", self.iteration)

        trajectories.sort(key=lambda x: x["idx"])

        stepwise_tree_training = self.use_stepwise_advantage and self.trajectory_generation_method == "tree"
        if self.trajectory_generation_method == "tree":
            if stepwise_tree_training:
                per_prompt_counts = _count_trajectories_per_prompt(
                    trajectories, self.global_batch_size or 1)
                # Beams may expand unevenly across prompts (some hit a 2nd layer -> 8,
                # others stay at 4). The actual per-prompt sample count is finalized
                # AFTER padding to a multiple of global_batch_size (see stepwise_td_reset
                # below), so we only log the raw pool here and set the authoritative
                # last_new_samples_per_prompt once the padded TD size is known.
                pool_size = max(1, len(trajectories) // max(1, self.global_batch_size))
                self.last_new_samples_per_prompt = pool_size
                logger.info(
                    f"[beam_stepwise_bypass] iter={self.iteration} "
                    f"using_engine_candidate_pool={len(trajectories)} "
                    f"counts_per_prompt={per_prompt_counts} "
                    f"raw_pool_per_prompt={pool_size}"
                )
            else:
                complete_trajectories = [traj for traj in trajectories if traj.get("collect_reward", False)]
                if complete_trajectories:
                    trajectories = complete_trajectories
                beam_train_n = self.beam_train_n_samples
                if beam_train_n > 0:
                    trajectories = self.select_beam_trajectories(trajectories, beam_train_n)
                self.last_new_samples_per_prompt = max(1, beam_train_n) if beam_train_n > 0 else self.n_samples_per_prompt
        elif self.use_stepwise_advantage:
            self.last_new_samples_per_prompt = max(1, len(trajectories) // max(1, self.global_batch_size))
        else:
            self.last_new_samples_per_prompt = self.n_samples_per_prompt

        self.stepwise_normalization(trajectories)

        final_gen_batch_output, metrics = self._transform_agent_trajectories(trajectories)
        if self.use_stepwise_advantage or self.trajectory_generation_method == "tree":
            final_gen_batch_output = self.stepwise_pad_datapro(final_gen_batch_output)
        responses = final_gen_batch_output['responses']
        input_ids = final_gen_batch_output['input_ids']

        if self.use_stepwise_advantage and isinstance(responses, list):
            mean_resp_len = sum(int(len(r)) for r in responses) / max(len(responses), 1)
            logger.info(
                f"[stepwise_msrl_varlen] iter={self.iteration} samples={len(responses)} "
                f"response_length_mean={mean_resp_len:.1f}"
            )

        if not self.use_stepwise_advantage and self.remove_padding_and_split_to_list is not None:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
            responses = self.remove_padding_and_split_to_list(responses, pad_token_id, pad_token_id)

        outputs = {
            "prompt_length": final_gen_batch_output['prompt_length'],
            "rm_scores": final_gen_batch_output["token_level_scores"],
            "token_level_rewards": final_gen_batch_output["token_level_scores"],
            "response_mask": final_gen_batch_output['traj_mask'],  # 工具输出mask掉了
            "index_in_batch_list": final_gen_batch_output.get("index_in_batch_list", []),
        }
        if self.use_stepwise_advantage:
            outputs["mc_returns"] = final_gen_batch_output.get("mc_returns", final_gen_batch_output["token_level_scores"])
            outputs["idxs"] = final_gen_batch_output.get("idxs", [])
            outputs["index_in_steps_list"] = final_gen_batch_output.get("index_in_steps_list", [])
            outputs["is_last_step"] = final_gen_batch_output.get("is_last_step", [])
        self.add_output_for_verl(final_gen_batch_output, responses, outputs)
        self.add_output_for_msrl(responses, input_ids, outputs)

        logger.info(f"outputs keys: {list(outputs.keys())}")

        self.write_file(trajectories, prefix="trajectories")
        self.write_file(outputs, prefix="outputs")
        self.iteration += 1
        end_time = time.time()

        experience_count = _batch_size_from_tensor_batch(final_gen_batch_output)
        if experience_count <= 0:
            experience_count = len(trajectories)
        if self.trajectory_generation_method == "tree" or self.use_stepwise_advantage:
            # experience_count is the padded TD size (multiple of global_batch_size).
            # The value returned to the trainer must match it exactly, otherwise the
            # ref/actor dispatch size (global_batch_size * new_samples_per_prompt) will
            # not divide the TD max_len and dispatch_transfer_dock_data raises
            # "TD max_len need be divisible by experience_count".
            indexes = [i for i in range(experience_count)]
            gbs = max(1, self.global_batch_size or 1)
            self.last_new_samples_per_prompt = max(1, experience_count // gbs)
            self.data_manager.reset_experience_len(experience_count)
            logger.info(
                f"[stepwise_td_reset] iter={self.iteration - 1} experience_count={experience_count} "
                f"n_samples_per_prompt={self.last_new_samples_per_prompt}"
            )

        rollout_cost = end_time - start_time
        rollout_metrics = _stat_rollout_metrics(rollout_cost, resharding_to_infer, metrics)
        # 异步分离模式, 通过put_data将outputs和rollout metrics传递到训练端
        self.data_manager.put_data(outputs, indexes, rollout_metrics)
        # 共卡模式, 需要将rollout metrics直接写回到td中
        self.hybrid_mode_metrics_handle(metrics, start_time, end_time)

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
        trajectories = []
        for group in traj_groups.values():
            if len(group) == n_sample or (
                self.trajectory_generation_method == "tree" and self.use_stepwise_advantage and len(group) >= n_sample
            ):
                trajectories.extend(group[:n_sample])
        trajectories = trajectories[:concurrency]
        logger.info(f"|perf-stat|rollout| ====finish trajectories: {len(trajectories)}/{concurrency}, "
                    f"terminate trajectories: {self.terminate_trajectories}")
        return trajectories

    def multi_batches_final_handle(self, traj_groups, all_prompt_ids,
                                   concurrency, indexes, start_time, resharding_to_infer,
                                   samples_per_prompt=None):
        if not all_prompt_ids:
            logger.info(f"prompt id is empty, go to next iteration")
            return
        if samples_per_prompt is None:
            samples_per_prompt = self._samples_per_prompt_for_collection()
        logger.info(f"maybe early terminated, traj_groups: {len(traj_groups)}, all_prompt_ids: {len(all_prompt_ids)}")
        trajectories = self.get_train_batch_traj(traj_groups, concurrency, samples_per_prompt)
        clean_traj_groups(traj_groups, all_prompt_ids, trajectories)
        if not trajectories:
            logger.warning(f"skip empty trajectories, go to next iteration")
            return
        # Pad with dummy trajectories when fewer than concurrency results are available.
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
        concurrency = self._rollout_batch_concurrency(len(agent_tasks), actual_batch_num)
        stepwise_tree_training = self._is_stepwise_tree_training()
        samples_per_prompt = self._samples_per_prompt_for_collection()
        mode = "Step" if self.use_stepwise_advantage else "Token"
        logger.info(
            f"|perf-stat|rollout| collection concurrency={concurrency}, "
            f"samples_per_prompt={samples_per_prompt}, mode={mode}"
        )
        result_stream = self.stream_generate_trajectories(
            agent_tasks, agent_router, mode=mode, concurrency=concurrency)
        traj_groups = defaultdict(list)
        all_prompt_ids = get_all_prompt_ids(agent_tasks)
        done_batch_count = 0

        if stepwise_tree_training:
            # stepwise tree: consume the full engine candidate pool once per rollout call.
            collected = []
            async for trajectory in result_stream:
                if trajectory is None:
                    continue
                prompt_id = get_trajectory_prompt_id(trajectory)
                if prompt_id is None:
                    logger.warning(
                        f"skip trajectory without prompt_id, keys: "
                        f"{list(trajectory.keys()) if isinstance(trajectory, dict) else type(trajectory)}"
                    )
                    continue
                if isinstance(trajectory, dict):
                    trajectory["prompt_id"] = prompt_id
                else:
                    trajectory.prompt_id = prompt_id
                collected.append(trajectory)
            if collected:
                per_prompt_counts = _count_trajectories_per_prompt(
                    collected, self.global_batch_size or 1)
                logger.info(
                    f"[beam_stepwise_collect] total={len(collected)}, "
                    f"counts_per_prompt={per_prompt_counts}"
                )
                self.handle_full_batch_trajectories(
                    indexes, start_time, resharding_to_infer, collected)
            return

        async for trajectory in result_stream:
            if trajectory is None:
                continue
            prompt_id = get_trajectory_prompt_id(trajectory)
            if prompt_id is None:
                logger.warning(f"skip trajectory without prompt_id, keys: "
                               f"{list(trajectory.keys()) if isinstance(trajectory, dict) else type(trajectory)}")
                continue
            if isinstance(trajectory, dict):
                trajectory["prompt_id"] = prompt_id
            else:
                trajectory.prompt_id = prompt_id
            traj_groups[prompt_id].append(trajectory)
            logger.info(f"prompt_id: {prompt_id}, group len: {len(traj_groups[prompt_id])}")
            trajectories = self.get_train_batch_traj(traj_groups, concurrency, samples_per_prompt)
            if not self.trajectories_collect_done(trajectories, concurrency, done_batch_count, actual_batch_num):
                continue
            clean_traj_groups(traj_groups, all_prompt_ids, trajectories)
            self.handle_full_batch_trajectories(indexes, start_time, resharding_to_infer, trajectories)
            done_batch_count += 1
            if done_batch_count < actual_batch_num:
                logger.info(f'|perf-stat|rollout| generate_sequences iteration: {self.iteration} begin')
                start_time = time.time()

        # Handle remaining data if truncation ended the loop early.
        self.multi_batches_final_handle(
            traj_groups, all_prompt_ids, concurrency, indexes, start_time, resharding_to_infer,
            samples_per_prompt=samples_per_prompt,
        )

    async def generate_sequences(self, actual_batch_num=1):
        resharding_to_infer = await self._do_update_model_weights(actual_batch_num)
        tasks, indexes, start_time = self.get_data_for_generation()
        agent_tasks, agent_router = await self.get_agents(tasks)
        self.terminate_trajectories = 0
        await self.multi_batches_generate_sequences(
            agent_tasks, agent_router, indexes, start_time, resharding_to_infer, actual_batch_num)
        await agent_router.clear_cache(self.agent_service)
        await self._do_offload_model_weights()
        return self.last_new_samples_per_prompt

    def write_file(self, data_dict, prefix):
        if self.traj_output_path is None:
            return
        os.makedirs(self.traj_output_path, exist_ok=True)

        def convert_to_string(value):
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
        with open(os.path.join(self.traj_output_path, f'rollout_{prefix}_{self.perf_timestamp}.json'), 'a') as f:
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
        idxs = []
        index_in_batch_list = []
        index_in_steps_list = []
        is_last_step_list = []
        cancel_logprobs = False

        for traj in trajectories:
            prompt_index = int(traj.get("prompt_index", traj.get("idx", 0)) if isinstance(traj, dict) else 0)
            if self.use_stepwise_advantage and self.trajectory_generation_method == "tree":
                steps = traj["steps"] if isinstance(traj, dict) else []
                if not steps:
                    continue
                step = steps[-1]
                step_idx = int(traj.get("step_depth", len(steps) - 1))
                prompt_id = traj.get("prompt_id", 0)
                step_scores = traj.get("mc_returns", [])
                trajectory_reward = traj.get("trajectory_reward", traj.get("reward", 0.0))

                prompt_text = step["prompt"]
                response_text = step["response"]
                prompt = torch.tensor(self.tokenizer.encode(prompt_text, add_special_tokens=False), dtype=torch.long)
                all_initial_tokens_list.append(prompt)
                response = torch.tensor(self.tokenizer.encode(response_text, add_special_tokens=False), dtype=torch.long)
                all_response_tokens_list.append(response)
                all_masks_list.append(torch.ones_like(response, dtype=torch.long))
                all_prompt_ids.append(prompt_id)

                score = step_scores[-1] if len(step_scores) > 0 else trajectory_reward
                traj_scores.append(score)
                idxs.append(torch.tensor([traj.get("idx", 0)]))
                index_in_batch_list.append(torch.tensor([prompt_index]))
                index_in_steps_list.append(torch.tensor([step_idx]))
                is_last_step_list.append(torch.tensor([bool(traj.get("is_last_step", True))]))
            elif self.use_stepwise_advantage:
                # step mode
                steps = traj["steps"] if isinstance(traj, dict) else traj.steps
                prompt_id = traj.get("prompt_id", 0) if isinstance(traj, dict) else getattr(traj, "prompt_id", 0)
                step_scores = traj.get("mc_returns", []) if isinstance(traj, dict) else []
                trajectory_reward = (
                    traj.get("trajectory_reward", 0.0)
                    if isinstance(traj, dict)
                    else getattr(traj, "reward", 0.0)
                )
                for step_index, step in enumerate(steps):
                    if isinstance(step, dict):
                        prompt_text = step["prompt"]
                        response_text = step["response"]
                    else:
                        chat_completions = step.chat_completions
                        prompt_text = self.chat_parser.parse(
                            chat_completions[:-1], is_first_msg=True, add_generation_prompt=True
                        )
                        response_text = chat_completions[-1]["content"]

                    prompt = torch.tensor(self.tokenizer.encode(prompt_text, add_special_tokens=False), dtype=torch.long)
                    all_initial_tokens_list.append(prompt)

                    response = torch.tensor(self.tokenizer.encode(response_text, add_special_tokens=False), dtype=torch.long)
                    all_response_tokens_list.append(response)
                    all_masks_list.append(torch.ones_like(response, dtype=torch.long))
                    all_prompt_ids.append(prompt_id)

                    score = step_scores[step_index] if step_index < len(step_scores) else trajectory_reward
                    traj_scores.append(score)
                    idxs.append(torch.tensor([traj.get("idx", 0)] if isinstance(traj, dict) else [0]))
                    index_in_batch_list.append(torch.tensor([prompt_index]))
                    index_in_steps_list.append(torch.tensor([step_index]))
                    is_last_step_list.append(torch.tensor([True]))
            else:
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
                if "logprobs" in traj and len(traj["logprobs"]) != 0 and not cancel_logprobs:
                    all_logprobs_list.append(torch.tensor(traj["logprobs"]))
                else:
                    cancel_logprobs = True
                all_masks_list.append(traj["response_masks"])
                traj_scores.append(traj["trajectory_reward"])
                chat_completions.append(traj["chat_completions"])
                all_prompt_ids.append(prompt_id)
                index_in_batch_list.append(torch.tensor([prompt_index]))

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

        rollout_log_probs_batch = None
        if not cancel_logprobs and all_logprobs_list:
            rollout_log_probs_batch = torch.nn.utils.rnn.pad_sequence(
                all_logprobs_list,
                batch_first=True,
                padding_value=0.0,
            )

        tensor_batch = {
            "input_ids": input_ids_list,
            "prompt_length": prompt_length_list,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "traj_mask": traj_mask,
            "prompt_ids": all_prompt_ids,
            "index_in_batch_list": index_in_batch_list,
        }
        if self.use_stepwise_advantage:
            mc_return_batch = score_batch.clone()
            tensor_batch["mc_returns"] = mc_return_batch
            tensor_batch["idxs"] = idxs
            tensor_batch["index_in_steps_list"] = index_in_steps_list
            tensor_batch["is_last_step"] = is_last_step_list
        if rollout_log_probs_batch is not None:
            tensor_batch["rollout_log_probs"] = rollout_log_probs_batch
        # visualize uses padded 2D responses + traj_mask; must run before msrl list conversion.
        self.visualize_trajectory(tensor_batch)
        if self.use_stepwise_advantage and self.remove_padding_and_split_to_list is not None:
            # msrl + stepwise only; verl keeps padded tensors (add_output_for_verl).
            tensor_batch["input_ids"] = [
                torch.cat((prompt, response), dim=0)
                for prompt, response in zip(all_initial_tokens_list, all_response_tokens_list)
            ]
            tensor_batch["responses"] = list(all_response_tokens_list)
            tensor_batch["response_length"] = [
                torch.tensor([len(response)]) for response in all_response_tokens_list
            ]

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
        from aura.base.misc.misc import colorful_print

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

    def run_trajectories_perf_metric(self, trajectories):
        traj_metrics = []
        metrics = {}
        for traj in trajectories:
            if "metrics" not in traj:
                logger.warning(f"skip trajectory metrics without metrics field, keys: {list(traj.keys())}")
                continue
            if traj["metrics"]["total_time"] == 0.0:
                continue
            traj_metrics.append(traj["metrics"])

        if not traj_metrics:
            return metrics

        # Flatten traj_metrics into a dict of lists
        traj_metrics = {k: [d[k] for d in traj_metrics]
                        for k in traj_metrics[0]}
        # Aggregate metrics (mean, min, max)
        for k, v_list in traj_metrics.items():
            if k == "traj_start_time":
                continue
            if k in ["llm_step_times", "env_step_times", "step_reward"]:
                v_list = [item for sublist in v_list for item in sublist]
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
            weights_version = ray.get(self.rollout_weight_manager.get_weights_version.remote())
            if self.current_weights_version < weights_version:
                break

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
            logger.info(f"|perf-stat|rollout| one_step_off skip update_weights on iteration: {self.iteration}")
            return

        logger.info(f"update_model_weights {actual_batch_num=}")
        weights_version = self._wait_available_version(wait_timeout=self.wait_timeout)
        ray.get(self.rollout_weight_manager.update_max_version.remote(add_version_num=actual_batch_num))

        if weights_version == UNAVAILABLE_WEIGHT_VERSION:
            return

        start_time = time.time()
        weights_path = (self.weight_save_dir +
                        ROLLOUT_WEIGHTS_PREFIX + "/weights_" + str(weights_version))
        logger.info(f"|perf-stat|rollout| start update_weights from {weights_path}")

        _synchronize_and_collect()
        await self.rollout_engine.update_weights(weights_path)
        _synchronize_and_collect()
        self.current_weights_version = weights_version
        logger.info(f"|perf-stat|rollout| infer update_weights done, cost: {time.time() - start_time}, "
                    f"current version: {self.current_weights_version} ===")
