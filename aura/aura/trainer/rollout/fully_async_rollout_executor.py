#!/usr/bin/env python3
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

import time
import ray

from aura.base.exceptions.exceptions import RolloutShutdownException
from aura.base.log.loggers import Loggers
from aura.controllers.rollout_controller.rollout_queue import get_rollout_queue_actor
from aura.controllers.utils.utils import MIN_SLEEP_TIME, TRAIN_CONTROLLER_NAMESPACE

logger = Loggers(__name__).get_logger()

INIT_WAIT_TIMES = 100


class FullyAsyncRolloutExecutor:
    """Rollout executor for the fully-async mode."""

    def __init__(
        self, controller, rollout_worker, train_iters, padding_dict_to_tensor_dict, put_prompts_experience, **kwargs
    ):
        """
        Initialize the FullyAsyncRolloutExecutor.

        Args:
            controller: Train controller used to publish weights and finish rollout.
            rollout_worker: Rollout worker that generates trajectories in a fully-async manner.
            train_iters: Total number of training iterations driving the generation loop.
            padding_dict_to_tensor_dict: Callable that converts a raw dict into padded tensor dicts.
            put_prompts_experience: Callable that builds (batch_dict, indexes) from raw prompt batches.
            **kwargs: Additional rollout config; expected to contain ``data_optimized``,
                ``dataset_additional_keys``, ``n_samples_per_prompt`` and ``hybrid_batch_num``.
        """
        required_kwargs = ("data_optimized", "dataset_additional_keys", "n_samples_per_prompt", "hybrid_batch_num")
        missing_kwargs = [key for key in required_kwargs if key not in kwargs]
        if missing_kwargs:
            raise ValueError(f"Missing required rollout config keys: {missing_kwargs}")

        self.controller = controller
        self.rollout_worker = rollout_worker
        self.queue_actor = get_rollout_queue_actor()
        self.dispatch_actor = ray.get_actor("dispatch", namespace=TRAIN_CONTROLLER_NAMESPACE)
        self.train_iters = train_iters
        self.padding_dict_to_tensor_dict = padding_dict_to_tensor_dict
        self.put_prompts_experience = put_prompts_experience

        self.data_optimized = kwargs["data_optimized"]
        self.dataset_additional_keys = kwargs["dataset_additional_keys"] + ["response_mask"]
        self.n_samples_per_prompt = kwargs["n_samples_per_prompt"]
        self.hybrid_batch_num = kwargs["hybrid_batch_num"]
        self.sleep_time = MIN_SLEEP_TIME

        ray.get(self.rollout_worker.init_weight_manager.remote(self.controller.get_weight_manager()))

    def get_batch_dict(self, batch):
        """
        Preprocess a raw merged prompt batch into the format required by the data manager.

        Args:
            batch: Merged raw prompt batch (dict of lists) to be preprocessed.

        Returns:
            tuple: (batch_dict, indexes), where ``batch_dict`` is the padded tensor dict
            and ``indexes`` maps samples to their prompt groups, used by
            ``data_manager_put_experience``.
        """
        if self.data_optimized:
            from aura.trainer.rollout.rollout_dataset import optimized_preprocess_input
            from aura.trainer.rollout.rollout_dataset import optimized_put_prompt_experience

            mini_batches, prompt_ids = optimized_preprocess_input(batch)
            batch_dict, indexes = optimized_put_prompt_experience(
                mini_batches, prompt_ids, self.padding_dict_to_tensor_dict
            )
        else:
            batch_dict, indexes = self.put_prompts_experience(
                batch, self.n_samples_per_prompt, self.dataset_additional_keys
            )
        return batch_dict, indexes

    @staticmethod
    def merge_batch_list(batches):
        """
        Merge multiple prompt batches into a single batch by extending each field.

        Args:
            batches: List of raw prompt batches popped from the rollout queue.

        Returns:
            dict: A merged batch where every field value is the concatenation of the
            corresponding values across the input batches. Returns an empty dict if
            ``batches`` is empty.
        """
        merged_batch = {}
        if not batches:
            return merged_batch
        keys = batches[0].keys()
        for key in keys:
            merged_batch[key] = []
            for batch in batches:
                merged_batch[key].extend(batch[key])
        return merged_batch

    def fit(self):
        """
        Run the main fully-async rollout generation loop.

        Repeatedly waits until the rollout queue has enough prompt batches, then per
        iteration:
        1. Pops up to ``hybrid_batch_num`` batches and refills the queue via the
           DispatchActor;
        2. Merges and preprocesses them, then puts them into the shared data manager;
        3. Triggers the RolloutWorker to generate trajectories (streaming samples back
           to the SampleQueue).

        The loop runs until ``train_iters`` is reached or the queue is shut down, and
        finally notifies the train controller to finish rollout.
        """
        iteration = 0
        logger.info(f"start fully_async rollout loop, iteration: {iteration}/{self.train_iters} ...")

        init_wait_times = INIT_WAIT_TIMES
        while iteration < self.train_iters and not ray.get(self.queue_actor.is_shutdown.remote()):
            queue_size = ray.get(self.queue_actor.queue_size.remote())
            is_running = ray.get(self.queue_actor.is_running.remote())
            if queue_size <= 0 or not is_running:
                time.sleep(self.sleep_time)
                continue
            if queue_size < self.hybrid_batch_num and init_wait_times > 0:
                time.sleep(self.sleep_time)
                init_wait_times -= 1
                continue

            start_time = time.time()
            actual_batch_num = min(self.hybrid_batch_num, queue_size)
            if iteration + actual_batch_num > self.train_iters:
                actual_batch_num = (iteration + actual_batch_num) - self.train_iters

            batch_list = [ray.get(self.queue_actor.pop_queue.remote()) for _ in range(actual_batch_num)]
            self.dispatch_actor.send_batch_groups.remote(actual_batch_num)
            logger.info(f"|perf-stat|rollout| fully_async refilled batch_queue with {actual_batch_num} groups")
            batch = self.merge_batch_list(batch_list)

            batch_dict, indexes = self.get_batch_dict(batch)
            ray.get(self.rollout_worker.data_manager_put_experience.remote(batch_dict=batch_dict, index=indexes))

            try:
                ray.get(self.rollout_worker.generate_sequences_fully_async.remote(actual_batch_num))
            except RolloutShutdownException as e:
                logger.error(f"rollout aborted due to SampleQueue shutdown, stopping executor: {e}", exc_info=True)
                break

            iteration += actual_batch_num
            logger.info(
                f"|perf-stat|rollout| fully_async iter {iteration}/{self.train_iters} "
                f"({actual_batch_num} batches), cost={time.time() - start_time:.2f}s"
            )

        self.controller.finish_rollout()
        logger.info("fully_async rollout process succeed!")
