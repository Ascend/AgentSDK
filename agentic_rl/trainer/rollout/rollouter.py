#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
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
# -------------------------------------------------------------------------


import time
import ray

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.controllers.rollout_controller.rollout_queue import get_rollout_queue_actor
from agentic_rl.controllers.utils.utils import MIN_SLEEP_TIME

logger = Loggers(__name__).get_logger()


class OneStepOffRollouter:
    def __init__(
        self,
        controller,
        rollout_worker,
        train_iters,
        padding_dict_to_tensor_dict,
        put_prompts_experience,
        **kwargs
    ):
        self.controller = controller
        self.rollout_worker = rollout_worker
        self.queue_actor = get_rollout_queue_actor()
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
        if self.data_optimized:
            from agentic_rl.trainer.rollout.rollout_dataset import \
                optimized_preprocess_input
            from agentic_rl.trainer.rollout.rollout_dataset import \
                optimized_put_prompt_experience
            mini_batches, prompt_ids = optimized_preprocess_input(batch)
            batch_dict, indexes = optimized_put_prompt_experience(
                mini_batches, prompt_ids, self.padding_dict_to_tensor_dict)
        else:
            batch_dict, indexes = self.put_prompts_experience(
                batch, self.n_samples_per_prompt, self.dataset_additional_keys)
        return batch_dict, indexes

    @staticmethod
    def merge_batch_list(batches):
        merged_batch = {}
        if not batches:
            return merged_batch
        keys = batches[0].keys()
        # For each key, merge lists from all batches
        for key in keys:
            merged_batch[key] = []
            for batch in batches:
                merged_batch[key].extend(batch[key])
        return merged_batch

    def fit(self):
        iteration = 0
        logger.info(f"start grpo rollout loop, iteration: {iteration}/{self.train_iters} ...")

        while iteration < self.train_iters and not ray.get(self.queue_actor.is_shutdown.remote()):
            queue_size = ray.get(self.queue_actor.queue_size.remote())
            is_running = ray.get(self.queue_actor.is_running.remote())
            if queue_size <= 0 or not is_running:
                time.sleep(self.sleep_time)
                continue

            # Get data from queue
            start_time = time.time()
            actual_batch_num = min(self.hybrid_batch_num, queue_size)
            # Total number of hybrid batches cannot exceed the number of iterations
            if iteration + actual_batch_num > self.train_iters:
                actual_batch_num = ((iteration + actual_batch_num) - self.train_iters)

            logger.info(f"available rollout queue size: {queue_size}, "
                        f"hybrid_batch_num: {self.hybrid_batch_num}, "
                        f"actual_batch_num: {actual_batch_num}")

            batch_list = [ray.get(self.queue_actor.pop_queue.remote()) for _ in range(actual_batch_num)]
            batch = self.merge_batch_list(batch_list)

            logger.info(f"|perf-stat|rollout| rollout got {actual_batch_num} batch ... ")
            # Put the data into data_manager
            batch_dict, indexes = self.get_batch_dict(batch)
            logger.info("finish putting prompts experience...")
            ray.get(self.rollout_worker.data_manager_put_experience.remote(batch_dict=batch_dict, index=indexes))

            # Extract data from data_manager for inference
            logger.info(f'rollout_worker start, actual_batch_num={actual_batch_num}, tasks={len(indexes)} ...')
            ray.get(self.rollout_worker.generate_sequences.remote(actual_batch_num))

            iteration += actual_batch_num
            logger.info(f"|perf-stat|rollout| ===rollout iteration for {actual_batch_num} batches: "
                        f"{iteration}/{self.train_iters}, timing/rollout : {time.time() - start_time:.4f}===")

        self.controller.finish_rollout()
        logger.info("one step off rollout process succeed!")
