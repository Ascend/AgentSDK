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


import traceback

from codetiming import Timer
import ray
import torch

from mindspeed_rl import RayGRPOTrainer, Metric
from mindspeed_rl.trainer.utils import compute_grpo_data_metrics
from mindspeed_rl.utils.utils import compute_tps, metrics_post_processing, metrics_sort

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.trainer.train_adapter.mindspeed_rl import patch
from agentic_rl.trainer.train_adapter.mindspeed_rl.utils.trainer_utils import CommonGRPOTrainer

log = Loggers(__name__)
logger = log.get_logger()


class AgentGRPOTrainer(CommonGRPOTrainer):

    def __init__(self, rollout_worker, *args, **kwargs):
        self.rollout_worker = rollout_worker
        self.validate_freq = kwargs["validate_freq"]
        self.test_before_train = kwargs["test_before_train"]
        self.test_only = kwargs["test_only"]
        super().__init__(*args, **kwargs)

    def transfer_dock_init(self):
        dataset_additional_keys = self.dataset_additional_keys
        self.dataset_additional_keys = dataset_additional_keys + ["response_mask"]
        super().transfer_dock_init()
        self.dataset_additional_keys = dataset_additional_keys
        ray.get(self.rollout_worker.init_data_manager.remote(self.transfer_dock))

    def fit(self, data_iters, val_dataloader, test_dataloader):
        """
        The utils loop of GRPO
        """
        from mindspeed_rl.trainer.utils.transfer_dock import put_prompts_experience

        metrics = Metric()

        iteration = self.actor_worker.get_iteration()

        if self.blocking:
            logger.info('sync start grpo training at iteration: {}/{} ...'.format(iteration, self.train_iters))
        else:
            logger.info('async start grpo training at iteration: {}/{} ...'.format(iteration, self.train_iters))

        if self.test_before_train and not test_dataloader is None:
            val_metrics = self._validate_agent(test_dataloader, True)
            logger.warning("test before training")
            for key, value in val_metrics.items():
                logger.warning(f"{key}: {value}")
            if self.test_only:
                return

        while iteration < self.train_iters:
            with Timer(name='iteration', logger=None) as all_timer:
                batch = next(data_iters)
                batch_dict, indexes = put_prompts_experience(batch, self.n_samples_per_prompt, self.dataset_additional_keys)
                ray.get(self.transfer_dock.put_experience.remote(data_dict=batch_dict, indexes=indexes, is_prompt=True))

                # generate sequences
                # self.actor_worker.generate_sequences(blocking=self.blocking)

                logger.info('rollout_worker start ...')
                try:
                    ray.get(self.rollout_worker.generate_sequences.remote())
                except Exception as e:
                    traceback.print_exc()
                    print(f"error: {e}")

                # logger.info('compute rm scores start ...')
                # # compute rm scores.
                # rule_reward = []
                # for reward_worker in self.reward_list:
                #     if isinstance(reward_worker, RayActorGroup):
                #         reward_worker.compute_rm_score(blocking=self.blocking)
                #     else:
                #         logger.info('reward_worker start ...')
                #         rule_reward.append(ray.get(reward_worker.compute_rm_score.remote()))

                logger.info('compute_advantage start ...')
                # compute advantages, executed on the driver process
                self.compute_advantage(blocking=False, guarantee_order=self.guarantee_order)

                logger.info('compute_ref_log_prob start ...')
                # compute reference log_prob
                self.ref_worker.compute_ref_log_prob(blocking=self.blocking)

                logger.info(f'compute_log_prob start self.skip_actor_log_prob={self.skip_actor_log_prob}...')
                # compute old log_prob
                if not self.skip_actor_log_prob:
                    self.actor_worker.compute_log_prob(blocking=self.blocking)
                self.actor_worker.wait_all_ref_objs_run_over()

                logger.info('wait_all_ref_objs_run_over start ...')
                self.ref_worker.wait_all_ref_objs_run_over()
                for idx, reward in enumerate(self.reward_list):
                    if hasattr(reward, 'wait_all_ref_objs_run_over'):
                        reward.wait_all_ref_objs_run_over()

                logger.info('update start ...')
                # update actor
                self.actor_worker.update(self.kl_ctrl, self.skip_actor_log_prob)
                self.actor_worker.wait_all_ref_objs_run_over()

                # validate
                if not val_dataloader is None and self.validate_freq > 0 and iteration % self.validate_freq == 0:
                    val_metrics: dict = self._validate_agent(val_dataloader, True)
                    logger.warning("validate result ...")
                    for key, value in val_metrics.items():
                        logger.warning(f"{key}: {value}")
                        metrics.update(key, value)

                logger.info('compute_grpo_data_metrics start ...')
                # collect metrics
                grpo_data_metrics = compute_grpo_data_metrics(self.transfer_dock,
                                                              self.global_batch_size * self.n_samples_per_prompt,
                                                              self.tokenizer,
                                                              self.global_batch_size * self.n_samples_per_prompt,
                                                              self.guarantee_order)
                metrics_result = ray.get(self.transfer_dock.get_metrics.remote())

            metrics_result = metrics_post_processing(metrics_result)
            metrics_result = metrics_sort(metrics_result, all_timer.last)
            log_max_throughput = False
            tps = compute_tps(self.kwargs, grpo_data_metrics, self.global_batch_size, self.n_samples_per_prompt,
                              all_timer.last, log_max_throughput)
            update_tps = compute_tps(self.kwargs, grpo_data_metrics, self.global_batch_size, self.n_samples_per_prompt,
                                     metrics_result["timing/update"], log_max_throughput)
            vllm_tps = compute_tps(self.kwargs, grpo_data_metrics, self.global_batch_size, self.n_samples_per_prompt,
                                   metrics_result["timing/rollout"], log_max_throughput)
            metrics.update(value=metrics_result)
            metrics.update(value=grpo_data_metrics)
            metrics.update("tokens/p/s", tps)
            metrics.update("update_tps", update_tps)
            metrics.update("vllm_throughput", vllm_tps)

            iteration += 1
            log.format_info(metrics.metric, iteration, self.train_iters)
            ray.get(self.transfer_dock.clear.remote())
            if self.tensorboard is not None:
                for k, v in metrics.metric.items():
                    self.tensorboard.add_scalar(f"train/{k}", v, iteration)
            if iteration % self.save_interval == 0 or iteration == self.train_iters:
                self.save_checkpoint(iteration)

        if not test_dataloader is None:
            val_metrics: dict = self._validate_agent(test_dataloader, True)
            logger.warning("test result")
            for key, value in val_metrics.items():
                logger.warning(f"{key}: {value}")

        logger.info('after grpo training is done')
        ray.shutdown()