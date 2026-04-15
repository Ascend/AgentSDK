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
import time
from typing import Any, Dict, List, Tuple

import ray
from codetiming import Timer

from mindspeed_rl import Metric
from mindspeed_rl.trainer.utils import compute_grpo_data_metrics
from mindspeed_rl.utils.utils import metrics_post_processing, compute_tps, metrics_sort

from agentic_rl.controllers.train_controller.train_controller import TrainController
from agentic_rl.trainer.train_adapter.mindspeed_rl.utils.trainer_utils import CommonGRPOTrainer
from agentic_rl.data_manager.data_transform import padding_dict_to_tensor_dict
from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class OneStepOffTrainExecutor(CommonGRPOTrainer):
    """Executor for one-step off-policy GRPO training."""

    def __init__(self, controller: TrainController, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the one-step off-policy training executor.

        Args:
            controller: The training controller managing rollout and weight updates.
            *args: Positional arguments forwarded to CommonGRPOTrainer.
            **kwargs: Keyword arguments including training configuration such as
                validate_freq, test_before_train, test_only, weight_save_dir,
                update_weights_interval, ckpt_delta, and data_optimized.
        """
        self.validate_freq = kwargs["validate_freq"]
        self.test_before_train = kwargs["test_before_train"]
        self.test_only = kwargs["test_only"]
        self.controller = controller
        self.weight_save_dir = kwargs["weight_save_dir"]
        self.update_weights_interval = kwargs["update_weights_interval"]
        self.delta = kwargs["ckpt_delta"]
        self.data_optimized = kwargs["data_optimized"]

        super().__init__(*args, **kwargs)

    def transfer_dock_init(self) -> None:
        """Initialize the transfer dock, adding response_mask key when data optimization is disabled."""
        if self.data_optimized:
            super().transfer_dock_init()
        else:
            dataset_additional_keys = self.dataset_additional_keys
            self.dataset_additional_keys = dataset_additional_keys + ["response_mask"]
            super().transfer_dock_init()
            self.dataset_additional_keys = dataset_additional_keys

    def put_data_to_td(self, output: Dict[str, Any], index: List[int]) -> None:
        """
        Move batch output data to the transfer dock.

        Args:
            output: Dictionary of tensors or lists to transfer.
            index: List of sample indices within the batch.
        """
        output = {key: value.cpu() if not isinstance(value, list) else value for key, value in output.items()}
        output = padding_dict_to_tensor_dict(output)
        self.transfer_dock.put_experience.remote(data_dict=output, indexes=index)

    def update_rollout_metrics(self, rollout_metric: Dict[str, Any]) -> None:
        """
        Push rollout timing and reward metrics to the transfer dock.

        Args:
            rollout_metric: Dictionary containing rollout cost, resharding time,
                and per-trajectory reward statistics.
        """
        rollout_cost = float(rollout_metric['rollout_cost'])
        resharding_to_infer = float(rollout_metric['resharding_to_infer'])
        res_reward_mean = float(rollout_metric['res_reward_mean'])
        res_reward_min = float(rollout_metric['res_reward_min'])
        res_reward_max = float(rollout_metric['res_reward_max'])
        toolcall_reward_mean = float(rollout_metric['toolcall_reward_mean'])
        toolcall_reward_min = float(rollout_metric['toolcall_reward_min'])
        toolcall_reward_max = float(rollout_metric['toolcall_reward_max'])

        ray.get(self.transfer_dock.update_metrics.remote(
            "timing/rollout", value=[round(rollout_cost, 4), round(0.0, 4)], cumulate=True))
        ray.get(self.transfer_dock.update_metrics.remote(
            "timing/resharding_to_infer", value=[round(resharding_to_infer, 4)], cumulate=True))
        ray.get(self.transfer_dock.update_metrics.remote(
            "traj/res_reward_mean", value=[res_reward_mean], cumulate=True))
        ray.get(self.transfer_dock.update_metrics.remote(
            "traj/res_reward_min", value=[res_reward_min], cumulate=True))
        ray.get(self.transfer_dock.update_metrics.remote(
            "traj/res_reward_max", value=[res_reward_max], cumulate=True))
        ray.get(self.transfer_dock.update_metrics.remote(
            "traj/toolcall_reward_mean", value=[toolcall_reward_mean], cumulate=True))
        ray.get(self.transfer_dock.update_metrics.remote(
            "traj/toolcall_reward_min", value=[toolcall_reward_min], cumulate=True))
        ray.get(self.transfer_dock.update_metrics.remote(
            "traj/toolcall_reward_max", value=[toolcall_reward_max], cumulate=True))

    def collect_iteration_timer_metrics(
        self, iteration: int, rollout_metric: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compute GRPO data metrics and collect transfer dock metrics for one iteration.

        Args:
            iteration: Current training iteration index (0-based).
            rollout_metric: Rollout timing and reward metrics from the rollout unit.

        Returns:
            A tuple of (grpo_data_metrics, metrics_result) dictionaries.
        """
        logger.info('compute_grpo_data_metrics start ...')
        self.update_rollout_metrics(rollout_metric)
        grpo_data_metrics = compute_grpo_data_metrics(
            self.transfer_dock, self.global_batch_size * self.n_samples_per_prompt, self.tokenizer,
            self.global_batch_size * self.n_samples_per_prompt, self.guarantee_order)
        metrics_result = ray.get(self.transfer_dock.get_metrics.remote())
        self.controller.finish_training_iteration(iteration=(iteration + 1))
        return grpo_data_metrics, metrics_result

    def collect_iteration_metrics(
        self,
        iteration: int,
        all_timer: Timer,
        metrics: Metric,
        grpo_data_metrics: Dict[str, Any],
        metrics_result: Dict[str, Any],
    ) -> None:
        """
        Post-process, log, and record metrics for a completed iteration.

        Args:
            iteration: Current training iteration index (0-based).
            all_timer: Timer capturing the total iteration wall time.
            metrics: Metric accumulator to update with new values.
            grpo_data_metrics: GRPO-specific data metrics for this iteration.
            metrics_result: Raw metrics from the transfer dock.
        """
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

        log.format_info(metrics.metric, (iteration + 1), self.train_iters)
        ray.get(self.transfer_dock.clear.remote())
        if self.tensorboard is not None:
            for metric_name, metric_value in metrics.metric.items():
                self.tensorboard.add_scalar(f"train/{metric_name}", metric_value, (iteration + 1))

    def update_weights_to_rollout_unit(self, last_iteration: bool, iteration: int) -> None:
        """
        Send updated model weights to the rollout unit at configured intervals.

        Args:
            last_iteration: Whether this is the final training iteration.
            iteration: Current training iteration index (0-based).
        """
        if not last_iteration and (iteration % self.update_weights_interval == 0):
            logger.info("update weights start ...")
            start_time = time.time()
            self.controller.update_rollout_weights(iteration=iteration)
            logger.info(f"update weights done, cost: {time.time() - start_time}")

    def fit(self) -> None:
        """
        Run the main one-step off-policy GRPO training loop.

        Iterates from the current checkpoint iteration to train_iters,
        fetching batches from the controller, computing advantages and
        log-probabilities, updating the actor, and collecting metrics.
        Checkpoints are saved at configured intervals.
        """
        metrics = Metric()
        iteration = self.actor_worker.get_iteration()

        logger.info(
            f'async start one step off grpo training at iteration: {iteration}/{self.train_iters} ...'
        )

        while iteration < self.train_iters:
            last_iteration = (iteration == (self.train_iters - 1))
            batch, rollout_metric = self.controller.get_next_training_batch(last_iteration)
            if batch is None or rollout_metric is None:
                continue

            with Timer(name='iteration', logger=None) as all_timer:
                prompt_length = batch['prompt_length']
                indexes = list(range(len(prompt_length)))
                logger.info(f"get train batch prompt_length: {prompt_length}, indexes: {indexes}")
                self.put_data_to_td(batch, indexes)

                logger.info('compute_advantage start ...')
                self.compute_advantage(blocking=False, guarantee_order=self.guarantee_order)

                logger.info('compute_ref_log_prob start ...')
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
                self.actor_worker.update(self.kl_ctrl, self.skip_actor_log_prob)
                self.actor_worker.wait_all_ref_objs_run_over()

                self.update_weights_to_rollout_unit(last_iteration, iteration)

                grpo_data_metrics, metrics_result = self.collect_iteration_timer_metrics(iteration, rollout_metric)

            self.collect_iteration_metrics(iteration, all_timer, metrics, grpo_data_metrics, metrics_result)
            iteration += 1
            if iteration % self.save_interval == 0 or iteration == self.train_iters:
                self.save_checkpoint(iteration)

        self.controller.finish_training()
        logger.info("one step off train process succeed!")