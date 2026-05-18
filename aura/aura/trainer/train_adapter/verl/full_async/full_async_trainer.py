# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0 OR MulanPSL-2.0
# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl.experimental.fully_async_policy.detach_utils import MetricsAggregator
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking
from verl import DataProto

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class TrainingStopException(Exception):
    """Exception raised to signal training should stop"""

    pass


@ray.remote(num_cpus=10)
class FullyAsyncTrainer(SeparateRayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
        delta=None,
        weight_save_dir: str = None,
        update_weights_interval=1,
    ):
        self.delta = delta
        self.weight_save_dir = weight_save_dir
        self.update_weights_interval = update_weights_interval
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        if self.hybrid_engine:
            raise ValueError("hybrid_engine must be False")

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_rm = need_reward_model(self.config)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # KL loss control currently not supported
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # ==================== SeparateRayPPOTrainer config ====================
        self.global_steps = 0
        self.epoch = 0
        self.max_steps_duration = 0
        self.progress_bar = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        # ==================== fully async config ====================

        # self.message_queue_client = None

        # Statistics
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0
        self.train_role = Role.ActorRollout if config.async_training.use_trainer_do_validate else Role.Actor

        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)
        self.controller = None
        self.data_manager = None

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        self._init_reward_loop()
        self._init_async_rollout_manager()

    def set_controller(self, controller) -> None:
        """Set the training controller."""
        self.controller = controller

    def set_data_manager(self, data_manager) -> None:
        """Set data manager."""
        self.data_manager = data_manager

    def set_total_train_steps(self, total_training_steps):
        self.total_train_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            logger.error(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

        self.progress_bar = tqdm(total=self.total_train_steps, initial=0, desc="Training Progress")

    def get_actor_wg(self):
        """Get actor worker group."""
        return self.actor_wg

    def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """Get samples from the data manager.

        Returns:
            Tuple of (epoch, processed_batch), or (None, None) when data is exhausted.
        """
        logger.info(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue",
        )

        processed_batch, _ = self.data_manager.get_data(
            experience_consumer_stage="train", experience_columns=None, experience_count=self.required_samples
        )
        return 0, processed_batch

    def _create_actor_rollout_classes(self) -> None:
        """Register the actor class in the resource pool."""
        for role in [self.train_role]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self) -> None:
        """Initialize all model worker groups (critic, ref policy, RM, actor)."""
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_wg = self.all_wg[str(self.train_role)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg

    def _init_async_rollout_manager(self) -> None:
        """Override: no async rollout manager needed in fully-async mode."""
        pass

    async def fit(self) -> None:
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        logger.info("start fully async training...")

        self.max_steps_duration = 0

        self.global_steps += 1

        # 适配verl续训
        self._load_checkpoint()

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while True:
            try:
                await self.fit_step()
            except TrainingStopException:
                logger.info("training stopped by termination signal")
                break

        self.progress_bar.close()
        if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
            await self._fit_update_weights()
            await self._fit_validate()
        self._fit_save_checkpoint(force=True)
        self.controller.finish_training()
        logger.info("fully async training finished!!!")

    async def fit_step(self) -> None:
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing
        """
        logger.info(f"=== start train iteration {self.global_steps} ===")
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            batch = await self._fit_generate()
            logger.info("got a batch, start compute reward")
            batch = self._fit_compute_reward(batch)
            logger.info("start compute log prob")
            batch = self._fit_compute_log_prob(batch)
            logger.info("start compute ref log prob")
            batch = self._fit_compute_ref_log_prob(batch)
            logger.info("start compute critic")
            batch = self._fit_compute_critic(batch)
            logger.info("start compute advantage")
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)

            logger.info("start update...")
            batch = self._fit_update_actor(batch)
            self._fit_update_local_step()

            logger.info(f"start update weights, self.local_trigger_step: {self.local_trigger_step}")
            await self._fit_update_weights()
            self._fit_dump_data(batch)

        # await self._fit_validate()
        self._fit_save_checkpoint(force=True)
        self._fit_stop_profile()

        logger.info("start collect metrics")
        self._fit_collect_metrics(batch)

        self._fit_torch_memory()
        self._fit_postprocess_step()
        logger.info(f"finished train iteration {self.global_steps}/{self.total_train_steps}")

        if self.global_steps == self.total_train_steps:
            raise TrainingStopException("end of training loop")

    async def _fit_generate(self) -> DataProto | None:
        # metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            # 数据获取会循环等待, 直到有数据获取到, 所以不需要对rollout_return_batch判空处理
            epoch, rollout_return_batch = self._get_samples_from_queue()
            if epoch is None or rollout_return_batch is None:
                logger.warning("[FullyAsyncTrainer] No samples available from queue, skipping generation step")
                return None
            tensors = {
                "prompts": rollout_return_batch["prompts"],
                "responses": rollout_return_batch["responses"],
                "input_ids": rollout_return_batch["input_ids"],
                "rm_scores": rollout_return_batch["rm_scores"],
                "token_level_rewards": rollout_return_batch["token_level_rewards"],
                "position_ids": rollout_return_batch["position_ids"],
                "attention_mask": rollout_return_batch["attention_mask"],
                "response_mask": rollout_return_batch["response_mask"],
            }
            if "rollout_log_probs" in rollout_return_batch:
                tensors["rollout_log_probs"] = rollout_return_batch["rollout_log_probs"]
            batch = DataProto.from_dict(tensors=tensors)
            batch.non_tensor_batch["uid"] = np.array(rollout_return_batch["prompt_ids"])
            batch.meta_info["global_token_num"] = torch.sum(rollout_return_batch["attention_mask"], dim=-1).tolist()
            # self._collect_metrics_from_samples(batch, metrics)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        rewards = batch.batch["token_level_rewards"]
        rm_scores = batch.batch["rm_scores"]
        responses = batch.batch["responses"]
        logger.info(f"batch token_level_rewards: {rewards}")
        logger.info(f"batch rm_scores: {rm_scores}")
        logger.info(f"batch response len: {len(responses)}, shape: {responses.shape}")
        return batch

    def _fit_update_local_step(self) -> None:
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logger.info(
            f"[FullyAsyncTrainer] global_steps: {self.global_steps} "
            f"local_trigger_step: {self.local_trigger_step} "
            f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
            f"{time_str}"
        )
        if self.local_trigger_step < self.trigger_parameter_sync_step:
            self.local_trigger_step += 1
        else:
            self.current_param_version += 1
            self.local_trigger_step = 1

    async def _fit_update_weights(self) -> None:
        if self.local_trigger_step != 1:
            return

        with marked_timer("timing_s/param_sync", self.timing_raw):
            self.controller.update_rollout_weights(self.current_param_version)
        logger.info(
            f"[FullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f} seconds "
            f"self.current_param_version: {self.current_param_version}"
        )

        # Log aggregated training metrics
        # self.logger.log(
        #     data=self.metrics_aggregator.get_aggregated_metrics(),
        #     step=self.current_param_version,
        # )
        # self.metrics_aggregator.reset()

    def _fit_save_checkpoint(self, force=False) -> None:
        if self.current_param_version == self.last_ckpt_version:
            return

        timing_raw = self.timing_raw
        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        # Check if the conditions for saving a checkpoint are met.
        # The conditions include a mandatory condition (1) and
        # one of the following optional conditions (2/3/4):
        # 1. The save frequency is set to a positive value.
        # 2. It's the last training step.
        # 3. The current step number is a multiple of the save frequency.
        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
        if self.config.trainer.save_freq > 0 and (
            force and self.current_param_version % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                logger.info("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                # sleep replicas to avoid OOM during checkpoint saving
                self._save_checkpoint()
                self.last_ckpt_version = self.current_param_version

    def _fit_postprocess_step(self) -> None:
        self.global_steps += 1

        self.metrics_aggregator.add_step_metrics(
            metrics=self.metrics, sample_count=self.required_samples, timestamp=time.time()
        )

        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=self.current_param_version,
        )
        self.metrics_aggregator.reset()

        if self.local_trigger_step == 1:
            self.progress_bar.update(1)

    def _save_checkpoint(self) -> None:
        """Persist actor (and optionally critic) model checkpoints to disk."""
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )

        logger.info(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            logger.warning(
                "[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = None
            if self.config.trainer.default_hdfs_dir is not None:
                hdfs_folder = self.config.trainer.default_hdfs_dir + f"global_step_{self.current_param_version}"
                critic_remote_path = os.path.join(hdfs_folder, str(Role.Critic))
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )

        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    def load_checkpoint(self) -> int:
        """Load model checkpoint from disk or HDFS.

        Returns:
            The parameter version restored from the checkpoint (0 if training from scratch).
        """
        if self.config.trainer.resume_mode == "disable":
            return 0

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                if not isinstance(self.config.trainer.resume_from_path, str):
                    raise ValueError("resume ckpt must be str type")
                if "global_step_" not in self.config.trainer.resume_from_path:
                    raise ValueError("resume ckpt must specify the global_steps")
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)

        logger.info(f"[FullyAsyncTrainer] Load from checkpoint folder: {global_step_folder}")
        self.current_param_version = int(global_step_folder.split("global_step_")[-1]).strip(" /")
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        logger.info(
            f"[FullyAsyncTrainer] Setting global step to {self.global_steps}, "
            f"current_param_version to {self.current_param_version}"
        )
        logger.info(f"[FullyAsyncTrainer] Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))

        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )

        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
        logger.info("[FullyAsyncTrainer] End loading checkpoint for resume ...")
        return self.current_param_version

    def _collect_metrics_from_samples(self, batch: DataProto, metrics: dict) -> None:
        """Collect staleness and async metrics from the sample batch."""
        if hasattr(batch, "meta_info") and batch.meta_info:
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            metrics.update(
                {
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                }
            )
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s"):
                    metrics[key] = value

    def _trigger_parameter_sync_after_step(self, validate: bool = False, global_steps: int | None = None) -> None:
        """Trigger parameter synchronization after training step.

        This ensures rollouter always uses the latest trained parameters.
        """
        self.current_param_version += 1
        self.local_trigger_step = 1
        data = self.metrics_aggregator.get_aggregated_metrics()
        logger.info(f"[FullyAsyncTrainer] Metrics: {data}")
        self.logger.log(
            data=data,
            step=self.current_param_version,
        )
        self.progress_bar.update(1)
        self.metrics_aggregator.reset()

    def _log_validation_data(self) -> None:
        """Log validation data (currently a no-op pending MessageQueue integration)."""
        return
