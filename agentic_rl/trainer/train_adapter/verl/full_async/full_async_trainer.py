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
import json
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from recipe.fully_async_policy.detach_utils import MetricsAggregator
from recipe.fully_async_policy.ray_trainer import FullyAsyncRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl import DataProto

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class FullyAsyncTrainer(FullyAsyncRayPPOTrainer):
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
            reward_fn=None,
            delta=None,
            val_reward_fn=None,
            weight_save_dir: str | None = None,
            update_weights_interval: int = 1,
            device_name: str | None = None,
    ):
        self.delta = delta
        self.weight_save_dir = weight_save_dir
        self.update_weights_interval = update_weights_interval
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        logger.info(f"[FullyAsyncTrainer] self.reward_fn={self.reward_fn}")

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        if self.hybrid_engine:
            raise ValueError("hybrid_engine must be False")

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # KL loss control currently not supported
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.param_synchronizer = None

        # we start from step 1
        self.global_steps = 1
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_samples_processed = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0

        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.compute_prox_log_prob = self.config.async_training.compute_prox_log_prob
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)
        self.controller = None
        self.data_manager = None

    def set_controller(self, controller) -> None:
        """Set the training controller."""
        self.controller = controller

    def set_data_manager(self, data_manager) -> None:
        """Set data manager."""
        self.data_manager = data_manager

    def set_parameter_synchronizer(self, param_synchronizer) -> None:
        """Set parameter synchronizer."""
        self.param_synchronizer = param_synchronizer

    def set_total_train_steps(self, total_train_steps: int) -> None:
        """Set total training steps and initialize the progress bar."""
        self.total_train_steps = total_train_steps
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
            experience_consumer_stage="train",
            experience_columns=None,
            experience_count=self.required_samples
        )
        return 0, processed_batch

    def _prepare_single_generation_data(self, batch_dict: dict, config) -> DataProto:
        """Prepare a single sample for generation.

        Args:
            batch_dict: Single sample dictionary containing tensors.
            config: Training configuration.

        Returns:
            Prepared DataProto for training.

        Raises:
            RuntimeError: If '_prompt_id' is missing from batch_dict.
        """
        if '_prompt_id' in batch_dict:
            uid = batch_dict.pop('_prompt_id')
        else:
            raise RuntimeError("No prompt_id.")

        full_batch = DataProto.from_single_dict(batch_dict)
        full_batch.non_tensor_batch["uid"] = uid

        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = []
        full_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        return full_batch

    def _create_actor_rollout_classes(self) -> None:
        """Register the actor class in the resource pool."""
        for role in [Role.Actor]:
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

        self.actor_wg = self.all_wg[str(Role.Actor)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg

    def _init_async_rollout_manager(self) -> None:
        """Override: no async rollout manager needed in fully-async mode."""
        pass

    def fit(self) -> None:
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        logger.info("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
        if self.param_synchronizer is None:
            raise ValueError("param_synchronizer client not set. Call set_parameter_synchronizer() first.")

        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.max_steps_duration = 0

        self._log_validation_data()

        while True:
            metrics = {}
            timing_raw = {}

            with marked_timer("step", timing_raw):
                with marked_timer("gen", timing_raw, color="red"):
                    epoch, rollout_return_batch = self._get_samples_from_queue()
                    if rollout_return_batch is None:
                        break
                    tensors = {
                        "prompts": rollout_return_batch["prompts"],
                        "responses": rollout_return_batch["responses"],
                        "input_ids": rollout_return_batch["input_ids"],
                        "rm_scores": rollout_return_batch["rm_scores"],
                        "token_level_rewards": rollout_return_batch["token_level_rewards"],
                        "position_ids": rollout_return_batch["position_ids"],
                        "attention_mask": rollout_return_batch["attention_mask"],
                        "response_mask": rollout_return_batch["response_mask"],
                        "rollout_log_probs": rollout_return_batch["rollout_log_probs"],
                    }
                    batch = DataProto.from_dict(tensors=tensors)
                    batch.non_tensor_batch["uid"] = np.array(rollout_return_batch["prompt_ids"])
                    batch.meta_info["global_token_num"] = torch.sum(
                        rollout_return_batch["attention_mask"], dim=-1
                    ).tolist()

                score_batch = batch.batch['token_level_rewards']
                mean_reward = (score_batch.sum() / score_batch.size(0)).item()
                metrics['training/reward_mean'] = mean_reward
                logger.info(f"[FullyAsyncTrainer] Mean Reward of this batch: {mean_reward:.4f}, {score_batch.sum()=}")

                score_batch = batch.batch['rm_scores']
                mean_reward = (score_batch.sum() / score_batch.size(0)).item()
                metrics['training/rm_scores_mean'] = mean_reward
                logger.info(f"[FullyAsyncTrainer] Mean rm_scores of this batch: {mean_reward:.4f}, {score_batch.sum()=}")

                batch, reward_extra_infos_dict = self._process_batch_common(
                    batch, metrics, timing_raw, self.local_trigger_step if self.compute_prox_log_prob else None
                )
                self._log_rollout(batch, reward_extra_infos_dict, timing_raw)

                start_time = time.time()
                self.controller.update_rollout_weights(self.global_steps)
                logger.info(f"update weights done, cost: {time.time() - start_time}")

            self._collect_metrics(batch, 0, metrics, timing_raw)
            self.metrics_aggregator.add_step_metrics(
                metrics=metrics, sample_count=self.required_samples, timestamp=time.time()
            )

            time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            logger.info(
                f"[FullyAsyncTrainer] global_steps: {self.global_steps} "
                f"local_trigger_step: {self.local_trigger_step} "
                f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
                f"{time_str}"
            )
            self._trigger_parameter_sync_after_step(global_steps=self.global_steps)
            self._log_validation_data()
            self._check_save_checkpoint(timing_raw)
            self.global_steps += 1

        # Final parameter sync and validate
        ray.get(self.param_synchronizer.wait_last_valid.remote())
        self._log_validation_data()

        if self.current_param_version % self.config.trainer.save_freq != 0 or self.local_trigger_step > 1:
            self._trigger_parameter_sync_after_step(validate=True, global_steps=self.global_steps)
            ray.get(self.param_synchronizer.wait_last_valid.remote())
            self._log_validation_data()
        self.progress_bar.close()

        self._check_save_checkpoint(timing_raw)

    def _check_save_checkpoint(self, timing_raw: dict) -> None:
        if self.current_param_version == self.last_ckpt_version:
            return

        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )

        if self.config.trainer.save_freq > 0 and (
                self.current_param_version % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                logger.warning("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                self._save_checkpoint()
                self.last_ckpt_version = self.current_param_version

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
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.current_param_version}",
                    str(Role.Critic),
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        ray.get(self.param_synchronizer.rollouter_save_checkpoint.remote(local_global_step_folder))

        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        ckpt_path = os.path.realpath(local_latest_checkpointed_iteration)
        file_descriptor = os.open(ckpt_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(file_descriptor, "w") as file_handle:
            file_handle.write(str(self.current_param_version))

    def load_checkpoint(self) -> int:
        """Load model checkpoint from disk or HDFS.

        Returns:
            The parameter version restored from the checkpoint (0 if training from scratch).
        """
        if self.config.trainer.resume_mode == "disable":
            self.actor_rollout_wg.load_checkpoint(None)
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
                logger.info("[FullyAsyncTrainer] Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
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
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
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
        return self.current_param_version

    def _collect_metrics_from_samples(self, batch: DataProto, metrics: dict) -> None:
        """Collect staleness and async metrics from the sample batch."""
        if hasattr(batch, "meta_info") and batch.meta_info:
            samples_param_versions = batch.meta_info["rollout_param_versions"]
            stale_count = sum(1 for v in samples_param_versions if self.current_param_version - v >= 1)
            self.stale_samples_processed += stale_count
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            metrics.update(
                {
                    "fully_async/count/stale_samples_processed": self.stale_samples_processed,
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                }
            )
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s"):
                    metrics[key] = value

    def _trigger_parameter_sync_after_step(
        self, validate: bool = False, global_steps: int | None = None
    ) -> None:
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

    def _log_rollout(self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict) -> None:
        def safe(tensor_or_array: Any) -> Any:
            if torch.is_tensor(tensor_or_array):
                return tensor_or_array.detach().cpu().tolist()
            if isinstance(tensor_or_array, np.ndarray):
                return tensor_or_array.tolist()
            return tensor_or_array

        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                log_attention_mask = safe(batch.batch["attention_mask"])
                log_response_mask = safe(batch.batch["response_mask"])
                log_position_ids = safe(batch.batch["position_ids"])
                log_rm_scores = safe(batch.batch["rm_scores"])
                log_token_level_rewards = safe(batch.batch["token_level_rewards"])
                log_inputs = safe(batch.batch["prompts"])
                log_outputs = safe(batch.batch["responses"])
                sample_gts = [
                    item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                    for item in batch
                ]

                if "request_id" in batch.non_tensor_batch:
                    reward_extra_infos_dict.setdefault(
                        "request_id",
                        batch.non_tensor_batch["request_id"].tolist(),
                    )

                self._dump_generations(
                    raw_input=log_inputs,
                    raw_output=log_outputs,
                    attention_mask=log_attention_mask,
                    response_mask=log_response_mask,
                    position_ids=log_position_ids,
                    rm_scores=log_rm_scores,
                    token_level_rewards=log_token_level_rewards,
                    inputs=inputs,
                    outputs=outputs,
                    gts=sample_gts,
                    scores=scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=rollout_data_dir,
                )

    def _dump_generations(
        self,
        raw_input,
        raw_output,
        attention_mask,
        response_mask,
        position_ids,
        rm_scores,
        token_level_rewards,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict: dict,
        dump_path: str,
    ) -> None:
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.realpath(os.path.join(dump_path, f"{self.global_steps}.jsonl"))

        num_samples = len(inputs)
        base_data = {
            "raw_input": raw_input,
            "raw_output": raw_output,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
            "rm_scores": rm_scores,
            "token_level_rewards": token_level_rewards,
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * num_samples,
        }

        for key, value in reward_extra_infos_dict.items():
            if len(value) == num_samples:
                base_data[key] = value

        lines = []
        for idx in range(num_samples):
            entry = {key: value[idx] for key, value in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        file_descriptor = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(file_descriptor, "w") as file_handle:
            file_handle.write("\n".join(lines) + "\n")

        logger.info(f"Dumped generations to {filename}")
