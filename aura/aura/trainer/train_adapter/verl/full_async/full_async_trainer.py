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
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking
from verl import DataProto

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def _left_pad(t, target, val):
    """Right-align ``t`` to ``target`` length by left-padding with ``val``."""
    t = torch.as_tensor(t)
    if t.shape[-1] >= target:
        return t[..., -target:]
    return torch.nn.functional.pad(t, (target - t.shape[-1], 0), value=val)


def _right_pad(t, target, val):
    """Left-align ``t`` to ``target`` length by right-padding with ``val``."""
    t = torch.as_tensor(t)
    if t.shape[-1] >= target:
        return t[..., :target]
    return torch.nn.functional.pad(t, (0, target - t.shape[-1]), value=val)


def _extract_scalar(v):
    """Unwrap a single-element list/tensor into its scalar value."""
    if not isinstance(v, list):
        return v
    if len(v) != 1:
        return v
    inner = v[0]
    if isinstance(inner, torch.Tensor):
        return inner.item() if inner.numel() == 1 else inner
    return inner


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
        self.device_name = device_name or self.config.trainer.device

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = self._compute_ref_in_actor(config)

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
        self.train_role = self._compute_train_role(config)

        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.sample_queue = None
        self.last_weight_versions = []
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

    @staticmethod
    def _compute_ref_in_actor(config):
        """Determine whether the reference policy shares the actor weights (LoRA mode)."""
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        return lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

    @staticmethod
    def _compute_train_role(config):
        """Pick the actor role, including validation duty when configured."""
        if config.async_training.use_trainer_do_validate:
            return Role.ActorRollout
        return Role.Actor

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

    def set_sample_queue(self, sample_queue):
        """Inject the SampleQueue actor used for streaming sample collection."""
        self.sample_queue = sample_queue
        logger.info("[async-trainer] sample_queue set for fully_async streaming")

    _FULL_SEQ_KEYS = frozenset({"input_ids", "attention_mask", "position_ids"})
    _RESP_ONLY_KEYS = frozenset({
        "responses", "response_mask", "traj_mask",
        "rm_scores", "token_level_rewards", "rollout_log_probs",
    })
    _PROMPT_ONLY_KEYS = frozenset({"prompts"})

    def _get_samples_from_streaming_queue(self):
        """Collect samples from the streaming SampleQueue and merge into a batch.

        Returns:
            Tuple ``(0, merged_batch)`` on success, ``(None, None)`` when no
            sample arrives before the queue drains.
        """
        logger.info(f"[async-trainer] start collecting {self.required_samples} samples from queue")
        samples, weight_versions = self._collect_samples_from_queue()
        if not samples:
            return None, None
        flat_samples = self._flatten_samples(samples)
        merged = self._merge_and_pad_samples(flat_samples)
        self.last_weight_versions = weight_versions
        self._log_sample_batch(weight_versions)
        return 0, merged

    def _collect_samples_from_queue(self):
        """Pull raw samples from the SampleQueue until ``required_samples`` are gathered."""
        samples, weight_versions = [], []
        collected = 0
        while collected < self.required_samples:
            sample = ray.get(self.sample_queue.get_sample.remote())
            if sample is None:
                if not samples:
                    return [], []
                break
            outputs, weight_version, _ = sample
            samples.append(outputs)
            weight_versions.append(weight_version)
            collected += 1
        logger.info(
            f"[async-trainer] collected {collected} prompt-samples "
            f"(required_samples={self.required_samples})"
        )
        return samples, weight_versions

    def _flatten_samples(self, samples):
        """Expand per-prompt samples into per-trajectory rows for the merge step."""
        flat_samples = []
        for s in samples:
            traj_count = self._infer_traj_count(s.get("input_ids"))
            for i in range(traj_count):
                flat_samples.append(self._extract_trajectory(s, i))
        logger.info(f"[async-trainer] flattened to {len(flat_samples)} trajectories")
        return flat_samples

    @staticmethod
    def _infer_traj_count(input_ids):
        """Determine the number of trajectories encoded in ``input_ids``."""
        if isinstance(input_ids, torch.Tensor):
            return input_ids.shape[0]
        if isinstance(input_ids, list):
            return len(input_ids)
        return 1

    @staticmethod
    def _extract_trajectory(sample, idx):
        """Pick a single trajectory out of a per-prompt sample dict."""
        traj = {}
        for key, val in sample.items():
            if isinstance(val, (torch.Tensor, list, np.ndarray)):
                traj[key] = val[idx]
            else:
                traj[key] = val
        return traj

    def _merge_and_pad_samples(self, samples):
        """Pad and stack per-trajectory fields into batched tensors.

        Group-based streaming samples carry different ``p_g``/``r_g`` lengths, so
        fields are realigned to ``[pad_prompt | prompt | response | pad_resp]``
        to keep responses right-aligned for verl's right-side slicing.
        """
        merged = {}
        real_prompt_lens, real_resp_lens = self._compute_real_lens(samples)
        p_max = max(real_prompt_lens) if real_prompt_lens else 0
        r_max = max(real_resp_lens) if real_resp_lens else 0
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        for key in samples[0]:
            merged[key] = self._merge_field(
                key, samples, real_prompt_lens, p_max, r_max, pad_id
            )
        return merged

    @staticmethod
    def _compute_real_lens(samples):
        """Return per-sample real prompt lengths and response lengths."""
        real_prompt_lens, real_resp_lens = [], []
        for s in samples:
            pl = FullyAsyncTrainer._normalize_prompt_length(s["prompt_length"])
            am = torch.as_tensor(s["attention_mask"])
            real_prompt_lens.append(pl)
            real_resp_lens.append(max(int(am.sum().item()) - pl, 0))
        return real_prompt_lens, real_resp_lens

    @staticmethod
    def _normalize_prompt_length(pl):
        """Normalize a ``prompt_length`` field (list/tensor/scalar) to ``int``."""
        if isinstance(pl, list):
            pl = pl[0] if len(pl) == 1 else pl
        if isinstance(pl, torch.Tensor):
            pl = pl.item() if pl.numel() == 1 else pl
        return int(pl)

    def _merge_field(self, key, samples, real_prompt_lens, p_max, r_max, pad_id):
        """Merge a single field across all samples according to its category."""
        vals = [s[key] for s in samples]
        first = vals[0]
        if isinstance(first, torch.Tensor):
            return self._merge_tensor_field(key, vals, samples, real_prompt_lens, p_max, r_max, pad_id)
        if isinstance(first, np.ndarray):
            return np.stack(vals, axis=0)
        if isinstance(first, list):
            return self._merge_list_field(vals)
        return vals

    def _merge_tensor_field(self, key, vals, samples, real_prompt_lens, p_max, r_max, pad_id):
        """Stack tensor fields with appropriate padding based on their category."""
        first = vals[0]
        if first.dim() == 0:
            return torch.stack(vals, dim=0)
        if key in self._FULL_SEQ_KEYS:
            return self._stack_full_seq(key, vals, samples, real_prompt_lens, p_max, r_max, pad_id)
        if key in self._RESP_ONLY_KEYS:
            return self._stack_resp_only(key, vals, r_max, pad_id)
        if key in self._PROMPT_ONLY_KEYS:
            return self._stack_prompt_only(vals, p_max, pad_id)
        return self._stack_generic_tensor(vals)

    @staticmethod
    def _stack_prompt_only(vals, p_max, pad_id):
        """Left-pad prompt-only fields to ``p_max`` keeping real tokens right-aligned."""
        return torch.stack([_left_pad(v, p_max, pad_id) for v in vals], dim=0)

    @staticmethod
    def _stack_generic_tensor(vals):
        """Right-pad generic tensor fields to the longest sample length."""
        target = max(v.shape[-1] for v in vals)
        return torch.stack([_right_pad(v, target, 0) for v in vals], dim=0)

    @staticmethod
    def _stack_resp_only(key, vals, r_max, pad_id):
        """Right-pad response-only fields to ``r_max`` keeping real tokens left-aligned."""
        pad_val = pad_id if key == "responses" else 0
        return torch.stack([_right_pad(v, r_max, pad_val) for v in vals], dim=0)

    def _stack_full_seq(self, key, vals, samples, real_prompt_lens, p_max, r_max, pad_id):
        """Rebuild full-sequence fields as ``[pad_prompt | prompt | response | pad_resp]``."""
        out = []
        pad_val = pad_id if key == "input_ids" else 0
        for i, v in enumerate(vals):
            v = torch.as_tensor(v)
            am = torch.as_tensor(samples[i]["attention_mask"]).bool()
            real = v[am]
            p_i = real_prompt_lens[i]
            prompt_part, resp_part = real[:p_i], real[p_i:]
            p_pad = p_max - p_i
            r_pad = r_max - (real.numel() - p_i)
            out.append(torch.cat([
                torch.full((p_pad,), pad_val, dtype=v.dtype),
                prompt_part.to(v.dtype),
                resp_part.to(v.dtype),
                torch.full((r_pad,), pad_val, dtype=v.dtype),
            ], dim=0))
        return torch.stack(out, dim=0)

    @staticmethod
    def _merge_list_field(vals):
        """Stack list/scalar fields into a tensor when possible, otherwise keep list."""
        extracted = [_extract_scalar(v) for v in vals]
        try:
            return torch.tensor(extracted)
        except (ValueError, TypeError):
            return extracted

    def _log_sample_batch(self, weight_versions):
        """Log a brief summary of the collected batch's weight version distribution."""
        if not weight_versions:
            logger.info("[async-trainer] collected 0 samples, version_dist={}")
            return
        version_counts = {}
        for v in weight_versions:
            version_counts[v] = version_counts.get(v, 0) + 1
        latest_version = max(weight_versions)
        stale_count = sum(1 for v in weight_versions if latest_version - v > 0)
        stale_ratio = stale_count / len(weight_versions)
        logger.info(
            f"[async-trainer] collected {len(weight_versions)} samples, "
            f"version_dist={version_counts}, "
            f"latest_rollout_version={latest_version}, "
            f"trainer_version={self.current_param_version}, "
            f"stale_count={stale_count}/{len(weight_versions)} ({stale_ratio:.1%})"
        )

    def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """Get samples from the data manager or streaming SampleQueue.

        When ``self.sample_queue`` is set (fully_async mode), samples are pulled
        one sub-batch at a time from the SampleQueue actor and concatenated until
        ``required_samples`` trajectories are gathered. Otherwise (one_step_off
        mode) the original data_manager path is used unchanged.
        """
        if self.sample_queue is not None:
            return self._get_samples_from_streaming_queue()
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
            except Exception as e:
                logger.error(f"[FullyAsyncTrainer] unexpected error in fit_step: {e}", exc_info=True)
                self._shutdown_sample_queue("[FullyAsyncTrainer] SampleQueue shutdown signaled on error")
                raise

        self.progress_bar.close()
        await self._maybe_sync_final_weights()
        self._fit_save_checkpoint(force=True)
        self._shutdown_sample_queue("[async-trainer] SampleQueue shutdown signaled")
        self.controller.finish_training()
        logger.info("fully async training finished!!!")

    async def _maybe_sync_final_weights(self) -> None:
        """Sync the final weights to rollout if the loop ended mid-trigger cycle."""
        if self.current_param_version % self.config.trainer.test_freq == 0 and self.local_trigger_step <= 1:
            return
        logger.info(
            f"[FullyAsyncTrainer] Training loop ended, syncing final weights to rollout "
            f"(param_version={self.current_param_version}, local_trigger_step={self.local_trigger_step}, "
            f"trigger_parameter_sync_step={self.trigger_parameter_sync_step})"
        )
        await self._fit_update_weights()

    def _shutdown_sample_queue(self, success_msg: str) -> None:
        """Best-effort shutdown of the SampleQueue actor if present."""
        if getattr(self, "sample_queue", None) is None:
            return
        try:
            self.sample_queue.shutdown.remote()
            logger.info(success_msg)
        except Exception as e:
            logger.warning(f"[async-trainer] failed to shutdown SampleQueue: {e}")

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
        logger.info(f"finished train iteration {self.global_steps - 1}/{self.total_train_steps}")

        if self.global_steps > self.total_train_steps:
            raise TrainingStopException("end of training loop")

    async def _fit_generate(self) -> DataProto | None:
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            # 数据获取会循环等待, 直到有数据获取到, 所以不需要对rollout_return_batch判空处理
            epoch, rollout_return_batch = self._get_samples_from_queue()
            if epoch is None or rollout_return_batch is None:
                logger.warning("[FullyAsyncTrainer] No samples available from queue, skipping generation step")
                return None
            batch = DataProto.from_dict(tensors=self._extract_generate_tensors(rollout_return_batch))
            batch.non_tensor_batch["uid"] = np.array(rollout_return_batch["prompt_ids"])
            batch.meta_info["global_token_num"] = torch.sum(rollout_return_batch["attention_mask"], dim=-1).tolist()
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return batch

    @staticmethod
    def _extract_generate_tensors(rollout_return_batch):
        """Build the tensor dict consumed by ``DataProto.from_dict`` from a rollout batch."""
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
        return tensors

    def _compute_old_log_prob(self, batch: DataProto):
        """Compute old log probabilities with MIS (multiple importance sampling) restore.

        When ``local_trigger_step == 1`` the actor weights are already current,
        so we simply save a CPU snapshot. Otherwise we swap the trainer's
        snapshot to version 1 for the forward pass, then restore the trainer's
        working version to keep MIS bookkeeping consistent.
        """
        mis_enter_version = self.current_param_version
        mis_enter_trigger = self.local_trigger_step
        if self.local_trigger_step == 1:
            self.actor_rollout_wg.save_model_to_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
        else:
            self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            self.actor_rollout_wg.restore_model_from_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
            self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
        logger.info(
            f"[async-trainer] MIS done: trainer_version={mis_enter_version}, "
            f"trigger_step={mis_enter_trigger}"
        )
        return old_log_prob, old_log_prob_mfu

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
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        if not self._should_save_now(force, esi_close_to_expiration):
            return
        if esi_close_to_expiration:
            logger.info("Force saving checkpoint: ESI instance expiration approaching.")
        with marked_timer("save_checkpoint", self.timing_raw, color="green"):
            # sleep replicas to avoid OOM during checkpoint saving
            self._save_checkpoint()
            self.last_ckpt_version = self.current_param_version

    def _should_save_now(self, force, esi_close_to_expiration):
        """Decide whether a checkpoint should be saved at the current step.

        Conditions: (1) ``save_freq`` is positive, and (2/3/4) ESI is close to
        expiration, or this is a forced save landing on a save_freq boundary.
        """
        if self.config.trainer.save_freq <= 0:
            return False
        if esi_close_to_expiration:
            return True
        return force and self.current_param_version % self.config.trainer.save_freq == 0

    def _fit_postprocess_step(self) -> None:
        self.global_steps += 1

        self.metrics_aggregator.add_step_metrics(
            metrics=self.metrics, sample_count=self.required_samples, timestamp=time.time()
        )

        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=self.global_steps - 1,
        )
        self.metrics_aggregator.reset()

        self.progress_bar.update(1)
        self.progress_bar.set_postfix(
            weight_version=self.current_param_version,
            sync=f"{self.local_trigger_step}/{self.trigger_parameter_sync_step}",
        )

    def _save_checkpoint(self) -> None:
        """Persist actor (and optionally critic) model checkpoints to disk."""
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )
        logger.info(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = self._build_remote_path("actor")
        max_actor_ckpt_to_keep, max_critic_ckpt_to_keep = self._resolve_ckpt_keep_counts()
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )
        self._save_critic_checkpoint(local_global_step_folder, max_critic_ckpt_to_keep)
        self._write_latest_iteration()

    def _build_remote_path(self, sub_dir):
        """Build the HDFS path for a checkpoint sub-directory, or ``None`` when HDFS is disabled."""
        if self.config.trainer.default_hdfs_dir is None:
            return None
        return os.path.join(
            self.config.trainer.default_hdfs_dir,
            f"global_step_{self.current_param_version}",
            sub_dir,
        )

    def _resolve_ckpt_keep_counts(self):
        """Resolve max ckpt-to-keep settings, honoring the deprecated ``remove_previous_ckpt_in_save``."""
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            logger.warning(
                "[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
            return 1, 1
        max_actor = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        max_critic = self.config.trainer.get("max_critic_ckpt_to_keep", None)
        return max_actor, max_critic

    def _save_critic_checkpoint(self, local_global_step_folder, max_critic_ckpt_to_keep):
        """Save the critic checkpoint if a critic worker group is in use."""
        if not self.use_critic:
            return
        critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
        critic_remote_path = self._build_critic_remote_path()
        self.critic_wg.save_checkpoint(
            critic_local_path,
            critic_remote_path,
            self.current_param_version,
            max_ckpt_to_keep=max_critic_ckpt_to_keep,
        )

    def _build_critic_remote_path(self):
        """Build the HDFS path for the critic checkpoint, or ``None`` when HDFS is disabled."""
        if self.config.trainer.default_hdfs_dir is None:
            return None
        hdfs_folder = self.config.trainer.default_hdfs_dir + f"global_step_{self.current_param_version}"
        return os.path.join(hdfs_folder, str(Role.Critic))

    def _write_latest_iteration(self):
        """Persist the current parameter version to ``latest_checkpointed_iteration.txt``."""
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
        global_step_folder = self._resolve_checkpoint_folder()
        if global_step_folder is None:
            return 0
        self._restore_from_checkpoint(global_step_folder)
        return self.current_param_version

    def _resolve_checkpoint_folder(self):
        """Resolve the checkpoint folder based on ``resume_mode`` (local only)."""
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        checkpoint_folder = self._resolve_local_checkpoint_folder()
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if self.config.trainer.resume_mode == "auto":
            return global_step_folder
        return self._resolve_resume_path(global_step_folder)

    def _resolve_local_checkpoint_folder(self):
        """Resolve the local checkpoint folder, anchored to CWD when not absolute."""
        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        return checkpoint_folder

    def _resolve_resume_path(self, global_step_folder):
        """Resolve and validate a user-specified ``resume_path`` checkpoint folder."""
        if self.config.trainer.resume_mode != "resume_path":
            return global_step_folder
        if not isinstance(self.config.trainer.resume_from_path, str):
            raise ValueError("resume ckpt must be str type")
        if "global_step_" not in self.config.trainer.resume_from_path:
            raise ValueError("resume ckpt must specify the global_steps")
        global_step_folder = self.config.trainer.resume_from_path
        if not os.path.isabs(global_step_folder):
            global_step_folder = os.path.join(os.getcwd(), global_step_folder)
        return global_step_folder

    def _restore_from_checkpoint(self, global_step_folder):
        """Restore model weights and global step counters from a checkpoint folder."""
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
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )

        if self.use_critic:
            critic_path = os.path.join(global_step_folder, str(Role.Critic))
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
        logger.info("[FullyAsyncTrainer] End loading checkpoint for resume ...")

    def _collect_metrics_from_samples(self, batch: DataProto, metrics: dict) -> None:
        """Collect staleness and async metrics from the sample batch."""
        if not self._has_sample_meta_info(batch):
            return
        stale_traj_count = self._count_stale_trajectories(batch.meta_info["trajectory_param_versions"])
        self.stale_trajectory_processed += stale_traj_count
        metrics.update(
            {
                "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                "fully_async/count/current_param_version": self.current_param_version,
            }
        )
        for key, value in batch.meta_info.items():
            if self._is_async_metric_key(key):
                metrics[key] = value

    @staticmethod
    def _has_sample_meta_info(batch):
        """Check whether the batch carries ``meta_info`` with staleness data."""
        return hasattr(batch, "meta_info") and bool(batch.meta_info)

    @staticmethod
    def _is_async_metric_key(key):
        """Check whether a meta-info key should be exposed as an async metric."""
        return key.startswith("fully_async") or key.startswith("timing_s")

    def _count_stale_trajectories(self, trajectory_param_versions):
        """Count trajectories whose parameter version lags the trainer's by at least one."""
        return sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)

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
