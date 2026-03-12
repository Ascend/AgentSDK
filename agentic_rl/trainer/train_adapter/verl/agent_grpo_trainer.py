# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
# ruff: noqa: E402
from typing import Optional, List
from functools import reduce
import os
import math
import uuid
import asyncio
import torch
import ray
import numpy as np
from tqdm import tqdm
from ray.exceptions import RayError
from transformers import PreTrainedTokenizerBase
from torch.utils.tensorboard import SummaryWriter

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    RayWorkerGroup,
    ResourcePoolManager,
    compute_advantage,
    marked_timer,
)
from verl.utils.metric import reduce_metrics
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_timing_metrics, compute_throughout_metrics
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.configs.agentic_rl_config import AgenticRLConfig, GenConfig
from agentic_rl.trainer.rollout.rollout_worker import RolloutWorker
from agentic_rl.trainer.train_adapter.verl import patch

patch.apply_patch()


logger = Loggers(__name__)


def truncate_rows(tensor, index_tensor, left_pad=False):
    if tensor.dim() < 2:
        raise ValueError(
            f"truncate_rows expects tensor with at least 2 dimensions, got shape {tensor.shape}"
        )
    if tensor.shape[0] != index_tensor.shape[0]:
        raise ValueError(
            f"truncate_rows batch size mismatch: tensor.shape[0]={tensor.shape[0]} "
            f"!= index_tensor.shape[0]={index_tensor.shape[0]}"
        )
    mbs = tensor.shape[0]
    max_idx = tensor.shape[1]
    indices = torch.arange(max_idx).unsqueeze(0).repeat(mbs, 1).to(tensor.device)
    if left_pad:
        mask = indices >= (max_idx - index_tensor.unsqueeze(1))
    else:
        mask = indices < index_tensor.unsqueeze(1)

    truncated_tensor = torch.where(mask, tensor, torch.tensor(-1, dtype=tensor.dtype, device=tensor.device))
    truncated_tensor = [truncated_tensor[i, mask[i]].cpu() for i in range(mbs)]
    return truncated_tensor


def remove_padding_tensor_dict_to_dict(data_dict):
    remove_padding_tensors = {}
    if data_dict is None:
        return remove_padding_tensors
    if "original_length" not in data_dict.keys():
        return data_dict
    data_lengths = data_dict["original_length"]
    for idx, (key, dict_value) in enumerate(data_dict.items()):
        if key == "original_length":
            continue
        remove_padding_tensors[key] = truncate_rows(
            dict_value, data_lengths[idx * len(dict_value): (idx + 1) * len(dict_value)]
        )

    return remove_padding_tensors


def remove_padding_and_split_to_list(responses, eos_token_id: int, pad_token_id: int, to_list: bool = False):
    output = []
    for i in range(responses.shape[0]):
        response = responses[i]
        nonzeros = torch.nonzero(response == pad_token_id, as_tuple=False)
        if len(nonzeros) != 0:
            first_pad_index = nonzeros[0][0]
        else:
            first_pad_index = len(response)
        if pad_token_id == eos_token_id:
            response = response[: first_pad_index + 1]
        else:
            response = response[:first_pad_index]
        if to_list:
            response = response[:-1].cpu().numpy().tolist()
        output.append(response)

    return output


class AgentGRPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls,
        reward_fn=None,
        val_reward_fn=None,
        tokenizer_path=None,
        dataset_additional_keys=None,
        generate_config: Optional[GenConfig] = None,
        agentic_rl_config: Optional[AgenticRLConfig] = None,
        use_tensorboard: bool = False,
        tensorboard_flush_interval: int = 50,
        tensorboard_log_dir: Optional[str] = None,
        console_metrics: Optional[List[str]] = None,
    ):
        self._check_args(
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            (generate_config, agentic_rl_config),
        )

        try:
            super().__init__(
                config=config,
                tokenizer=tokenizer,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
            )
        except AttributeError as e:
            logger.error(f"trainer does not have the attribute: {e}")
            raise e
        except Exception as e:
            logger.error(f"Failed to initialize RayPPOTrainer: {e}")
            raise e

        self.config = config
        self.tokenizer_path = tokenizer_path
        self.dataset_additional_keys = dataset_additional_keys
        self.generate_config = generate_config
        self.agentic_rl_config = agentic_rl_config
        self.rollout_worker = None
        self.is_last_step = False

        # Initialize tensorboard if enabled
        self.use_tensorboard = use_tensorboard
        self.tensorboard_flush_interval = tensorboard_flush_interval
        self.console_metrics = console_metrics or self._default_console_metrics()
        self.tensorboard_writer = None
        if self.use_tensorboard:
            log_dir = tensorboard_log_dir or "tensorboard_logs"
            project_name = self.config.trainer.get("project_name", "default-agent")
            experiment_name = self.config.trainer.get("experiment_name", "default-experiment")
            log_dir = os.path.join(log_dir, f"{project_name}/{experiment_name}")
            os.makedirs(log_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"Tensorboard logging enabled. Log directory: {log_dir}")

    def init_workers(self):
        super().init_workers()
        n_samples_per_prompt = self.config.actor_rollout_ref.rollout.n
        max_prompt_length = self.generate_config.max_num_batched_tokens
        actor_rollout_dispatch_size = n_samples_per_prompt
        try:
            self.rollout_worker = RolloutWorker.remote(
                n_parallel_agents=n_samples_per_prompt,
                max_prompt_length=max_prompt_length,
                actor_rollout_dispatch_size=actor_rollout_dispatch_size,
                tokenizer_name_or_path=self.tokenizer_path,
                dataset_additional_keys=self.dataset_additional_keys,
                generate_config=self.generate_config,
                agentic_rl_config=self.agentic_rl_config,
                worker_group=self.actor_rollout_wg,
                remove_padding_tensor_dict_to_dict=remove_padding_tensor_dict_to_dict,
                remove_padding_and_split_to_list=remove_padding_and_split_to_list,
            )
            ray.get(self.rollout_worker.wait_init_finished.remote())
        except AttributeError as e:
            logger.error(f"rollout worker does not have the attribute: {e}")
            raise e
        except RayError as e:
            logger.error(f"rollout worker init by ray failed: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error occurred when init rollout worker: {e}")
            raise e

    def _check_worker_health(self):
        """Check health of all workers"""
        try:
            # Check rollout worker health
            ray.get(self.rollout_worker.wait_init_finished.remote(), timeout=10)
            logger.info("Rollout worker health check passed")
            return True
        except Exception as e:
            logger.error(f"Worker health check failed: {e}")
            return False
    
    def _check_numerical_stability(self, batch):
        """Check for numerical stability issues in batch data"""
        try:
            for key, value in batch.batch.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any():
                        logger.error(f"NaN values found in {key}")
                        return False
                    if torch.isinf(value).any():
                        logger.error(f"Infinite values found in {key}")
                        return False
                    if key == "token_level_scores" and value.std().item() < 1e-6:
                        logger.warning("Very low reward variance detected")
            return True
        except Exception as e:
            logger.error(f"Numerical stability check failed: {e}")
            return False

    @staticmethod
    def _check_args(
        tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, agentic_configs
    ):
        generate_config, agentic_rl_config = agentic_configs
        if tokenizer is not None and not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError("tokenizer must be an instance of PreTrainedTokenizerBase")
        if not isinstance(role_worker_mapping, dict):
            raise ValueError("role_worker_mapping must be a dict")
        if resource_pool_manager is None or not isinstance(resource_pool_manager, ResourcePoolManager):
            raise ValueError("resource_pool_manager must be an instance of ResourcePoolManager")
        if ray_worker_group_cls is None or not issubclass(ray_worker_group_cls, RayWorkerGroup):
            raise ValueError("ray_worker_group_cls must be a subclass of RayWorkerGroup")
        if generate_config is None or not isinstance(generate_config, GenConfig):
            raise ValueError("generate_config must be an instance of GenConfig")
        if agentic_rl_config is None or not isinstance(agentic_rl_config, AgenticRLConfig):
            raise ValueError("agentic_rl_config must be an instance of AgenticRLConfig")

    @staticmethod
    def _default_console_metrics() -> List[str]:
        return [
            "actor/entropy",
            "actor/grad_norm",
            "actor/lr",
            "traj/steps_mean",
            "traj/env_time_mean",
            "traj/llm_time_mean",
            "traj/total_time_mean",
            "perf/max_memory_allocated_gb",
            "perf/max_memory_reserved_gb",
            "perf/cpu_memory_used_gb",
            "timing_s/update_actor",
            "timing_per_token_ms/update_actor",
            "batch/solve_all",
            "batch/solve_partial",
            "batch/solve_none",
        ]

    def _format_metrics_for_console(self, metrics):
        result = []
        for key in self.console_metrics:
            if key in metrics:
                val = metrics[key]
                if isinstance(val, float):
                    if abs(val) < 0.0001 or abs(val) >= 10000:
                        result.append(f"{key}={val:.2e}")
                    else:
                        result.append(f"{key}={val:.4f}")
                else:
                    result.append(f"{key}={val}")
        return " | ".join(result)

    def _prepare_batch(self, batch_dict):
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
        return batch

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch
        world_size = reduce(math.lcm, world_sizes)
        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        for i in range(pad_size):
            idx = original_batch_size + i
            if "is_last_step" in batch.non_tensor_batch:
                batch.non_tensor_batch["is_last_step"][idx] = False
            if "is_pad_step" in batch.non_tensor_batch:
                batch.non_tensor_batch["is_pad_step"][idx] = True
        return batch

    def _reject_low_reward_sequences(self, batch, reward_tensor, metrics):
        uids = batch.non_tensor_batch["uid"]
        unique_uids = np.unique(uids)
        solve_none = 0
        solve_all = 0

        for uid in unique_uids:
            uid_mask = uids == uid
            uid_rewards = reward_tensor[uid_mask].sum(-1)
            if (uid_rewards <= 0).all():
                solve_none += 1
            elif (uid_rewards >= 1).all():
                solve_all += 1

        metrics["batch/solve_none"] = solve_none
        metrics["batch/solve_all"] = solve_all
        metrics["batch/solve_partial"] = len(unique_uids) - solve_none - solve_all

    def _compute_rollout_probs_diff(self, batch):
        if "rollout_log_probs" not in batch.batch.keys():
            return {}
        rollout_old_log_probs = batch.batch["rollout_log_probs"]
        actor_old_log_probs = batch.batch["actor_log_probs"]
        attention_mask = batch.batch["attention_mask"]
        responses = batch.batch["responses"]
        response_length = responses.size(1)
        response_mask = attention_mask[:, -response_length:]

        rollout_probs = torch.exp(rollout_old_log_probs)
        actor_probs = torch.exp(actor_old_log_probs)
        rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
        rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
        rollout_probs_diff_max = torch.max(rollout_probs_diff)
        rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
        rollout_probs_diff_std = torch.std(rollout_probs_diff)
        return {
            "training/rollout_probs_diff_max": rollout_probs_diff_max.item(),
            "training/rollout_probs_diff_mean": rollout_probs_diff_mean.item(),
            "training/rollout_probs_diff_std": rollout_probs_diff_std.item(),
        }

    def _compute_rewards_and_advantages(self, batch, metrics, timing_raw):
        # Compute reward model score
        if self.use_rm:
            reward_tensor = self.rm_wg.compute_rm_score(batch)
            batch = batch.union(reward_tensor)

        # Compute reward function
        if "token_level_scores" not in batch.batch.keys():
            reward_tensor = self.reward_fn(batch)
            batch.batch["token_level_scores"] = reward_tensor
        else:
            reward_tensor = batch.batch["token_level_scores"]

        # Rejection sampling
        self._reject_low_reward_sequences(batch, reward_tensor, metrics)
        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
        batch = batch.union(old_log_prob)
        # Compute entropy
        entropys = old_log_prob.batch["entropys"]
        response_masks = batch.batch["response_mask"]
        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
        metrics["actor/entropy"] = entropy_agg.detach().item()
        # Warn about very low entropy which may indicate policy collapse
        entropy_val = entropy_agg.detach().item()
        if entropy_val < 0.1:
            logger.warning(f"Entropy is very low: {entropy_val}. This may indicate policy collapse.")

        old_log_prob.batch.pop("entropys")
        batch = batch.union(old_log_prob)
        # Compute rollout probs diff
        rollout_metrics = self._compute_rollout_probs_diff(batch)
        metrics.update(rollout_metrics)

        # Reference policy
        if self.use_reference_policy:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

        # Compute rewards with KL penalty if needed
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        # Safeguard: when all rewards are zero, record this in meta_info so
        # that the actor update can be skipped for this step to avoid
        # reinforcing a collapsed policy or triggering unstable updates on
        # degenerate batches.
        reward_tensor = batch.batch["token_level_scores"]
        if isinstance(reward_tensor, torch.Tensor):
            all_rewards_zero = reward_tensor.numel() > 0 and reward_tensor.max().item() == 0
        else:
            all_rewards_zero = False
        if batch.meta_info is None:
            batch.meta_info = {}
        batch.meta_info["all_rewards_zero"] = all_rewards_zero
        # Compute advantages
        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
            config=self.config.algorithm,
        )

    def _update_actor_and_critic(self, batch, metrics, timing_raw):
        # Update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw):
                critic_output = self.critic_wg.update_critic(batch)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update(critic_output_metrics)

        # Only update actor if enough steps have elapsed since critic warmup
        if self.config.trainer.critic_warmup <= self.global_steps:
            skip_actor = batch.meta_info.get("all_rewards_zero", False)
            if skip_actor:
                logger.warning(
                    f"Train step {self.global_steps}: skipping actor update because all rewards in batch are zero."
                )
                return

            # Check worker health
            if not self._check_worker_health():
                logger.error(f"Train step {self.global_steps}: Worker health check failed, skipping actor update")
                return

            # Check numerical stability
            if not self._check_numerical_stability(batch):
                logger.error(f"Train step {self.global_steps}: Numerical stability check failed, skipping actor update")
                return

            # Wrapper function to run actor update with timeout
            async def update_actor_with_timeout():
                timeout_seconds = 120  # 2 minutes

                def update_actor_sync():
                    return self.actor_rollout_wg.update_actor(batch)

                try:
                    result = await asyncio.wait_for(asyncio.to_thread(update_actor_sync), timeout=timeout_seconds)
                    return result
                except asyncio.TimeoutError:
                    logger.error(
                        f"Train step {self.global_steps}: actor update timed out after {timeout_seconds} seconds"
                    )
                    return None
                except Exception as e:
                    logger.error(f"Train step {self.global_steps}: Error during actor update: {e}")
                    raise

            # Log batch size and sequence length
            batch_size = batch.batch["input_ids"].shape[0]
            seq_length = batch.batch["input_ids"].shape[1]
            logger.info(f"Actor update batch size: {batch_size}, sequence length: {seq_length}")

            # Log reward stats
            rewards = batch.batch["token_level_scores"]
            min_reward = rewards.min().item()
            max_reward = rewards.max().item()
            mean_reward = rewards.mean().item()
            logger.info(f"Reward stats - min: {min_reward:.4f}, max: {max_reward:.4f}, mean: {mean_reward:.4f}")

            # Actually run the update
            with marked_timer("update_actor", timing_raw):
                logger.info(f"Starting actor update for step {self.global_steps}")
                actor_output = asyncio.run(update_actor_with_timeout())
                if actor_output is None:
                    logger.warning(f"Train step {self.global_steps}: actor update skipped due to timeout")
                    return
                logger.info(f"Actor update completed for step {self.global_steps}")

            # Log actor output metrics and update overall metrics
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            logger.info(f"Actor output metrics: {actor_output_metrics}")
            metrics.update(actor_output_metrics)

    def shutdown(self):
        """Shutdown the trainer."""
        self._flush_tensorboard()
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

    def _should_save_checkpoint(self) -> bool:
        save_freq = self.config.trainer.save_freq
        if save_freq <= 0 or self.global_steps <= 0:
            return False
        return (
            self.global_steps % save_freq == 0 or self.is_last_step
        )

    def _save_checkpoint(self):
        step = self.global_steps
        ckpt_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")
        try:
            super()._save_checkpoint()
            logger.info(f"Checkpoint saved successfully | global_steps={step} | path={ckpt_dir}")
        except (OSError, RayError, RuntimeError) as e:
            logger.error(f"Checkpoint save failed: {e} | global_steps={step} | path={ckpt_dir}")
            raise RuntimeError(f"Failed to save checkpoint at step {step}: {e}") from e

    def _train_step(self, batch):
        metrics = {}
        timing_raw = {}

        try:
            with marked_timer("step", timing_raw):
                final_gen_batch_output, generate_metrics = ray.get(
                    self.rollout_worker.generate_sequences_verl.remote(batch=batch)
                )
                if batch.batch.is_locked:
                    batch.batch.unlock_()
                batch = batch.union(final_gen_batch_output)
                metrics.update(generate_metrics)
                # Compute values
                if self.use_critic:
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)
                # Compute rewards and advantages
                self._compute_rewards_and_advantages(batch, metrics, timing_raw)
                # Balance batch
                batch = self._pad_dataproto_to_world_size(batch=batch)
                self._balance_batch(batch, metrics=metrics)
                # Compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
            # Update actor and critic
            self._update_actor_and_critic(batch, metrics, timing_raw)
            # Collect metrics
            metrics.update(compute_data_metrics(batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, n_gpus=n_gpus, timing_raw=timing_raw))
            return metrics
        except RayError as e:
            logger.error(f"Rollout worker generate sequences failed: {e}")
            raise e
        except Exception as e:
            logger.error(f"Failed to train step: {e}")
            raise e

    def _log_metrics_to_tensorboard(self, metrics, step):
        """
        Log metrics to tensorboard.
        
        Args:
            metrics (dict): A dictionary of metrics to log.
            step (int): The current training step.
        """
        if self.tensorboard_writer is None:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value):
                self.tensorboard_writer.add_scalar(f"train/{key}", value, step)
        if step % self.tensorboard_flush_interval == 0:
            self._flush_tensorboard()

    def _flush_tensorboard(self):
        if self.tensorboard_writer is not None:
            try:
                self.tensorboard_writer.flush()
                logger.info("Tensorboard flushed")
            except (OSError, RuntimeError) as e:
                logger.error(f"Failed to flush tensorboard: {e}")

    def fit(self):
        """The training loop of GRPO"""
        self.global_steps = 0
        try:
            self._load_checkpoint()
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error occurred when loading checkpoint: {e}")
            raise e
        self.is_last_step = False
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        epoch = 0
        try:
            while epoch < self.config.trainer.total_epochs and not self.is_last_step:
                logger.info(f"Epoch {epoch}, step {self.global_steps} started.")
                for batch_dict in self.train_dataloader:
                    batch = self._prepare_batch(batch_dict)
                    metrics = self._train_step(batch)
                    progress_bar.update(1)
                    self.global_steps += 1
                    self._log_metrics_to_tensorboard(metrics, self.global_steps)
                    self.is_last_step = self.global_steps >= self.total_training_steps
                    if self._should_save_checkpoint():
                        logger.info(f"Step {self.global_steps}, metrics: {metrics}")
                        self._save_checkpoint()
                    if self.is_last_step:
                        break
                epoch += 1
            logger.info("GRPO training finished.")
        except (ValueError, RuntimeError) as e:
            logger.error(f"Failed to fit: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error occurred when fitting: {e}")
            raise e
        finally:
            progress_bar.close()
            self._flush_tensorboard()