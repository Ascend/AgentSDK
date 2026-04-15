# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0 OR MulanPSL-2.0
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Amirhossein Kazemnejad and/or its affiliates
# Copyright 2025 Milad Aghajohari and/or its affiliates
# Copyright 2025 Kamran Chitsaz and/or its affiliates
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
import socket

import ray
from omni_rl.train.training_adapter import pgrl

from agentic_rl.trainer.train_adapter.omni_rl.hybrid.ray_trainer import HybridTrainer
from omni_rl.rl.pangu_adapt import Role
from omni_rl.rl.timely_agent.common.logging import ManageLogger
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config

logger = ManageLogger(__file__).get_logger()


@ray.remote
class HybridTaskRunner(TaskRunner):
    """Ray-remote task runner that assembles workers and launches hybrid GSPO training."""

    def __init__(self) -> None:
        """Initialize empty role-worker and resource-pool mappings."""
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config) -> tuple:
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from omni_rl.train.training_adapter import pgrl  # noqa: F811
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)

        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config) -> None:
        """Add critic worker to role mapping."""
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                logger.info("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker

        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    def init_resource_pool_mgr(self, config) -> "ResourcePoolManager":
        """Initialize resource pool manager."""

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # TODO Here you can use the new registration method to support dynamic registration of roles
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        if config.trainer.neural_checker.get("enable_resource_pool", False):
            if config.trainer.neural_checker.n_gpus_per_node <= 0:
                raise ValueError("config.neural_checker.n_gpus_per_node must be greater than 0")
            if config.trainer.neural_checker.nnodes <= 0:
                raise ValueError("config.neural_checker.nnodes must be greater than 0")
            neural_pool = [config.trainer.neural_checker.n_gpus_per_node] * config.trainer.neural_checker.nnodes
            resource_pool_spec["neural_pool"] = neural_pool

        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_worker(self, config) -> None:
        """Add reward model worker if enabled."""

        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    from verl.workers.fsdp_workers import RewardModelWorker
                elif config.reward_model.strategy == "megatron":
                    from verl.workers.megatron_workers import RewardModelWorker
                else:
                    raise NotImplementedError
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import RewardModelWorker

                logger.info("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls) -> None:
        """Add reference policy worker if KL loss or KL reward is used."""

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def add_neural_checker_worker(self, config) -> None:
        """Add neural checker worker if enabled in config."""

        if config.trainer.neural_checker.enable:
            if config.trainer.neural_checker.strategy == "megatron":
                from omni_rl.reward.worker.neural_reward_worker import NeuralRewardWorker
            else:
                logger.info("Using neural check worker implementation")
                raise NotImplementedError

            self.role_worker_mapping[Role.NeuralRewardModel] = ray.remote(NeuralRewardWorker)
            self.mapping[Role.NeuralRewardModel] = "neural_pool"

    def run(self, config) -> None:
        """
        Assemble all workers, datasets, and reward managers, then start GSPO training.

        Args:
            config: Resolved OmegaConf configuration for the full training run.
        """
        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        logger.info(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        logger.info(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        self.add_neural_checker_worker(config)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        if not config.trainer.get("pangu_adapt", False) or \
                config.reward_model.reward_manager == "pangu_genrm":
            # Create training and validation datasets.
            train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
            val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
            train_sampler = create_rl_sampler(config.data, train_dataset)

            # Initialize the GSPO trainer.
            trainer = HybridTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
            )
        else:
            trainer = HybridTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                train_dataset=None,
                val_dataset=None,
                collate_fn=collate_fn,
                train_sampler=None,
            )
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Start the training process.
        trainer.fit()


@ray.remote
def start_train(cluster_mode, train_config) -> None:
    """
    Ray remote entry point that launches PPO training with the HybridTaskRunner.

    Args:
        cluster_mode: Cluster execution mode identifier.
        train_config: OmegaConf training configuration.
    """
    from time import time

    from verl.trainer.main_ppo import run_ppo

    start_time = time()
    run_ppo(train_config, task_runner_class=HybridTaskRunner)
    logger.info("Total training time: %.2f seconds", time() - start_time)
