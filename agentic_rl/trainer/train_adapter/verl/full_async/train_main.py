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
import os
import socket
import threading

import ray
from omegaconf import OmegaConf

from recipe.fully_async_policy.fully_async_main import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.utils import Role

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


@ray.remote(num_cpus=1)
class FullyAsyncTaskRunner:
    """Ray remote task runner for fully asynchronous PPO training."""

    def __init__(self) -> None:
        self.running = False
        self.components: dict = {}
        self.shutdown_event = threading.Event()

    def run(self, config) -> None:
        """Entry point for the async training pipeline.

        Args:
            config: OmegaConf configuration object for the training run.
        """
        logger.info("[ASYNC MAIN] Starting fully async PPO training...")
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        """Create and wire all training components (tokenizer, workers, trainer, controller).

        Args:
            config: OmegaConf configuration object.
        """
        logger.info(f"[ASYNC MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        logger.info(f"[ASYNC MAIN] Config: {OmegaConf.to_container(config, resolve=True)}")
        OmegaConf.resolve(config)

        logger.info(f"[ASYNC MAIN] config={config}")
        from verl.utils import hf_processor, hf_tokenizer

        local_path = config.actor_rollout_ref.model.path
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        logger.info("[ASYNC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from agentic_rl.trainer.train_adapter.verl.full_async.workers.fsdp_workers import (
                FsdpDetachActorWorker as DetachActorWorker,
            )
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from agentic_rl.trainer.train_adapter.verl.full_async.workers.megatron_worker import (
                MegatronDetachActorWorker as DetachActorWorker,
            )
        else:
            raise NotImplementedError(f"Unsupported strategy: {config.actor_rollout_ref.actor.strategy}")
        role_worker_mapping[Role.Actor] = ray.remote(DetachActorWorker)

        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        logger.info("[ASYNC MAIN] Creating FullyAsyncTrainer...")
        self._create_trainer(config)

        self.components["trainer"].set_total_train_steps(config.total_train_steps)

        from agentic_rl.trainer.train_adapter.verl.full_async.param_sync import ParameterSynchronizer

        param_synchronizer = ParameterSynchronizer.remote()
        self.components["trainer"].set_parameter_synchronizer(param_synchronizer)

        from agentic_rl.controllers.train_controller.train_controller import TrainController
        from agentic_rl.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader import (
            default_train_dataloader,
        )

        controller = TrainController(
            actor_worker=self.components["trainer"].get_actor_wg(),
            global_batch_size=config.extras.global_batch_size,
            n_samples_per_prompt=config.extras.n_samples_per_prompt,
            validate_num_samples=config.extras.validate_num_samples,
            init_num_group_batches=config.extras.init_num_group_batches,
            max_queue_size=config.extras.max_queue_size,
            train_iters=config.extras.train_iters,
            weight_save_dir=config.extras.weight_save_dir,
            delta=config.extras.delta,
            data_loader=config.extras.data_loader,
            initialize_rollout_dataloader=default_train_dataloader,
            consumed_train_samples=config.extras.consumed_train_samples,
            data_optimized=False,
        )
        controller.pre_initialize()
        controller.wait_for_rollout_unit_ready()
        controller.initialize_rollout()

        from agentic_rl.data_manager.data_manager import DataManager

        data_manager = DataManager(train_backend="verl", service_mode="train")
        data_manager.sync_init_data_manager(controller)
        pad_token_id = data_manager.set_pad_token_id_from_tokenizer(self.components["tokenizer"])
        logger.info(f"[ASYNC MAIN] DataManager pad_token_id set to {pad_token_id} (from tokenizer)")

        self.components["trainer"].set_data_manager(data_manager)
        self.components["trainer"].set_controller(controller)
        logger.info("[ASYNC MAIN] All components initialized successfully")

    def _create_trainer(self, config) -> None:
        """Instantiate the FullyAsyncTrainer and initialize its workers.

        Args:
            config: OmegaConf configuration object.
        """
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        from agentic_rl.trainer.train_adapter.verl.full_async.full_async_trainer import FullyAsyncTrainer

        trainer = FullyAsyncTrainer(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
            update_weights_interval=config.extras.update_weights_interval,
            weight_save_dir=config.extras.weight_save_dir,
            delta=config.extras.delta,
        )

        trainer.init_workers()
        self.components["trainer"] = trainer
        logger.info("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")

    def _run_training_loop(self) -> None:
        """Run the main training loop, handling exceptions gracefully."""
        self.running = True

        logger.info("[ASYNC MAIN] Starting Trainer...")
        try:
            self.components["trainer"].fit()
        except Exception:
            logger.exception("[ASYNC MAIN] Training failed")
            raise
        finally:
            logger.info("[ASYNC MAIN] Training completed or interrupted")


@ray.remote
def start_train(cluster_mode: str, train_config) -> None:
    """Launch the fully-async PPO training pipeline.

    Args:
        cluster_mode: Cluster deployment mode identifier.
        train_config: OmegaConf training configuration.
    """
    logger.info(f"train_config={train_config}, cluster_mode={cluster_mode}")
    from verl.trainer.main_ppo import run_ppo

    if not hasattr(train_config, "async_training"):
        raise RuntimeError("must set async_training config")

    from time import time

    start_time = time()
    run_ppo(train_config, task_runner_class=FullyAsyncTaskRunner)
    logger.info(f"total time: {time() - start_time:.2f} seconds")
