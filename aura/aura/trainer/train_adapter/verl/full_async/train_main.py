# -*- coding: utf-8 -*-
#
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2023-2024 Bytedance Ltd. and/or its affiliates
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
from pprint import pprint

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from omegaconf import OmegaConf

from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.utils import Role
from verl.trainer.main_ppo import run_ppo
from verl.utils import hf_processor, hf_tokenizer

from aura.controllers.utils.utils import DEFAULT_SLEEP_TIME
from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def start_train_controller(actor_worker, config):
    from aura.controllers.train_controller.train_controller import TrainController
    from aura.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader import default_train_dataloader

    controller = TrainController(
        actor_worker=actor_worker,
        global_batch_size=config.extras.data_loader.global_batch_size,
        n_samples_per_prompt=config.rollout.n,
        validate_num_samples=config.extras.validate_num_samples,
        init_num_group_batches=config.extras.init_num_group_batches,
        max_queue_size=config.extras.max_queue_size,
        train_iters=config.extras.data_loader.train_iters,
        weight_save_dir=config.extras.weight_save_dir,
        delta=config.extras.delta,
        data_loader=config.extras.data_loader,
        initialize_rollout_dataloader=default_train_dataloader,
        consumed_train_samples=config.extras.consumed_train_samples,
        data_optimized=False,
    )
    return controller


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
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        logger.info(f"[ASYNC MAIN] config={config}")

        local_path = config.actor_rollout_ref.model.path
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        logger.info("[ASYNC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers import (
                FsdpDetachActorWorker as DetachActorWorker,
            )
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from aura.trainer.train_adapter.verl.full_async.workers.megatron_worker import (
                MegatronDetachActorWorker as DetachActorWorker,
            )
        else:
            raise NotImplementedError(f"Unsupported strategy: {config.actor_rollout_ref.actor.strategy}")
        role_worker_mapping[Role.Actor] = ray.remote(DetachActorWorker)

        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        logger.info("[ASYNC MAIN] Creating FullyAsyncTrainer...")
        self._create_trainer(config)

        logger.info(f"total_train_steps {config.total_train_steps}")
        ray.get(self.components["trainer"].set_total_train_steps.remote(config.total_train_steps))

        controller = start_train_controller(ray.get(self.components["trainer"].get_actor_wg.remote()), config)
        controller.pre_initialize()
        controller.wait_for_rollout_unit_ready()
        controller.initialize_rollout()

        from aura.data_manager.data_manager import DataManager

        data_manager = DataManager(train_backend="verl", service_mode="train")
        data_manager.sync_init_data_manager(controller)
        pad_token_id = data_manager.set_pad_token_id_from_tokenizer(self.components["tokenizer"])
        logger.info(f"[ASYNC MAIN] DataManager pad_token_id set to {pad_token_id} (from tokenizer)")

        ray.get(self.components["trainer"].set_data_manager.remote(data_manager))
        ray.get(self.components["trainer"].set_controller.remote(controller))
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

        from aura.trainer.train_adapter.verl.full_async.full_async_trainer import FullyAsyncTrainer

        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
            delta=config.extras.delta,
            weight_save_dir=config.extras.weight_save_dir,
            update_weights_interval=config.extras.update_weights_interval,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        logger.info("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")

    def _run_training_loop(self) -> None:
        """Run the main training loop, handling exceptions gracefully."""
        self.running = True

        logger.info("[ASYNC MAIN] Starting Trainer...")
        trainer_future = self.components["trainer"].fit.remote()
        futures = [trainer_future]
        try:
            while futures:
                # Use ray.wait to monitor all futures and return when any one is completed.
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    try:
                        ray.get(future)
                        logger.info("[ASYNC MAIN] One component completed successfully")
                    except Exception as e:
                        logger.error(f"[ASYNC MAIN] Component failed with error: {e}")
                        for remaining_future in remaining_futures:
                            ray.cancel(remaining_future)
                        raise e

                futures = remaining_futures

        except Exception as e:
            logger.error(f"[ASYNC MAIN] Training failed: {e}")
            for future in futures:
                ray.cancel(future)
            raise
        finally:
            logger.info("[ASYNC MAIN] Training completed or interrupted")


@ray.remote
def start_train(work_mode, train_config, rollout_config, agent_service, infer_service):
    from aura.trainer.rollout.rollout_service import start_rollout

    # 异步提交 rollout 任务
    ref = start_rollout.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().node_id,
            soft=False,  # 强制硬亲和
        )
    ).remote(rollout_config=rollout_config, agent_service=agent_service, infer_service=infer_service)

    # 在新线程里监控 rollout 异常
    def monitor_rollout():
        try:
            ray.get(ref)  # 这里阻塞直到 rollout 完成或抛异常
        except Exception as e:
            logger.error(f"Rollout failed: {e}", exc_info=True)
            os._exit(1)

    import threading

    threading.Thread(target=monitor_rollout, daemon=True).start()

    # 主流程继续执行，不会被阻塞
    logger.info(f"{train_config=}, {work_mode=}")

    if not hasattr(train_config, "async_training"):
        raise RuntimeError("must set async_training config")

    from time import time

    start_time = time()
    run_ppo(train_config, task_runner_class=FullyAsyncTaskRunner)
    logger.info(f"total time: {time() - start_time:.2f} seconds")


@ray.remote
def start_dummy_train(work_mode, train_config, rollout_config, agent_service, infer_service):
    from aura.trainer.rollout.rollout_service import start_rollout

    # 异步提交 rollout 任务
    start_rollout.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().node_id,
            soft=False,  # 强制硬亲和
        )
    ).remote(rollout_config=rollout_config, agent_service=agent_service, infer_service=infer_service)

    # 主流程继续执行，不会被阻塞
    logger.info(f"{train_config=}, {work_mode=}")

    # 只启动train controller即可
    controller = start_train_controller(None, train_config)
    controller.initialize_dispatch()
    controller.initialize_train_server()
    controller.wait_for_rollout_unit_ready()
    controller.initialize_rollout()

    import time

    while True:
        time.sleep(DEFAULT_SLEEP_TIME)
        finished = controller.data_iter_complete()
        if finished:
            controller.finish_training()
            break
    ray.shutdown()
