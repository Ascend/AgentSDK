# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0 OR MulanPSL-2.0
# Copyright 2025 Meituan Ltd. and/or its affiliates
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

        controller_config = config
        from aura.trainer.train_adapter.verl.full_async.replay_recovery import recovery_settings
        joint_recovery = recovery_settings(config)
        if config.trainer.resume_mode != "disable" and joint_recovery is None:
            version = ray.get(self.components["trainer"].load_checkpoint.remote())
            if version > 0:
                completed = version * config.async_training.trigger_parameter_sync_step
                controller_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
                controller_config.total_train_steps = config.total_train_steps - completed
                controller_config.extras.consumed_train_samples = completed * config.data.train_batch_size
                logger.info(f"[ASYNC MAIN] Resume version={version}, completed={completed}, "
                            f"remaining={controller_config.total_train_steps}, "
                            f"consumed_prompts={controller_config.extras.consumed_train_samples}")
        controller = start_train_controller(ray.get(self.components["trainer"].get_actor_wg.remote()), controller_config)
        controller.pre_initialize()
        from aura.memory.replay.config import replay_ratio
        if replay_ratio(config) > 0:
            controller.enable_replay_demand(config.replay.rpc_timeout)
            self.components["replay_controller"] = controller
        controller.wait_for_rollout_unit_ready()
        if joint_recovery is None:
            controller.initialize_rollout()

        from aura.data_manager.data_manager import DataManager

        data_manager = DataManager(train_backend="verl", service_mode="train")
        data_manager.sync_init_data_manager(controller)
        pad_token_id = data_manager.set_pad_token_id_from_tokenizer(self.components["tokenizer"])
        logger.info(f"[ASYNC MAIN] DataManager pad_token_id set to {pad_token_id} (from tokenizer)")

        ray.get(self.components["trainer"].set_data_manager.remote(data_manager))
        ray.get(self.components["trainer"].set_controller.remote(controller))
        if replay_ratio(config) > 0:
            ray.get(self.components["trainer"].set_replay_session.remote(dict(config.replay)))
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

        self._inject_sample_queue(trainer, config)

    def _inject_sample_queue(self, trainer, config):
        # This runner is shared by one-step-off and fully-async entry points.
        # A named queue left by another job must never divert one-step replay.
        if config.extras.work_mode != "fully_async":
            return
        try:
            from aura.controllers.rollout_controller.sample_queue import get_sample_queue
            _sq = get_sample_queue()
            ray.get(trainer.set_sample_queue.remote(_sq))
            logger.info("[ASYNC MAIN] SampleQueue injected into FullyAsyncTrainer (fully_async streaming mode)")
        except Exception as _sq_e:
            logger.error(f"[ASYNC MAIN] Required SampleQueue is unavailable: {_sq_e}")
            raise

    def _run_training_loop(self) -> None:
        """Run the main training loop, handling exceptions gracefully."""
        self.running = True
        logger.info("[ASYNC MAIN] Starting Trainer...")
        trainer_future = self.components["trainer"].fit.remote()
        futures = [trainer_future]
        try:
            futures = self._drain_futures(futures)
        except Exception as e:
            logger.error(f"[ASYNC MAIN] Training failed: {e}")
            self._cancel_all(futures)
            replay_controller = self.components.get("replay_controller")
            if replay_controller is not None:
                # A pure-replay stale-pool error must also wake the idle rollouter.
                try:
                    replay_controller.finish_training()
                except Exception as cleanup_error:
                    logger.error(f"[ASYNC MAIN] Replay shutdown failed: {cleanup_error}")
            raise
        finally:
            logger.info("[ASYNC MAIN] Training completed or interrupted")

    def _drain_futures(self, futures):
        """Wait for all futures to complete, cancelling the rest on first failure."""
        while futures:
            done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)
            for future in done_futures:
                self._handle_completed_future(future, remaining_futures)
            futures = remaining_futures
        return futures

    def _handle_completed_future(self, future, remaining_futures):
        """Process a single completed future, raising on failure after cancelling siblings."""
        try:
            ray.get(future)
            logger.info("[ASYNC MAIN] One component completed successfully")
        except Exception as e:
            logger.error(f"[ASYNC MAIN] Component failed with error: {e}")
            self._cancel_all(remaining_futures)
            raise

    @staticmethod
    def _cancel_all(futures):
        """Cancel every future in ``futures`` (best-effort)."""
        for future in futures:
            ray.cancel(future)


@ray.remote
def start_train(work_mode, train_config, rollout_config, agent_service, infer_service):
    OmegaConf.update(train_config, "extras.work_mode", work_mode, force_add=True)
    from aura.trainer.rollout.rollout_service import start_rollout
    from aura.trainer.train_adapter.verl.full_async.checkpoint_utils import checkpoint_version, resolve_checkpoint

    from aura.trainer.train_adapter.verl.full_async.replay_recovery import recovery_settings, select_pair
    joint_recovery = recovery_settings(train_config)
    path = None if joint_recovery else resolve_checkpoint(train_config.trainer)
    from aura.memory.replay.launcher import prepare_replay
    replay_actor = prepare_replay(train_config, rollout_config)
    version = checkpoint_version(path) if path else 0
    env = {"RESUME_ITERATION": str(version if path else -1), "AURA_RESUME_HF_PATH": ""}
    if joint_recovery and joint_recovery.get('from_branch'):
        from aura.memory.replay.config import ReplayConfig
        selected = select_pair(joint_recovery, ReplayConfig.from_mapping(train_config.replay), replay_actor)
        body = selected['body']
        from pathlib import Path
        model_shard = next(item for item in body['artifacts']
                           if Path(item['path']).name.startswith('model_world_size_'))
        hf_path = str(Path(joint_recovery['root']) / Path(model_shard['path']).parent / 'huggingface')
        if train_config.total_train_steps <= body['step']:
            raise ValueError('No remaining updates after selected paired checkpoint')
        # Pin the SAME verified commit for trainer and inference; never auto-select twice.
        from omegaconf import open_dict
        with open_dict(train_config):
            train_config.replay_recovery.commit_id = body['commit_id']
        env.update(RESUME_ITERATION=str(body['policy_version']), AURA_RESUME_HF_PATH=hf_path)
    if joint_recovery:
        env['AURA_PAIRED_RECOVERY'] = '1'
    if path:
        completed = version * train_config.async_training.trigger_parameter_sync_step
        if train_config.total_train_steps <= completed:
            raise ValueError("total_train_steps must exceed completed checkpoint steps")
        # Freeze the auto-resolved checkpoint for both components.
        train_config.trainer.resume_mode = "resume_path"
        train_config.trainer.resume_from_path = path
        rollout_config.train_iters = train_config.total_train_steps - completed
        hf_path = os.path.join(path, "actor", "huggingface")
        if not os.path.isdir(hf_path) or not any(
            name.endswith(".safetensors") and os.path.isfile(os.path.join(hf_path, name))
            for name in os.listdir(hf_path)
        ):
            raise ValueError("one-step resume requires checkpoint save_contents to include hf_model")
        env["AURA_RESUME_HF_PATH"] = hf_path

    # 异步提交 rollout 任务
    ref = start_rollout.options(
        runtime_env={"env_vars": env},
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
    try:
        run_ppo(train_config, task_runner_class=FullyAsyncTaskRunner)
    finally:
        if replay_actor is not None:
            try:
                ray.get(replay_actor.close_replay_store.remote(), timeout=train_config.replay.rpc_timeout)
            finally:
                ray.kill(replay_actor, no_restart=True)
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
