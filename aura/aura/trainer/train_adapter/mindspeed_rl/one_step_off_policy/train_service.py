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

from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from aura.base.log.loggers import Loggers
from aura.controllers.train_controller.train_controller import TrainController
from aura.controllers.utils.utils import DEFAULT_SLEEP_TIME
from aura.trainer.train_adapter.mindspeed_rl.config_cls import ExtendedGenerateConfig
from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataloader import optimize_train_dataloader
from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_executor import OneStepOffTrainExecutor
from aura.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader import default_train_dataloader
from aura.trainer.train_adapter.mindspeed_rl.utils.prepare_train import prepare_train


logger = Loggers(__name__).get_logger()


def dummy_rollout(
    rl_config: Any,
    agentic_env_config: Any,
    actor_config: Any,
    generate_config: Any,
    actor_worker: Any,
    agent_service: Any,
    infer_service: Any,
) -> Any:
    """
    Create a dummy rollout worker that only starts the inference process.

    Bypasses the full rollout flow to work around init_sharding_manager failures.

    Args:
        rl_config: Reinforcement learning configuration.
        agentic_env_config: Agentic environment configuration.
        actor_config: Actor model configuration.
        generate_config: Generation configuration.
        actor_worker: Actor worker group.
        agent_service: Agent service handle.
        infer_service: Inference service handle.

    Returns:
        A remote RolloutWorker reference.
    """
    from aura.trainer.rollout.rollout_worker import RolloutWorker

    rollout_worker = RolloutWorker.remote(
        generate_config=generate_config,
        train_backend=generate_config.train_backend,
        trajectory_timeout=agentic_env_config.trajectory_timeout,
        weight_save_dir=generate_config.weight_save_dir,
        hybrid_batch_num=generate_config.hybrid_batch_num,
        use_on_policy=generate_config.use_on_policy,
        wait_available_weight_timeout=generate_config.wait_available_weight_timeout,
        n_parallel_agents=rl_config.n_samples_per_prompt,
        actor_rollout_dispatch_size=rl_config.actor_rollout_dispatch_size,
        validate_n_samples=rl_config.validate_n_samples,
        traj_output_path=agentic_env_config.rollout_output_path,
        tokenizer_name_or_path=actor_config.tokenizer_name_or_path,
        global_batch_size=actor_config.global_batch_size,
        worker_group=actor_worker,
        remove_padding_tensor_dict_to_dict=None,
        remove_padding_and_split_to_list=None,
        agent_service=agent_service,
        infer_service=infer_service,
    )
    return rollout_worker


def get_train_controller(
    actor_worker: Any,
    actor_config: Any,
    rl_config: Any,
    generate_config: Any,
    consumed_train_samples: int,
    data_optimized: bool,
) -> TrainController:
    """
    Create the appropriate train controller based on configuration.

    Args:
        actor_worker: Actor worker group.
        actor_config: Actor model configuration.
        rl_config: Reinforcement learning configuration.
        generate_config: Generation configuration.
        consumed_train_samples: Number of training samples already consumed.
        data_optimized: Whether data optimization is enabled.

    Returns:
        A TrainController or TrainMockController instance.
    """
    if rl_config.mock_rollout:
        from aura.controllers.train_controller.train_mock_controller import TrainMockController

        return TrainMockController(
            actor_worker=actor_worker,
            actor_config=actor_config,
            rl_config=rl_config,
            generate_config=generate_config,
            initialize_rollout_dataloader=default_train_dataloader,
            consumed_train_samples=consumed_train_samples,
            data_optimized=data_optimized,
        )
    if data_optimized:
        controller = TrainController(
            actor_worker=actor_worker,
            global_batch_size=actor_config.global_batch_size,
            n_samples_per_prompt=rl_config.n_samples_per_prompt,
            validate_num_samples=rl_config.validate_num_samples,
            init_num_group_batches=generate_config.init_num_group_batches,
            max_queue_size=generate_config.max_queue_size,
            train_iters=actor_config.train_iters,
            weight_save_dir=generate_config.weight_save_dir,
            delta=generate_config.ckpt_delta,
            data_loader=actor_config,
            initialize_rollout_dataloader=optimize_train_dataloader,
            consumed_train_samples=consumed_train_samples,
            data_optimized=data_optimized,
        )
    else:
        controller = TrainController(
            actor_worker=actor_worker,
            global_batch_size=actor_config.global_batch_size,
            n_samples_per_prompt=rl_config.n_samples_per_prompt,
            validate_num_samples=rl_config.validate_num_samples,
            init_num_group_batches=generate_config.init_num_group_batches,
            max_queue_size=generate_config.max_queue_size,
            train_iters=actor_config.train_iters,
            weight_save_dir=generate_config.weight_save_dir,
            delta=generate_config.ckpt_delta,
            data_loader=actor_config,
            initialize_rollout_dataloader=default_train_dataloader,
            consumed_train_samples=consumed_train_samples,
            data_optimized=data_optimized,
        )
    return controller


def create_rollout_worker(
    config: Any,
    rl_config: Any,
    agentic_env_config: Any,
    actor_config: Any,
    generate_config: Any,
    agent_service: Any,
    infer_service: Any,
) -> None:
    """
    Launch an asynchronous rollout worker on the current Ray node.

    Currently does not support starting rollout and train in the same cluster.

    Args:
        config: Full training configuration.
        rl_config: Reinforcement learning configuration.
        agentic_env_config: Agentic environment configuration.
        actor_config: Actor model configuration.
        generate_config: Generation configuration.
        agent_service: Agent service handle.
        infer_service: Inference service handle.
    """
    from aura.trainer.rollout.rollout_service import start_async_rollout_worker

    start_async_rollout_worker.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().node_id, soft=False)
    ).remote(
        config=config,
        rl_config=rl_config,
        agentic_env_config=agentic_env_config,
        actor_config=actor_config,
        generate_config=generate_config,
        agent_service=agent_service,
        infer_service=infer_service,
    )


@ray.remote
def train(work_mode, train_config, rollout_config, agent_service, infer_service) -> None:
    """
    Main one-step off-policy training entry point.

    Prepares training components, initializes rollout workers, configures the
    train controller and executor, then starts the training loop.

    Args:
        config: Full training configuration.
        agent_service: Agent service handle.
        infer_service: Inference service handle.
    """
    _ = rollout_config
    (
        actor_config,
        rl_config,
        generate_config,
        agentic_env_config,
        actor_worker,
        reference_worker,
        reward_list,
        tokenizer,
        data_iters,
        val_dataloader,
        test_dataloader,
    ) = prepare_train(train_config, work_mode)

    create_rollout_worker(
        config=train_config,
        rl_config=rl_config,
        agentic_env_config=agentic_env_config,
        actor_config=actor_config,
        generate_config=generate_config,
        agent_service=agent_service,
        infer_service=infer_service,
    )

    dummy_rollout_worker = dummy_rollout(
        rl_config, agentic_env_config, actor_config, generate_config, actor_worker, agent_service, infer_service
    )
    ray.get(dummy_rollout_worker.wait_init_finished.remote(is_proxy_mode=False))

    temp_actor_ref_objs = [actor.init_sharding_manager.remote() for actor in actor_worker.actor_handlers]
    ray.get(temp_actor_ref_objs)

    extended_generate_config = ExtendedGenerateConfig(config.get("generate_config"))
    consumed_train_samples = actor_worker.get_consumed_train_samples()

    controller = get_train_controller(
        actor_worker=actor_worker,
        actor_config=actor_config,
        rl_config=rl_config,
        generate_config=extended_generate_config,
        consumed_train_samples=consumed_train_samples,
        data_optimized=extended_generate_config.data_optimized,
    )
    controller.pre_initialize()
    controller.wait_for_rollout_unit_ready()
    controller.initialize_rollout()

    trainer = OneStepOffTrainExecutor(
        controller,
        actor_worker,
        reference_worker,
        reward_list,
        tokenizer=tokenizer,
        global_batch_size=actor_config.global_batch_size,
        micro_batch_size=rl_config.adv_dispatch_size,
        train_iters=actor_config.train_iters,
        save_interval=actor_config.save_interval,
        dataset_additional_keys=actor_config.dataset_additional_keys,
        **rl_config.dict(),
        **extended_generate_config.dict(),
    )

    logger.info(">>> Ready to start the one step off training fit")
    trainer.fit()
    ray.shutdown()


@ray.remote
def dummy_train(config: Any, agent_service: Any, infer_service: Any) -> None:
    """
    Lightweight training entry point that skips model initialization.

    Sets up rollout and train controller dispatch without loading model weights,
    then enters a sleep loop waiting for external rollout data.

    Args:
        config: Full training configuration.
        agent_service: Agent service handle.
        infer_service: Inference service handle.
    """
    from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import parse_training_config

    (
        actor_config,
        ref_config,
        reward_config,
        rl_config,
        generate_config,
        profiler_config,
        msprobe_config,
        agentic_env_config,
    ) = parse_training_config(config).values()

    create_rollout_worker(
        config=config,
        rl_config=rl_config,
        agentic_env_config=agentic_env_config,
        actor_config=actor_config,
        generate_config=generate_config,
        agent_service=agent_service,
        infer_service=infer_service,
    )

    controller = get_train_controller(
        actor_worker=None,
        actor_config=actor_config,
        rl_config=rl_config,
        generate_config=generate_config,
        consumed_train_samples=0,
        data_optimized=generate_config.data_optimized,
    )
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
