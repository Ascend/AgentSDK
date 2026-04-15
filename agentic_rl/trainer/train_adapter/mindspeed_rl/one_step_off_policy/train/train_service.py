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
from typing import Any, Callable

import ray

from mindspeed_rl.utils.pad_process import (
    remove_padding_tensor_dict_to_dict,
    remove_padding_and_split_to_list,
    padding_dict_to_tensor_dict
)
from mindspeed_rl.trainer.utils.transfer_dock import put_prompts_experience

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.controllers.train_controller.train_controller import TrainController
from agentic_rl.controllers.utils.utils import DEFAULT_SLEEP_TIME
from agentic_rl.trainer.train_adapter.mindspeed_rl.config_cls import ExtendedGenerateConfig
from agentic_rl.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train.train_dataloader import \
    optimize_train_dataloader
from agentic_rl.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train.train_executor import \
    OneStepOffTrainExecutor
from agentic_rl.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader import default_train_dataloader
from agentic_rl.trainer.train_adapter.mindspeed_rl.utils.prepare_train import prepare_train

logger = Loggers(__name__).get_logger()


def dummy_rollout(
    rl_config: Any,
    agentic_env_config: Any,
    actor_config: Any,
    generate_config: Any,
    actor_worker: Any,
    agent_service: Any,
    infer_service: Any
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
    from agentic_rl.trainer.rollout.rollout_worker import RolloutWorker

    rollout_worker = RolloutWorker.remote(
        n_parallel_agents=rl_config.n_samples_per_prompt,
        max_prompt_length=rl_config.max_prompt_length,
        actor_rollout_dispatch_size=rl_config.actor_rollout_dispatch_size,
        simplify_think_content=rl_config.simplify_think_content,
        validate_n_samples=rl_config.validate_n_samples,
        traj_output_path=agentic_env_config.rollout_output_path,
        tokenizer_name_or_path=actor_config.tokenizer_name_or_path,
        global_batch_size=actor_config.global_batch_size,
        generate_config=generate_config,
        agentic_env_config=agentic_env_config,
        worker_group=actor_worker,
        remove_padding_tensor_dict_to_dict=remove_padding_tensor_dict_to_dict,
        remove_padding_and_split_to_list=remove_padding_and_split_to_list,
        agent_service=agent_service,
        infer_service=infer_service
    )
    return rollout_worker


def get_train_controller(
    actor_worker: Any,
    actor_config: Any,
    rl_config: Any,
    generate_config: Any,
    consumed_train_samples: int,
    data_optimized: bool
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
        from agentic_rl.controllers.train_controller.train_mock_controller import TrainMockController
        return TrainMockController(
            actor_worker=actor_worker,
            actor_config=actor_config,
            rl_config=rl_config,
            generate_config=generate_config,
            initialize_rollout_dataloader=default_train_dataloader,
            consumed_train_samples=consumed_train_samples,
            data_optimized=data_optimized
        )

    dataloader_func = optimize_train_dataloader if data_optimized else default_train_dataloader

    controller = TrainController(
        actor_worker=actor_worker,
        actor_config=actor_config,
        rl_config=rl_config,
        generate_config=generate_config,
        initialize_rollout_dataloader=dataloader_func,
        consumed_train_samples=consumed_train_samples,
        data_optimized=data_optimized
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
    remove_padding_fn: Callable,
    remove_padding_split_fn: Callable,
    padding_fn: Callable,
    put_experience_fn: Callable
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
        remove_padding_fn: Function to remove padding from tensor dicts.
        remove_padding_split_fn: Function to remove padding and split to list.
        padding_fn: Function to pad dicts to tensor dicts.
        put_experience_fn: Function to put prompts experience into transfer dock.
    """
    from agentic_rl.trainer.rollout.rollout_service import start_async_rollout_worker
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    start_async_rollout_worker.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().node_id,
            soft=False
        )
    ).remote(
        config=config,
        rl_config=rl_config,
        agentic_env_config=agentic_env_config,
        actor_config=actor_config,
        generate_config=generate_config,
        agent_service=agent_service,
        infer_service=infer_service,
        remove_padding_tensor_dict_to_dict=remove_padding_fn,
        remove_padding_and_split_to_list=remove_padding_split_fn,
        padding_dict_to_tensor_dict=padding_fn,
        put_prompts_experience=put_experience_fn
    )


@ray.remote
def train(config: Any, agent_service: Any, infer_service: Any) -> None:
    """
    Main one-step off-policy training entry point.

    Prepares training components, initializes rollout workers, configures the
    train controller and executor, then starts the training loop.

    Args:
        config: Full training configuration.
        agent_service: Agent service handle.
        infer_service: Inference service handle.
    """
    (actor_config, rl_config, _, generate_config, agentic_env_config, actor_worker,
     reference_worker, reward_list, tokenizer, _, _, _) = prepare_train(config, "one_step_off")

    create_rollout_worker(
        config=config,
        rl_config=rl_config,
        agentic_env_config=agentic_env_config,
        actor_config=actor_config,
        generate_config=generate_config,
        agent_service=agent_service,
        infer_service=infer_service,
        remove_padding_fn=remove_padding_tensor_dict_to_dict,
        remove_padding_split_fn=remove_padding_and_split_to_list,
        padding_fn=padding_dict_to_tensor_dict,
        put_experience_fn=put_prompts_experience
    )

    dummy_rollout_worker = dummy_rollout(
        rl_config, agentic_env_config, actor_config,
        generate_config, actor_worker, agent_service, infer_service
    )
    ray.get(dummy_rollout_worker.wait_init_finished.remote(is_proxy_mode=False))

    temp_actor_ref_objs = [
        actor.init_sharding_manager.remote()
        for actor in actor_worker.actor_handlers
    ]
    ray.get(temp_actor_ref_objs)

    extended_generate_config = ExtendedGenerateConfig(config.get("generate_config"))
    consumed_train_samples = actor_worker.get_consumed_train_samples()

    controller = get_train_controller(
        actor_worker=actor_worker,
        actor_config=actor_config,
        rl_config=rl_config,
        generate_config=extended_generate_config,
        consumed_train_samples=consumed_train_samples,
        data_optimized=extended_generate_config.data_optimized
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
        **extended_generate_config.dict()
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
    from agentic_rl.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import parse_training_config

    actor_config, _, _, rl_config, \
        generate_config, _, _, \
        agentic_env_config = parse_training_config(config).values()

    create_rollout_worker(
        config=config,
        rl_config=rl_config,
        agentic_env_config=agentic_env_config,
        actor_config=actor_config,
        generate_config=generate_config,
        agent_service=agent_service,
        infer_service=infer_service,
        remove_padding_fn=remove_padding_tensor_dict_to_dict,
        remove_padding_split_fn=remove_padding_and_split_to_list,
        padding_fn=padding_dict_to_tensor_dict,
        put_experience_fn=put_prompts_experience
    )

    controller = get_train_controller(
        actor_worker=None,
        actor_config=actor_config,
        rl_config=rl_config,
        generate_config=generate_config,
        consumed_train_samples=0,
        data_optimized=generate_config.data_optimized
    )
    controller.initialize_dispatch()
    controller.initialize_train_server()
    controller.wait_for_rollout_unit_ready()
    controller.initialize_rollout()

    try:
        while True:
            time.sleep(DEFAULT_SLEEP_TIME)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Dummy train received shutdown signal, exiting gracefully")
