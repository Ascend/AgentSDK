#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
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
# -------------------------------------------------------------------------


import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from aura.base.log.loggers import Loggers
from aura.trainer.rollout.rollout_executor import OneStepOffRolloutExecutor
from aura.trainer.rollout.rollout_manager import RolloutManager
from aura.trainer.rollout.rollout_worker import RolloutWorker
from aura.controllers.rollout_controller.rollout_controller import RolloutController

from aura.base.utils.pad_process import (
    remove_padding_tensor_dict_to_dict,
    remove_padding_and_split_to_list,
    padding_dict_to_tensor_dict,
    put_prompts_experience,
)

logger = Loggers(__name__).get_logger()


@ray.remote
def start_async_rollout_worker(
    config, rl_config, agentic_env_config, actor_config, generate_config, agent_service, infer_service
):
    rollout_worker = RolloutWorker.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().node_id,
            soft=False,  # Force hard affinity
        )
    ).remote(
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
        dataset_additional_keys=actor_config.dataset_additional_keys,
        global_batch_size=actor_config.global_batch_size,
        remove_padding_tensor_dict_to_dict=remove_padding_tensor_dict_to_dict,
        remove_padding_and_split_to_list=remove_padding_and_split_to_list,
        service_mode="infer",
        agent_service=agent_service,
        infer_service=infer_service,
    )
    ray.get(rollout_worker.wait_init_finished.remote(is_proxy_mode=True))

    controller = RolloutController(
        weight_save_dir=generate_config.weight_save_dir,
        tokenizer_name_or_path=actor_config.tokenizer_name_or_path,
        trust_remote_code=generate_config.trust_remote_code,
        infer_tensor_parallel_size=generate_config.infer_tensor_parallel_size,
        train_tensor_parallel_size=actor_config.tensor_model_parallel_size,
        infer_expert_parallel_size=generate_config.infer_expert_parallel_size,
        enable_version_control=generate_config.enable_version_control,
        use_on_policy=generate_config.use_on_policy,
        model_name=infer_service,
    )
    controller.send_ready_to_train()

    executor = OneStepOffRolloutExecutor(
        controller,
        rollout_worker,
        train_iters=actor_config.train_iters,
        padding_dict_to_tensor_dict=padding_dict_to_tensor_dict,
        put_prompts_experience=put_prompts_experience,
        dataset_additional_keys=actor_config.dataset_additional_keys,
        **rl_config.dict(),
        **generate_config.dict(),
    )
    executor.fit()
    logger.info("one step off rollout process successfully!")


def _publish_worker_handle(executor_handle, manager):
    """Publish the rollout worker handle so the driver can drain it on shutdown."""
    worker = manager.worker_handle
    if executor_handle is None or worker is None:
        return
    try:
        ray.get(executor_handle.register_rollout.remote({"worker": worker}))
    except Exception as exp:
        logger.warning(f"publish rollout worker handle failed: {type(exp).__name__}")


@ray.remote
def start_rollout(rollout_config, agent_service, infer_service, executor_handle=None):
    """Start the one-step-off rollout worker for verl backends.

    Args:
        rollout_config: Rollout config object providing worker/controller/executor params.
        agent_service: Name of the agent service.
        infer_service: Name of the inference service.
        executor_handle: TrainExecutor actor handle used to publish the worker handle.
    """
    logger.info(f"{rollout_config=}, {agent_service=}, {infer_service=}")
    manager = None
    try:
        manager = RolloutManager.create(
            "one_step_off",
            rollout_config=rollout_config,
            agent_service=agent_service,
            infer_service=infer_service,
        )
        _publish_worker_handle(executor_handle, manager)
        manager.run()
    finally:
        if manager is not None:
            manager.shutdown()
    logger.info("one step off rollout process successfully!")


@ray.remote
def start_fully_async_rollout(rollout_config, agent_service, infer_service, executor_handle=None):
    """Start the fully-async rollout worker with stream-based sample queue.

    Args:
        rollout_config: Rollout config object providing worker/controller/executor params.
        agent_service: Name of the agent service.
        infer_service: Name of the inference service.
        executor_handle: TrainExecutor actor handle used to publish the worker handle.
    """
    logger.info(f"fully_async {rollout_config=}, {agent_service=}, {infer_service=}")
    manager = None
    try:
        manager = RolloutManager.create(
            "fully_async",
            rollout_config=rollout_config,
            agent_service=agent_service,
            infer_service=infer_service,
        )
        _publish_worker_handle(executor_handle, manager)
        manager.run()
    finally:
        if manager is not None:
            manager.shutdown()
    logger.info("fully_async rollout process successfully!")
