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
from aura.trainer.rollout.rollout_worker import RolloutWorker
from aura.trainer.rollout.rollout_executor import OneStepOffRolloutExecutor
from aura.trainer.rollout.fully_async_rollout_executor import FullyAsyncRolloutExecutor
from aura.controllers.rollout_controller.rollout_controller import RolloutController

from aura.base.utils.pad_process import (
    remove_padding_tensor_dict_to_dict,
    remove_padding_and_split_to_list,
    padding_dict_to_tensor_dict,
    put_prompts_experience,
)

logger = Loggers(__name__).get_logger()


def _create_rollout_worker(rollout_config, agent_service, infer_service, generate_config=None,
                           split_to_list=None):
    """Build a RolloutWorker with hard node affinity.

    Args:
        rollout_config: Rollout config object providing worker parameters.
        agent_service: Name of the agent service used by the worker.
        infer_service: Name of the inference service used by the worker.
        generate_config: Optional generate config; None for verl backends.
        split_to_list: Optional callable to remove padding and split to list;
            None for verl backends.
    """
    return RolloutWorker.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().node_id,
            soft=False,
        )
    ).remote(
        generate_config=generate_config,
        train_backend=rollout_config.train_backend,
        trajectory_timeout=rollout_config.trajectory_timeout,
        weight_save_dir=rollout_config.weight_save_dir,
        hybrid_batch_num=rollout_config.hybrid_batch_num,
        use_on_policy=rollout_config.use_on_policy,
        wait_available_weight_timeout=rollout_config.wait_available_weight_timeout,
        n_parallel_agents=rollout_config.n_samples_per_prompt,
        actor_rollout_dispatch_size=rollout_config.actor_rollout_dispatch_size,
        validate_n_samples=rollout_config.validate_n_samples,
        traj_output_path=rollout_config.traj_output_path,
        tokenizer_name_or_path=rollout_config.tokenizer_name_or_path,
        dataset_additional_keys=rollout_config.dataset_additional_keys,
        global_batch_size=rollout_config.global_batch_size,
        remove_padding_tensor_dict_to_dict=remove_padding_tensor_dict_to_dict,
        remove_padding_and_split_to_list=split_to_list,
        service_mode="infer",
        agent_service=agent_service,
        infer_service=infer_service,
        llm_tokenizer_path=getattr(rollout_config, 'llm_tokenizer_path', None),
    )


def _create_rollout_controller(rollout_config, model_name):
    """Build a RolloutController from the rollout config.

    Args:
        rollout_config: Rollout config object providing controller parameters.
        model_name: Name of the inference model.
    """
    return RolloutController(
        weight_save_dir=rollout_config.weight_save_dir,
        tokenizer_name_or_path=rollout_config.tokenizer_name_or_path,
        trust_remote_code=rollout_config.trust_remote_code,
        infer_tensor_parallel_size=rollout_config.infer_tensor_parallel_size,
        train_tensor_parallel_size=rollout_config.train_tensor_parallel_size,
        infer_expert_parallel_size=rollout_config.infer_expert_parallel_size,
        enable_version_control=rollout_config.enable_version_control,
        use_on_policy=rollout_config.use_on_policy,
        model_name=model_name,
    )


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


@ray.remote
def start_rollout(rollout_config, agent_service, infer_service):
    """Start the one-step-off rollout worker for verl backends.

    Args:
        rollout_config: Rollout config object providing worker/controller/executor params.
        agent_service: Name of the agent service.
        infer_service: Name of the inference service.
    """
    logger.info(f"{rollout_config=}, {agent_service=}, {infer_service=}")
    rollout_worker = _create_rollout_worker(rollout_config, agent_service, infer_service)
    ray.get(rollout_worker.wait_init_finished.remote(is_proxy_mode=True))

    controller = _create_rollout_controller(rollout_config, infer_service)
    controller.send_ready_to_train()

    executor = OneStepOffRolloutExecutor(
        controller,
        rollout_worker,
        train_iters=rollout_config.train_iters,
        padding_dict_to_tensor_dict=padding_dict_to_tensor_dict,
        put_prompts_experience=put_prompts_experience,
        dataset_additional_keys=rollout_config.dataset_additional_keys,
        data_optimized=rollout_config.data_optimized,
        n_samples_per_prompt=rollout_config.n_samples_per_prompt,
        hybrid_batch_num=rollout_config.hybrid_batch_num,
    )
    executor.fit()
    logger.info("one step off rollout process successfully!")


@ray.remote
def start_fully_async_rollout(rollout_config, agent_service, infer_service):
    """Start the fully-async rollout worker with stream-based sample queue.

    Args:
        rollout_config: Rollout config object providing worker/controller/executor params.
        agent_service: Name of the agent service.
        infer_service: Name of the inference service.
    """
    logger.info(f"fully_async {rollout_config=}, {agent_service=}, {infer_service=}")

    rollout_worker = _create_rollout_worker(rollout_config, agent_service, infer_service)
    ray.get(rollout_worker.wait_init_finished.remote(is_proxy_mode=True))

    if isinstance(rollout_config, dict):
        max_required_samples = rollout_config.get("max_required_samples", 0)
    else:
        max_required_samples = getattr(rollout_config, "max_required_samples", 0)
    if max_required_samples:
        from aura.controllers.rollout_controller.sample_queue import get_sample_queue
        try:
            sample_queue = get_sample_queue()
            ray.get(rollout_worker.set_fully_async_config.remote(sample_queue, max_required_samples))
            logger.info(f"fully_async injected SampleQueue, max_required_samples={max_required_samples}")
        except Exception as e:
            logger.error(f"fully_async failed to inject SampleQueue: {e}", exc_info=True)
            raise

    controller = _create_rollout_controller(rollout_config, infer_service)
    controller.send_ready_to_train()

    executor = FullyAsyncRolloutExecutor(
        controller,
        rollout_worker,
        train_iters=rollout_config.train_iters,
        padding_dict_to_tensor_dict=padding_dict_to_tensor_dict,
        put_prompts_experience=put_prompts_experience,
        dataset_additional_keys=rollout_config.dataset_additional_keys,
        data_optimized=rollout_config.data_optimized,
        n_samples_per_prompt=rollout_config.n_samples_per_prompt,
        hybrid_batch_num=rollout_config.hybrid_batch_num,
    )
    executor.fit()
    logger.info("fully_async rollout process successfully!")
