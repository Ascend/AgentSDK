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

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.trainer.rollout.rollout_worker import RolloutWorker

logger = Loggers(__name__).get_logger()


@ray.remote
def start_rollout(
    cluster_mode,
    rollout_config,
    agent_service,
    infer_service,
):
    print(f"{rollout_config=}, {agent_service=}, {infer_service=}, {cluster_mode=}")

    # Rollouter Worker
    from mindspeed_rl.trainer.utils.transfer_dock import put_prompts_experience
    from mindspeed_rl.utils.pad_process import (
        remove_padding_tensor_dict_to_dict,
        remove_padding_and_split_to_list,
        padding_dict_to_tensor_dict
    ) # TODO: Decouple and remove
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    rollout_worker = RolloutWorker.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().node_id,
            soft=False  # Force hard affinity
        )
    ).remote(
        train_backend=rollout_config.train_backend,
        trajectory_timeout=rollout_config.trajectory_timeout,
        weight_save_dir=rollout_config.weight_save_dir,
        hybrid_batch_num=rollout_config.hybrid_batch_num,
        use_on_policy=rollout_config.use_on_policy,
        n_parallel_agents=rollout_config.n_samples_per_prompt,
        max_prompt_length=rollout_config.max_prompt_length,
        actor_rollout_dispatch_size=rollout_config.actor_rollout_dispatch_size,
        simplify_think_content=rollout_config.simplify_think_content,
        validate_n_samples=rollout_config.validate_n_samples,
        traj_output_path=rollout_config.traj_output_path,
        tokenizer_name_or_path=rollout_config.tokenizer_name_or_path,
        dataset_additional_keys=rollout_config.dataset_additional_keys,
        global_batch_size=rollout_config.global_batch_size,
        remove_padding_tensor_dict_to_dict=remove_padding_tensor_dict_to_dict,
        remove_padding_and_split_to_list=remove_padding_and_split_to_list,
        service_mode="infer",
        agent_service=agent_service,
        infer_service=infer_service
    )
    ray.get(rollout_worker.wait_init_finished.remote(is_proxy_mode=True))

    # Rollouter Controller
    # All inference processes have started, starting controller
    model_name = infer_service
    from agentic_rl.controllers.rollout_controller.rollout_controller import RolloutController
    controller = RolloutController(
                weight_save_dir=rollout_config.weight_save_dir,
                tokenizer_name_or_path=rollout_config.tokenizer_name_or_path,
                trust_remote_code=rollout_config.trust_remote_code,
                infer_tensor_parallel_size=rollout_config.infer_tensor_parallel_size,
                train_tensor_parallel_size=rollout_config.train_tensor_parallel_size,
                infer_expert_parallel_size=rollout_config.infer_expert_parallel_size,
                enable_version_control=rollout_config.enable_version_control,
                use_on_policy=rollout_config.use_on_policy,
                model_name=model_name)
    # Notify training side that inference is ready to receive inference data
    controller.send_ready_to_train()

    from agentic_rl.trainer.rollout.rollouter import OneStepOffRollouter
    executor = OneStepOffRollouter(
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