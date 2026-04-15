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
def start_async_rollout_worker(
    config,
    rl_config,
    agentic_env_config,
    actor_config,
    generate_config,
    agent_service,
    infer_service,
    remove_padding_tensor_dict_to_dict,
    remove_padding_and_split_to_list,
    padding_dict_to_tensor_dict,
    put_prompts_experience
):
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    rollout_worker = RolloutWorker.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().node_id,
            soft=False  # Force hard affinity
        )
    ).remote(
        n_parallel_agents=rl_config.n_samples_per_prompt,
        max_prompt_length=rl_config.max_prompt_length,
        actor_rollout_dispatch_size=rl_config.actor_rollout_dispatch_size,
        simplify_think_content=rl_config.simplify_think_content,
        validate_n_samples=rl_config.validate_n_samples,
        traj_output_path=agentic_env_config.rollout_output_path,
        tokenizer_name_or_path=actor_config.tokenizer_name_or_path,
        dataset_additional_keys=actor_config.dataset_additional_keys,
        global_batch_size=actor_config.global_batch_size,
        generate_config=generate_config,
        agentic_env_config=agentic_env_config,
        remove_padding_tensor_dict_to_dict=remove_padding_tensor_dict_to_dict,
        remove_padding_and_split_to_list=remove_padding_and_split_to_list,
        service_mode="infer",
        agent_service=agent_service,
        infer_service=infer_service
    )
    ray.get(rollout_worker.wait_init_finished.remote(is_proxy_mode=True))

    # All inference processes have started, starting controller
    model_name = list(config.get('model').keys())[0]
    from agentic_rl.controllers.rollout_controller.rollout_controller import RolloutController
    controller = RolloutController(actor_config, generate_config, model_name)
    # Notify training side that inference is ready to receive inference data
    controller.send_ready_to_train()

    from agentic_rl.trainer.rollout.rollouter import OneStepOffRollouter
    executor = OneStepOffRollouter(
        controller,
        rollout_worker,
        train_iters=actor_config.train_iters,
        padding_dict_to_tensor_dict=padding_dict_to_tensor_dict,
        put_prompts_experience=put_prompts_experience,
        dataset_additional_keys=actor_config.dataset_additional_keys,
        **rl_config.dict(),
        **generate_config.dict()
    )
    executor.fit()
    logger.info("one step off rollout process successfully!")
