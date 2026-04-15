#!/usr/bin/env python3
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
import ray

from mindspeed_rl.utils import get_tokenizer
from mindspeed_rl.utils.pad_process import (
    remove_padding_and_split_to_list,
    remove_padding_tensor_dict_to_dict,
)
from mindspeed_rl.workers.reward_woker import RewardWorker

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.controllers.rollout_controller.rollout_client import send_ready_to_train_remote
from agentic_rl.controllers.rollout_controller.rollout_controller import RolloutController
from agentic_rl.trainer.rollout.rollout_worker import RolloutWorker
from agentic_rl.trainer.train_adapter.mindspeed_rl.one_step_off_policy.rollout.rollout_executor import (
    OneStepOffRolloutExecutor
)
from agentic_rl.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import (
    get_megatron_module,
    gpt_model_provider,
    initialize_megatron,
    parse_training_config,
    rm_model_provider
)
from agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import ActorHybridWorker
from agentic_rl.trainer.train_adapter.mindspeed_rl.workers.integrated_worker import IntegratedWorker

logger = Loggers(__name__).get_logger()

@ray.remote
def rollout(config, agent_service):
    actor_config, ref_config, reward_config, rl_config, \
        generate_config, profiler_config, msprobe_config, \
        agentic_env_config = parse_training_config(config).values()

    if (hasattr(config['megatron_training'], "ai_framework") and
            config['megatron_training']['ai_framework'] == "mindspore"):
        from mindspeed_rl.workers.scheduler.launcher_ms import RayActorGroupMs as RayActorGroup
    else:
        from mindspeed_rl.workers.scheduler.launcher import RayActorGroup

    tokenizer = get_tokenizer(tokenizer_model=actor_config.tokenizer_name_or_path,
                              prompt_type=actor_config.prompt_type, prompt_type_path=actor_config.prompt_type_path)
    logger.info('start async initializing ray actor groups')

    if rl_config.use_integrated_worker:
        integrated_worker = RayActorGroup(
            worker=IntegratedWorker,
            placement_group=None,
            megatron_config=actor_config,
            rl_config=rl_config,
            generate_config=generate_config,
            model_provider=gpt_model_provider,
            profiler_config=profiler_config["integrated"],
            msprobe_config=msprobe_config,
            tokenizer=tokenizer,
            initialize_func=initialize_megatron,
            get_megatron_module=get_megatron_module,
            global_batch_size=actor_config.global_batch_size * rl_config.n_samples_per_prompt
        ).initialize()
        actor_worker = integrated_worker
    else:
        actor_worker = RayActorGroup(
            worker=ActorHybridWorker,
            placement_group=None,
            megatron_config=actor_config,
            rl_config=rl_config,
            generate_config=generate_config,
            model_provider=gpt_model_provider,
            profiler_config=profiler_config["integrated"],
            msprobe_config=msprobe_config,
            tokenizer=tokenizer,
            initialize_func=initialize_megatron,
            get_megatron_module=get_megatron_module,
            global_batch_size=actor_config.global_batch_size * rl_config.n_samples_per_prompt
        ).initialize()

    # Start inference process
    rollout_worker = RolloutWorker.remote(
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
        worker_group=actor_worker,
        remove_padding_tensor_dict_to_dict=remove_padding_tensor_dict_to_dict,
        remove_padding_and_split_to_list=remove_padding_and_split_to_list,
        service_mode="infer",
        agent_service=agent_service,
    )
    ray.get(rollout_worker.wait_init_finished.remote(is_proxy_mode=False))

    temp_actor_ref_objs = []
    for actor in actor_worker.actor_handlers:
        temp_actor_ref_objs.append(actor.init_sharding_manager.remote())
    ray.get(temp_actor_ref_objs)

    # All inference processes are started, launch controller
    model_name = list(config.get('model').keys())[0]
    controller = RolloutController(actor_config, generate_config, model_name)

    # Notify training side that inference is ready to receive inference data
    controller.send_ready_to_train()

    # Start inference execution loop
    # ref and actor share the same card, ref only starts but doesn't work
    reference_worker = actor_worker
    reward_list = []
    if rl_config.reward_resource:
        reward_worker = RayActorGroup(
            worker=RewardWorker,
            placement_group=None,
            megatron_config=reward_config,
            rl_config=rl_config,
            generate_config=generate_config,
            model_provider=rm_model_provider,
            tokenizer=tokenizer,
            initialize_func=initialize_megatron,
            get_megatron_module=get_megatron_module,
            global_batch_size=actor_config.global_batch_size * rl_config.n_samples_per_prompt
        ).initialize()
        reward_list.append(reward_worker)

    executor = OneStepOffRolloutExecutor(
        controller,
        rollout_worker,
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
        **generate_config.dict()
    )
    executor.fit()
    ray.shutdown()