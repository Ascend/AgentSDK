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

from mindspeed_rl.utils.pad_process import (
    remove_padding_tensor_dict_to_dict,
    remove_padding_and_split_to_list
)

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.trainer.rollout.rollout_worker import RolloutWorker
from agentic_rl.trainer.train_adapter.mindspeed_rl.hybrid_policy.hybrid_trainer import AgentGRPOTrainer
from agentic_rl.trainer.train_adapter.mindspeed_rl.utils.prepare_train import prepare_train

logger = Loggers(__name__).get_logger()


@ray.remote
def train(config, agent_service=None, infer_service=None):
    (actor_config, rl_config, rl_config, generate_config, agentic_env_config,
     actor_worker, reference_worker, reward_list, tokenizer,
     data_iters, val_dataloader, test_dataloader) = prepare_train(config, "hybrid")

    rollout_worker = RolloutWorker.remote(
        n_parallel_agents=rl_config.n_samples_per_prompt,
        max_prompt_length=rl_config.max_prompt_length,
        actor_rollout_dispatch_size=rl_config.actor_rollout_dispatch_size,
        simplify_think_content=rl_config.simplify_think_content,
        validate_n_samples=rl_config.validate_n_samples,
        traj_output_path=agentic_env_config.rollout_output_path,
        tokenizer_name_or_path=actor_config.tokenizer_name_or_path,
        dataset_additional_keys=actor_config.dataset_additional_keys,
        generate_config=generate_config,
        agentic_env_config=agentic_env_config,
        worker_group=actor_worker,
        remove_padding_tensor_dict_to_dict=remove_padding_tensor_dict_to_dict,
        remove_padding_and_split_to_list=remove_padding_and_split_to_list,
        agent_service=agent_service,
        infer_service=infer_service
    )

    ray.get(rollout_worker.wait_init_finished.remote(is_proxy_mode=False))

    temp_actor_ref_objs = []
    for actor in actor_worker.actor_handlers:
        temp_actor_ref_objs.append(actor.init_sharding_manager.remote())
    ray.get(temp_actor_ref_objs)

    trainer = AgentGRPOTrainer(
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
        **rl_config.dict()
    )

    trainer.fit(data_iters, val_dataloader, test_dataloader)
    logger.info("training process successfully!")