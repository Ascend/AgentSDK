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
import copy

import ray
from ray.util import placement_group

from mindspeed_rl.utils import get_tokenizer
from mindspeed_rl.utils.utils import MsProbe, get_node_nums
from mindspeed_rl.workers.rule_reward import RuleReward
from mindspeed_rl.workers.reward_woker import RewardWorker

from agentic_rl.base.utils.work_mode import set_work_mode
from agentic_rl.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader import default_train_dataloader
from agentic_rl.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import (
    get_megatron_module,
    initialize_megatron,
    rm_model_provider,
    gpt_model_provider,
    parse_training_config
)
from agentic_rl.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import ActorHybridWorker
from agentic_rl.trainer.train_adapter.mindspeed_rl.workers.integrated_worker import IntegratedWorker
from agentic_rl.trainer.train_adapter.mindspeed_rl.workers.reference_worker import ReferenceWorker
from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def prepare_train(config: dict, work_mode: str) -> tuple:
    """
    Prepare training components including workers, tokenizer, and data loaders.

    Args:
        config: Global training configuration dictionary.
        work_mode: Working mode string, e.g. "hybrid".

    Returns:
        A tuple containing actor_config, rl_config, generate_config, agentic_env_config,
        actor_worker, reference_worker, reward_list, tokenizer, data_iters,
        val_dataloader, and test_dataloader.
    """
    actor_config, ref_config, reward_config, rl_config, \
        generate_config, profiler_config, msprobe_config, \
        agentic_env_config = parse_training_config(config).values()

    if (hasattr(config['megatron_training'], "ai_framework") and
            config['megatron_training']['ai_framework'] == "mindspore"):
        from mindspeed_rl.workers.scheduler.launcher_ms import RayActorGroupMs as RayActorGroup
    else:
        from mindspeed_rl.workers.scheduler.launcher import RayActorGroup

    MsProbe.config_init(msprobe_config)
    MsProbe.save_configs({
        'actor': copy.deepcopy(actor_config.dict()),
        'ref': copy.deepcopy(ref_config.dict()),
        'reward': copy.deepcopy(reward_config.dict()),
        'rl': copy.deepcopy(rl_config.dict()),
        'generate': copy.deepcopy(generate_config.dict()),
    })

    tokenizer = get_tokenizer(tokenizer_model=actor_config.tokenizer_name_or_path,
                              prompt_type=actor_config.prompt_type, prompt_type_path=actor_config.prompt_type_path)
    logger.info('start async initializing ray actor groups')

    reward_list = []

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
        reference_worker = integrated_worker

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

        reference_worker = RayActorGroup(
            worker=ReferenceWorker,
            placement_group=None,
            megatron_config=ref_config,
            rl_config=rl_config,
            generate_config=generate_config,
            model_provider=gpt_model_provider,
            tokenizer=tokenizer,
            initialize_func=initialize_megatron,
            get_megatron_module=get_megatron_module,
            global_batch_size=actor_config.global_batch_size * rl_config.n_samples_per_prompt
        ).initialize()

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

    rule_reward_num_process = get_node_nums()
    if rl_config.rule_reward:
        pg = placement_group(
            [{"CPU": rl_config.num_cpus_for_local_task} for _ in range(rule_reward_num_process)],
            strategy='SPREAD'
        )

        ray.get(pg.ready())

        for i in range(rule_reward_num_process):
            rule_reward = RuleReward.options(placement_group=pg, placement_group_bundle_index=i).remote()
            rule_reward.initialize.remote(reward_config, rl_config, tokenizer)
            reward_list.append(rule_reward)

    actor_worker.wait_all_ref_objs_run_over()

    set_work_mode(work_mode)

    data_iters = None
    val_dataloader = None
    test_dataloader = None
    if work_mode == "hybrid":
        consumed_train_samples = actor_worker.get_consumed_train_samples()
        data_iters, val_dataloader, test_dataloader = default_train_dataloader(
            actor_config, rl_config.validate_num_samples, consumed_train_samples)

    reference_worker.wait_all_ref_objs_run_over()
    for reward in reward_list:
        if hasattr(reward, 'wait_all_ref_objs_run_over'):
            reward.wait_all_ref_objs_run_over()

    return (actor_config, rl_config, generate_config, agentic_env_config, actor_worker,
            reference_worker, reward_list, tokenizer, data_iters, val_dataloader, test_dataloader)