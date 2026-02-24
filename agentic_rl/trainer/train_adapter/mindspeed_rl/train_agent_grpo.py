#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
from typing import Any, Dict

import ray
from cli.train_grpo import gpt_model_provider, initialize_megatron, get_megatron_module
from datasets import load_dataset
from mindspeed_rl import MegatronConfig, GenerateConfig, RLConfig
from mindspeed_rl.utils import get_tokenizer
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.utils.utils import get_node_nums
from mindspeed_rl.workers.rule_reward import RuleReward
from mindspeed_rl.workers.scheduler.launcher import RayActorGroup
from ray.exceptions import RayError
from ray.util import placement_group

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.data_loader import GRPODataLoader
from agentic_rl.configs.agentic_rl_config import AgenticRLConfig
from agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer import AgentGRPOTrainer
from agentic_rl.trainer.train_adapter.mindspeed_rl.configs.parse_msrl_config import MSRLConfigParser
from agentic_rl.trainer.train_adapter.mindspeed_rl.workers.integrated_worker import IntegratedWorker

logger = Loggers(__name__)

_MAX_CONSUMED_SAMPLES = 1000 * 1000


def _create_worker(agentic_rl_config: AgenticRLConfig,
                   actor_config: MegatronConfig,
                   rl_config: RLConfig,
                   generate_config: GenerateConfig,
                   tokenizer: BaseTokenizer):
    logger.info('start async initializing ray actor groups')

    if rl_config.use_integrated_worker:
        try:
            integrated_worker = RayActorGroup(
                worker=IntegratedWorker,
                placement_group=None,
                megatron_config=actor_config,
                rl_config=rl_config,
                generate_config=generate_config,
                model_provider=gpt_model_provider,
                tokenizer=tokenizer,
                initialize_func=initialize_megatron,
                get_megatron_module=get_megatron_module,
                agentic_rl_config=agentic_rl_config,
                global_batch_size=actor_config.global_batch_size * rl_config.n_samples_per_prompt
            ).initialize()
            actor_worker = integrated_worker
            reference_worker = integrated_worker
        except AttributeError as e:
            logger.error(f"create actor worker failed with missing attribute, error: {e}")
            raise AttributeError("create actor worker failed with missing attribute") from e
        except Exception as e:
            logger.error(f"Unexpected error occurred when create actor worker, error: {e}")
            raise RuntimeError("Unexpected error occurred when create actor worker") from e
    else:
        logger.error("grpo training only support integrated mode now!")
        raise ValueError("grpo training only support integrated mode now!")

    return actor_worker, reference_worker


def _create_reward_worker(reward_config: MegatronConfig, rl_config: RLConfig, tokenizer: BaseTokenizer):
    reward_list = []
    rule_reward_num_process = get_node_nums()
    if rl_config.rule_reward:
        pg = placement_group(
            [{"CPU": rl_config.num_cpus_for_local_task} for _ in range(rule_reward_num_process)],
            strategy='SPREAD'
        )
        ray.get(pg.ready())

        for i in range(rule_reward_num_process):
            try:
                rule_reward = RuleReward.options(placement_group=pg, placement_group_bundle_index=i).remote()
                rule_reward.initialize.remote(reward_config, rl_config, tokenizer)
                reward_list.append(rule_reward)
            except RayError as e:
                logger.error(f"create reward worker failed with ray, error: {e}")
                raise RayError("create reward worker failed with ray") from e
            except Exception as e:
                logger.error(f"Unexpected error occurred when create reward worker, error: {e}")
                raise RuntimeError("Unexpected error occurred when create reward worker") from e
    else:
        logger.error("grpo training only support rule reward now!")
        raise ValueError("grpo training only support rule reward now!")

    return reward_list


def _process_dataset(
        actor_config: MegatronConfig, test_data_path: str, training_samples: int, consumed_train_samples: int):
    try:
        train_ds = load_dataset("json", data_files=actor_config.data_path)['train']
        test_ds = None
        if test_data_path:
            test_ds = load_dataset("json", data_files=test_data_path)['train']
    except ValueError as e:
        logger.error(f"loading data failed with value error: {e}")
        raise ValueError(f"loading data failed with value error") from e
    except Exception as e:
        logger.error(f"Unexpected error occurred when loading data, error: {e}")
        raise RuntimeError("Unexpected error occurred when loading data") from e

    if consumed_train_samples > _MAX_CONSUMED_SAMPLES:
        logger.error(f"consumed samples exceed the limit： {_MAX_CONSUMED_SAMPLES}")
        raise ValueError(f"consumed samples exceed the limit： {_MAX_CONSUMED_SAMPLES}")

    data_loader = GRPODataLoader(
        dataset=train_ds,
        dataset_additional_keys=actor_config.dataset_additional_keys,
        global_batch_size=actor_config.global_batch_size,
        num_samples=training_samples,
        num_workers=actor_config.num_workers,
        seed=actor_config.seed,
        no_shuffle=actor_config.no_shuffle
    )

    test_iters = None
    if test_ds:
        test_loader = GRPODataLoader(
            dataset=test_ds,
            dataset_additional_keys=actor_config.dataset_additional_keys,
            global_batch_size=actor_config.global_batch_size,
            num_workers=actor_config.num_workers,
        )
        test_iters = iter(test_loader)

    if actor_config.global_batch_size is None or actor_config.global_batch_size <= 0:
        logger.error("actor_config.global_batch_size must not be set and greater than 0.")
        raise ValueError("actor_config.global_batch_size must not be set and greater than 0.")

    try:
        data_iters = iter(data_loader)
        [next(data_iters) for _ in range(consumed_train_samples // actor_config.global_batch_size)]
    except StopIteration as e:
        logger.error(f"try to iter data_loader for skipping consumed samples failed, error: {e}")
        raise RuntimeError("try to iter data_loader for skipping consumed samples failed") from e

    return data_iters, test_iters


@ray.remote
def train(config: Dict[str, Any]):
    configs = MSRLConfigParser(config).process_config()

    agentic_rl_config = configs.get("agentic_rl_config")
    actor_config = configs.get("actor_config")
    reward_config = configs.get("reward_config")
    rl_config = configs.get("rl_config")
    generate_config = configs.get("generate_config")

    try:
        tokenizer = get_tokenizer(tokenizer_model=actor_config.tokenizer_name_or_path)
    except ValueError as e:
        logger.error(f"create tokenizer failed, error: {e}")
        raise ValueError("create tokenizer failed") from e
    logger.debug("tokenizer created success")

    actor_worker, reference_worker = _create_worker(
        agentic_rl_config, actor_config, rl_config, generate_config, tokenizer)
    logger.debug("actor and reference worker created success")

    reward_list = _create_reward_worker(reward_config, rl_config, tokenizer)
    logger.debug("reward workers created success")

    data_iters, _ = _process_dataset(actor_config,
                                     agentic_rl_config.test_data_path,
                                     actor_config.train_iters * actor_config.global_batch_size,
                                     actor_worker.get_consumed_train_samples())
    logger.debug("dataset created and processed success")

    actor_worker.wait_all_ref_objs_run_over()
    reference_worker.wait_all_ref_objs_run_over()
    for reward in reward_list:
        if hasattr(reward, 'wait_all_ref_objs_run_over'):
            reward.wait_all_ref_objs_run_over()

    try:
        trainer = AgentGRPOTrainer(rl_config, actor_config, generate_config, agentic_rl_config,
                                   actor_worker, reference_worker, reward_list, tokenizer)
    except (AttributeError, ValueError) as e:
        logger.error(f"create trainer failed, error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred when create trainer failed, error: {e}")
        raise RuntimeError("Unexpected error occurred when create trainer failed") from e

    logger.info("training start")
    try:
        trainer.fit(data_iters)
    except (AttributeError, RuntimeError, ValueError) as e:
        logger.error(f"trainer fit failed, error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred when fit, error: {e}")
        raise RuntimeError("Unexpected error occurred when fit") from e
    logger.info("training successfully!")
