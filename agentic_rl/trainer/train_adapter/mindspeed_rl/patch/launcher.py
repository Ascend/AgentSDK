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
import uuid

import ray
from mindspeed_rl.workers.scheduler.launcher import ActorHandlerParams
from ray.exceptions import RayError
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__)


def create_actor_handlers_patch(self, param: ActorHandlerParams) -> ray.actor.ActorHandle:
    """
    Handler to create actor, this patch add an extra name to the created actor.

    Args:
        self: RayActorGroup instance.
        param (ActorHandlerParams): params needed for create actor.
    """
    master_addr = param.master_addr if param.master_addr else "localhost"
    master_port = param.master_port if param.master_port else None
    world_size = param.world_size if param.world_size else 1
    rank = param.rank_index if param.rank_index else 0

    _check(master_port, world_size, rank)

    runtime_env = {
        "env_vars": {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port) if master_port is not None else "",
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
        }
    }

    try:
        actor_class_name = self.worker.__ray_metadata__.modified_class.__name__
    except AttributeError as e:
        logger.error(f"get actor class name failed, error: {e}")
        raise AttributeError("get actor class name failed.") from e
    except Exception as e:
        logger.error(f"Unexpected error occurred when get actor class name, error: {e}")
        raise RuntimeError("Unexpected error occurred when get actor class name.") from e

    try:
        worker = self.worker.options(
            name=f"{actor_class_name}_{param.rank_index}_{param.bundle_index}_{uuid.uuid4().hex[-10:]}",
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=param.placement_group,
                placement_group_bundle_index=param.bundle_index
            ),
            runtime_env=runtime_env
        ).remote(
            megatron_config=self.megatron_config,
            rl_config=self.rl_config,
            generate_config=self.generate_config,
            model_provider=self.model_provider,
            get_megatron_module=self.get_megatron_module,
            initialize_func=self.initialize_func,
            tokenizer=self.tokenizer,
            **self.kwargs
        )
    except RayError as e:
        logger.error(f"create actor failed, error: {e}")
        raise RayError("create actor failed.") from e
    except Exception as e:
        logger.error(f"Unexpected error occurred when create actor, error: {e}")
        raise RuntimeError("Unexpected error occurred when create actor.") from e
    return worker


def _check(master_port: int, world_size: int, rank: int):
    if master_port is not None:
        if not isinstance(master_port, int):
            logger.error("master port for create worker must be an integer")
            raise ValueError("master port for create worker must be an integer")
        if not (1 <= master_port <= 65535):
            logger.error("master port must be in range [1, 65535]")
            raise ValueError("master port must be in range [1, 65535]")

    if world_size < 1:
        logger.error("world size must be greater than or equal to 1.")
        raise ValueError("world size must be greater than or equal to 1.")

    if not (0 <= rank < world_size):
        logger.error("rank index must within range [0, world_size)")
        raise ValueError("rank index must within range [0, world_size)")


def update_ref_dispatch_size(self, batch_len: int):
    """Update ref_dispatch_size, experience count every forward step for reference"""
    if not isinstance(batch_len, int):
        raise ValueError("batch_len should be int type")
    actor_train_objs = []
    for actor in self.actor_handlers:
        actor_train_objs.append(actor.update_ref_dispatch_size.remote(batch_len))
    return ray.get(actor_train_objs)


def update_actor_logprob_dispatch_size(self, batch_len: int):
    """Update actor_logprob_dispatch_size, experience count every forward step for actor_logprob"""
    if not isinstance(batch_len, int):
        raise ValueError("batch_len should be int type")
    actor_train_objs = []
    for actor in self.actor_handlers:
        actor_train_objs.append(actor.update_actor_logprob_dispatch_size.remote(batch_len))
    return ray.get(actor_train_objs)


def update_actor_update_dispatch_size(self, batch_len: int):
    """Update actor_update_dispatch_size, experience count every forward step for actor update"""
    if not isinstance(batch_len, int):
        raise ValueError("batch_len should be int type")
    actor_train_objs = []
    for actor in self.actor_handlers:
        actor_train_objs.append(actor.update_actor_update_dispatch_size.remote(batch_len))
    return ray.get(actor_train_objs)


def update_mini_batch_size(
    self,
    original_n_samples_per_prompt: int,
    new_samples_per_prompt: int,
    use_stepwise_advantage: bool
):
    """Update mini_batch_size, mini batch size"""
    if not isinstance(original_n_samples_per_prompt, int) or not isinstance(new_samples_per_prompt, int):
        raise ValueError("original_n_samples_per_prompt and new_samples_per_prompt should be int type")
    if not isinstance(use_stepwise_advantage, bool):
        raise ValueError("use_stepwise_advantage should be bool type")
    actor_train_objs = []
    for actor in self.actor_handlers:
        actor_train_objs.append(
            actor.update_mini_batch_size.remote(
                original_n_samples_per_prompt,
                new_samples_per_prompt,
                use_stepwise_advantage
            )
        )
    return ray.get(actor_train_objs)
