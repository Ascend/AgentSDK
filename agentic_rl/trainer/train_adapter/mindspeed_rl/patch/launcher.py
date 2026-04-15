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
import uuid

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from mindspeed_rl.workers.scheduler.launcher import ActorHandlerParams
from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


def create_actor_handlers_patch(self, param: ActorHandlerParams) -> ray.actor.ActorHandle:
    """Create a Ray actor handler with a unique name and placement strategy."""
    logger.info("create_actor_handlers_patch")
    runtime_env = {
        "env_vars": {
            "MASTER_ADDR": param.master_addr if param.master_addr else "localhost",
            "MASTER_PORT": str(param.master_port) if param.master_port else "",
            "WORLD_SIZE": str(param.world_size),
            "RANK": str(param.rank_index),
        }
    }
    actor_class_name = self.worker.__ray_metadata__.modified_class.__name__
    return self.worker.options(
        name=f"{actor_class_name}_{param.rank_index}_{param.bundle_index}_{uuid.uuid4().hex[-10:]}",
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=param.placement_group,
            placement_group_bundle_index=param.bundle_index
        ),
        runtime_env=runtime_env
    ).remote(
        self.megatron_config,
        self.rl_config,
        self.generate_config,
        model_provider=self.model_provider,
        get_megatron_module=self.get_megatron_module,
        initialize_func=self.initialize_func,
        profiler_config=self.profiler_config,
        msprobe_config=self.msprobe_config,
        tokenizer=self.tokenizer,
        **self.kwargs
    )