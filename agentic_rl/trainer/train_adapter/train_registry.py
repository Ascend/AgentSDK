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

from typing import Optional
from ray.remote_function import RemoteFunction

from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__)


class TrainBackendRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name: str, train_fn: RemoteFunction):
        if not hasattr(train_fn, "remote"):
            raise ValueError(f"train_fn {name} is not a remote function")
        self._registry[name] = train_fn

    def get(self, name: str) -> Optional[RemoteFunction]:
        return self._registry.get(name, None)


train_backend_registry = TrainBackendRegistry()


def register_train_fn(name: str):
    if train_backend_registry.get(name) is not None:
        logger.warning(f"train_backend {name} is already registered")
        return
    if name == "mindspeed_rl":
        from agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo import (
            train as _msrl_train,
        )

        train_backend_registry.register(name, _msrl_train)
    elif name == "verl":
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import (
            train as _verl_train,
        )

        train_backend_registry.register(name, _verl_train)
    else:
        logger.error(f"train_backend {name} is not supported")
        raise ValueError(f"train_backend {name} is not supported")


def get_train_fn(name: str) -> Optional[RemoteFunction]:
    register_train_fn(name)
    logger.info(f"train_backend {name} is used.")
    return train_backend_registry.get(name)
