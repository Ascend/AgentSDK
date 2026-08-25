#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

import ray
from aura.controllers.utils.utils import create_actor, MAX_CONCURRENCY, DEFAULT_CPUS


class TrainBackendRegistry:
    def __init__(self):
        self._registry = dict()

    def register(self, train_engine, work_mode, train_method):
        self._registry[(train_engine, work_mode)] = train_method

    def get_method(self, train_engine, work_mode):
        return self._registry.get((train_engine, work_mode))


@ray.remote
class TrainRegister:
    def __init__(self):
        self.registry = TrainBackendRegistry()

    def registry_msrl_train(self):
        from aura.trainer.train_adapter.mindspeed_rl.hybrid_policy.train_service import train as msrl_hybrid_train
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import train as msrl_async_train
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service import (
            dummy_train as msrl_async_dummy_train,
        )

        self.registry.register("mindspeed_rl", "hybrid", msrl_hybrid_train)
        self.registry.register("mindspeed_rl", "one_step_off", msrl_async_train)
        self.registry.register("mindspeed_rl", "dummy_train", msrl_async_dummy_train)

    def registry_verl_train(self):
        from aura.trainer.train_adapter.verl.full_async.train_main import start_train as verl_async_train
        from aura.trainer.train_adapter.verl.hybrid.train_main import start_train as verl_hybrid_train
        from aura.trainer.train_adapter.verl.full_async.train_main import start_dummy_train as verl_async_dummy_train
        from aura.trainer.train_adapter.verl.fully_async.train_main import start_train as verl_fully_async_train

        self.registry.register("verl", "hybrid", verl_hybrid_train)
        self.registry.register("verl", "one_step_off", verl_async_train)
        self.registry.register("verl", "dummy_train", verl_async_dummy_train)
        self.registry.register("verl", "fully_async", verl_fully_async_train)

    def get_method(self, train_engine, work_mode):
        return self.registry.get_method(train_engine, work_mode)


TRAIN_REGISTER_ACTOR_NAME = "trainer_register_actor"
TRAIN_REGISTER_NAMESPACE = "register_raygroup"


def get_train_method(train_engine, work_mode):
    actor = ray.get_actor(TRAIN_REGISTER_ACTOR_NAME, namespace=TRAIN_REGISTER_NAMESPACE)
    return ray.get(actor.get_method.remote(train_engine, work_mode))


def get_train_actor():
    return create_actor(
        name=TRAIN_REGISTER_ACTOR_NAME,
        cls=TrainRegister,
        namespace=TRAIN_REGISTER_NAMESPACE,
        options={"num_cpus": DEFAULT_CPUS, "max_concurrency": MAX_CONCURRENCY},
    )
