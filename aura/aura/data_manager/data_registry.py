#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from aura.data_manager.infer_data import InferDataManager
from aura.data_manager.mindspeed_rl_data import MindSpeedRLDataManager
from aura.data_manager.verl_data import VerlDataManager
from threading import Lock


class DataManagerRegistry:
    def __init__(self):
        self._registry = dict()

    def register(self, train_backend, service_mode, instance):
        self._registry[(train_backend, service_mode)] = instance

    def get_instance(self, train_backend, service_mode):
        return self._registry.get((train_backend, service_mode))


@ray.remote
class DataManagerRegistryActor:
    def __init__(self):
        self.registry = DataManagerRegistry()
        self.lock = Lock()
        self.registry_done = False
        self.msrl_data_instance = None
        self.verl_data_instance = None
        self.infer_data_instance = None

    def registry_msrl_data_manager(self):
        with self.lock:
            if self.registry_done:
                return
            self.msrl_data_instance = MindSpeedRLDataManager()
            self.infer_data_instance = InferDataManager()
            self.registry.register("mindspeed_rl", "train", self.msrl_data_instance)
            self.registry.register("mindspeed_rl", "infer", self.infer_data_instance)
            self.registry_done = True

    def registry_verl_data_manager(self):
        with self.lock:
            if self.registry_done:
                return
            self.verl_data_instance = VerlDataManager()
            self.infer_data_instance = InferDataManager()
            self.registry.register("verl", "train", self.verl_data_instance)
            self.registry.register("verl", "infer", self.infer_data_instance)
            self.registry_done = True

    def get_instance(self, train_backend, service_mode):
        return self.registry.get_instance(train_backend, service_mode)


DATA_REGISTER_ACTOR_NAME = "data_register_actor"
DATA_REGISTER_NAMESPACE = "register_raygroup"


def get_data_manager_actor():
    return DataManagerRegistryActor.options(
        name=DATA_REGISTER_ACTOR_NAME, namespace=DATA_REGISTER_NAMESPACE, lifetime="detached"
    ).remote()


def get_data_manager_instance(train_backend: str, service_mode: str):
    actor = ray.get_actor(DATA_REGISTER_ACTOR_NAME, namespace=DATA_REGISTER_NAMESPACE)
    return ray.get(actor.get_instance.remote(train_backend, service_mode))
