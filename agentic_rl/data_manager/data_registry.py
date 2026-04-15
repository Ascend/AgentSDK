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

from agentic_rl.data_manager.infer_data import InferDataManager
from agentic_rl.data_manager.mindspeed_rl_data import MindSpeedRLDataManager
from agentic_rl.data_manager.verl_data import VerlDataManager


class DataManagerRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, train_backend, service_mode, cls):
        if train_backend not in self._registry:
            self._registry[train_backend] = {}
        self._registry[train_backend][service_mode] = cls

    def get_class(self, train_backend, service_mode):
        return self._registry.get(train_backend).get(service_mode)


registry = DataManagerRegistry()
registry.register("mindspeed_rl", "train", MindSpeedRLDataManager)
registry.register("mindspeed_rl", "infer", InferDataManager)
registry.register("verl", "train", VerlDataManager)
registry.register("verl", "infer", InferDataManager)


def data_manager_class(train_backend: str, service_mode: str):
    return registry.get_class(train_backend, service_mode)
