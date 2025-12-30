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

from agentic_rl.data_manager.mindspeed_rl_data import MindSpeedRLDataManager
from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__)


class DataManagerRegistry:
    def __init__(self):
        logger.info("DataManagerRegistry initialized")
        self._registry = {}

    def register(self, train_backend, cls):
        if not train_backend or not isinstance(train_backend, str):
            raise ValueError("train_backend must be a non-empty string")
        if not isinstance(cls, type):
            raise TypeError("cls must be a class")
        self._registry[train_backend] = cls

    def get_class(self, train_backend):
        if not train_backend or not isinstance(train_backend, str):
            raise ValueError("train_backend must be a non-empty string")
        if train_backend not in self._registry:
            raise KeyError(f"No data manager class found for train_backend '{train_backend}'")
        return self._registry.get(train_backend)


registry = DataManagerRegistry()
registry.register("mindspeed_rl", MindSpeedRLDataManager)


def data_manager_class(train_backend: str):
    if not train_backend or not isinstance(train_backend, str):
        raise ValueError("train_backend must be a non-empty string")
    cls = registry.get_class(train_backend)
    if cls is None:
        raise ValueError(f"No data manager class found for train_backend '{train_backend}'")
    return cls
