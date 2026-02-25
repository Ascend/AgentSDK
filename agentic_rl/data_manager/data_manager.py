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

from agentic_rl.data_manager.data_registry import data_manager_class
from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__)


class DataManager:
    def __init__(self, train_backend="mindspeed_rl"):
        if not train_backend:
            raise ValueError("train_backend cannot be empty")
        if not isinstance(train_backend, str):
            raise TypeError("train_backend must be a string")
        logger.info("DataManager initialized")
        self.train_backend = train_backend
        if train_backend == "verl":
            logger.info("verl train backend use DataProto, skip init data manager")
            return
        self.data_manager_instance = data_manager_class(train_backend)()
        logger.debug("Created data_manager_instance")

    def sync_init_data_manager(self, data_manager):
        if data_manager is None:
            raise ValueError("data_manager is None")
        logger.info("Starting to sync_init_data_manager with data_manager")
        self.data_manager_instance.sync_init_data_manager(data_manager)
        logger.info("sync_init_data_manager completed successfully")

    def all_consumed(self, experience_consumer_stage):
        if experience_consumer_stage is None or experience_consumer_stage == "":
            raise ValueError("experience_consumer_stage is None or empty")
        return self.data_manager_instance.all_consumed(experience_consumer_stage)

    def get_data(self, experience_consumer_stage, experience_columns, experience_count, get_n_samples=True):
        if experience_consumer_stage is None or experience_consumer_stage == "":
            raise ValueError("experience_consumer_stage cannot be None or empty")
        if not isinstance(experience_columns, list) or len(experience_columns) == 0:
            raise ValueError("experience_columns must be a non-empty list")
        if not isinstance(experience_count, int) or experience_count <= 0:
            raise ValueError("experience_count must be a positive integer")
        if not isinstance(get_n_samples, bool):
            raise TypeError("get_n_samples must be a boolean")
        return self.data_manager_instance.get_data(experience_consumer_stage,
                                                   experience_columns, experience_count, get_n_samples)

    def put_data(self, output, index):
        if output is None:
            raise ValueError("output cannot be None")
        if index is None:
            raise ValueError("index cannot be None")
        self.data_manager_instance.put_data(output, index)

    def update_metrics(self, k, value, cumulate):
        if k is None or k == "" or not isinstance(k, str):
            raise ValueError("k cannot be None or empty")
        if not all(isinstance(v, (int, float)) for v in value):
            raise TypeError("value must be a number int or float")
        if not isinstance(cumulate, bool):
            raise TypeError("cumulate must be a boolean")
        self.data_manager_instance.update_metrics(k, value, cumulate)

    def reset_experience_len(self, experience_len):
        if not isinstance(experience_len, int) or experience_len <= 0:
            raise ValueError("experience_len must be a positive integer")
        self.data_manager_instance.reset_experience_len(experience_len)
