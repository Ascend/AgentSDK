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

from typing import List

import ray

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.data_manager.data_transform import padding_dict_to_tensor_dict

logger = Loggers(__name__).get_logger()


class MindSpeedRLDataManager:
    def __init__(self):
        self.data_manager = None

    def sync_init_data_manager(self, data_manager):
        self.data_manager = data_manager

    def all_consumed(self, experience_consumer_stage):
        status = int(not ray.get(self.data_manager.all_consumed.remote(experience_consumer_stage)))
        logger.info(f'all_consumed status: {status}')
        return status

    def get_data(self, experience_consumer_stage, experience_columns, experience_count, get_n_samples=True):
        batch_data, index = ray.get(self.data_manager.get_experience.remote(experience_consumer_stage,
                                                                            experience_columns, experience_count,
                                                                            get_n_samples=get_n_samples))
        logger.info(f'get_transfer_dock_data batch_data: {batch_data.keys()}, {index}')
        if index:
            return batch_data, index
        return {}, []

    def put_data(self, output, index, metric=None):
        output = {key: value.cpu() if not isinstance(value, List) else value for key, value in output.items()}
        output = padding_dict_to_tensor_dict(output)
        self.data_manager.put_experience.remote(data_dict=output, indexes=index)

    def put_experience(self, batch_dict, indexes):
        self.data_manager.put_experience.remote(data_dict=batch_dict, indexes=indexes)

    def update_metrics(self, k, value, cumulate):
        ray.get(self.data_manager.update_metrics.remote(k, value, cumulate=cumulate))
