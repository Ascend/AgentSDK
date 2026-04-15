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

from collections import deque

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.controllers.rollout_controller.rollout_client import RolloutClient

logger = Loggers(__name__).get_logger()


class InferDataManager:
    def __init__(self):
        self.data_manager = deque()

    def all_consumed(self, experience_consumer_stage):
        status = len(self.data_manager)
        logger.info(f'all_consumed status: {status}')
        return status

    def get_data(self, experience_consumer_stage, experience_columns, experience_count, get_n_samples=True):
        batch_data, index = self.data_manager.popleft()
        logger.info(f'get_transfer_dock_data batch_data: {batch_data.keys()}, {index}')
        if index:
            return batch_data, index
        return {}, []

    def put_data(self, output, index, metric=None):
        rollout_client = RolloutClient()
        rollout_client.send_outputs_to_train_server(output, metric)

    def put_experience(self, batch_dict, indexes):
        self.data_manager.append((batch_dict, indexes))

    def update_metrics(self, k, value, cumulate):
        pass
