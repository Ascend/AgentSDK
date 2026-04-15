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

from agentic_rl.data_manager.data_registry import data_manager_class


class DataManager:
    def __init__(self, train_backend, service_mode="train"):
        self.data_manager_instance = data_manager_class(train_backend, service_mode)()

    def sync_init_data_manager(self, data_manager):
        self.data_manager_instance.sync_init_data_manager(data_manager)

    def all_consumed(self, experience_consumer_stage):
        return self.data_manager_instance.all_consumed(experience_consumer_stage)

    def get_data(self, experience_consumer_stage, experience_columns, experience_count, get_n_samples=True):
        return self.data_manager_instance.get_data(experience_consumer_stage,
                                                   experience_columns, experience_count, get_n_samples)

    def put_data(self, output, index, metric=None):
        self.data_manager_instance.put_data(output, index, metric)

    def put_experience(self, batch_dict, indexes):
        self.data_manager_instance.put_experience(batch_dict, indexes)

    def update_metrics(self, k, value, cumulate):
        self.data_manager_instance.update_metrics(k, value, cumulate)

    def set_pad_token_id(self, pad_token_id: int):
        if hasattr(self.data_manager_instance, 'set_pad_token_id'):
            self.data_manager_instance.set_pad_token_id(pad_token_id)

    def set_pad_token_id_from_tokenizer(self, tokenizer) -> int:
        if tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            pad_token_id = tokenizer.eos_token_id
        else:
            pad_token_id = 0

        self.set_pad_token_id(pad_token_id)
        return pad_token_id

    def get_pad_token_id_info(self) -> dict:
        pad_token_id = getattr(self.data_manager_instance, '_pad_token_id', None)
        return pad_token_id
