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

from typing import List

import ray

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.data_manager.data_transform import padding_dict_to_tensor_dict

logger = Loggers(__name__)


class MindSpeedRLDataManager:
    def __init__(self):
        logger.info("MindSpeedRLDataManager initialized")
        self.data_manager = None

    def sync_init_data_manager(self, data_manager):
        if data_manager is None:
            raise ValueError("data_manager cannot be None")
        miss_methods = (not hasattr(data_manager, 'all_consumed') or
                        not hasattr(data_manager, 'get_experience') or
                        not hasattr(data_manager, 'put_experience') or
                        not hasattr(data_manager, 'update_metrics'))
        if miss_methods:
            raise AttributeError(
                "data_manager must have all_consumed, get_experience, put_experience, and update_metrics methods")
        self.data_manager = data_manager

    def all_consumed(self, experience_consumer_stage):
        try:
            if not isinstance(experience_consumer_stage, str) or experience_consumer_stage == "":
                raise ValueError("experience_consumer_stage must be a non-empty string")
            status = int(not ray.get(self.data_manager.all_consumed.remote(experience_consumer_stage)))
            return status
        except ray.exceptions.RayError as e:
            raise RuntimeError(f"Ray operation failed in all_consumed: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to check all_consumed: {str(e)}") from e

    def get_data(self, experience_consumer_stage, experience_columns, experience_count, get_n_samples=True):
        try:
            if not isinstance(experience_consumer_stage, str) or experience_consumer_stage == "":
                raise ValueError("experience_consumer_stage must be a non-empty string")
            if not isinstance(experience_columns, list) or len(experience_columns) == 0:
                raise ValueError("experience_columns must be a non-empty list")
            if not isinstance(experience_count, int) or experience_count <= 0:
                raise ValueError("experience_count must be a positive integer")
            if not isinstance(get_n_samples, bool):
                raise TypeError("get_n_samples must be a boolean")
            batch_data, index = ray.get(
                self.data_manager.get_experience.remote(
                    experience_consumer_stage, experience_columns, experience_count, get_n_samples=get_n_samples))
            if index:
                logger.info(f"Successfully retrieved data with index: {index}")
                return batch_data, index
            logger.warning("No data retrieved, returning empty dictionary and empty list")
            return {}, []
        except ray.exceptions.RayError as e:
            raise RuntimeError(f"Ray operation failed in get_data: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve data: {str(e)}") from e

    def put_data(self, output, index):
        if not isinstance(output, dict):
            raise TypeError("output must be a dictionary")
        if not output:
            logger.warning("output is empty")
            return
        if not isinstance(index, list) or len(index) == 0:
            raise ValueError("index must be a non-empty list")
        output_tmp = {}
        for key, value in output.items():
            try:
                output_tmp[key] = value.cpu() if not isinstance(value, List) else value
            except AttributeError:
                output_tmp[key] = value
        try:
            output = padding_dict_to_tensor_dict(output_tmp)
            self.data_manager.put_experience.remote(data_dict=output, indexes=index)
        except ray.exceptions.RayError as e:
            raise RuntimeError(f"Ray operation failed in put_data: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to put data: {str(e)}") from e

    def update_metrics(self, k, value, cumulate):
        try:
            if not isinstance(k, str) or k == "":
                raise ValueError("k must be a non-empty string")
            if not isinstance(value, (list, tuple)):
                raise TypeError("value must be a list or tuple")
            if not all(isinstance(v, (int, float)) for v in value):
                raise TypeError("value must be a number list")
            if not isinstance(cumulate, bool):
                raise TypeError("cumulate must be a boolean")
            ray.get(self.data_manager.update_metrics.remote(k, value, cumulate=cumulate))
        except ray.exceptions.RayError as e:
            raise RuntimeError(f"Ray operation failed in update_metrics: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to update metrics: {str(e)}") from e

    def reset_experience_len(self, experience_len):
        if not isinstance(experience_len, int) or experience_len <= 0:
            raise ValueError("experience_len must be a positive integer")
        self.data_manager.reset_experience_len.remote(experience_len)
