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

from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch

from verl import DataProto


class VerlDataManager:
    """
    The data manager implementation of the Verl backend

    Responsibilities:
    1. Obtain data in the mindspeed_rl format from TrainController
    2. Convert it into the data format of Verl
    3. Provide an interface compatible with MindSpeedRLDataManager
    """

    def __init__(self):
        self.controller = None
        self._current_batch: Optional[DataProto] = None
        self._metrics: Dict[str, Any] = {}
        self._pad_token_id: int = 0

    def sync_init_data_manager(self, controller_or_config):
        """
        Initialize the data manager

        Args:
            controller_or_config: TrainController instance or configuration dictionary
        """
        if hasattr(controller_or_config, 'get_next_training_batch'):
            self.controller = controller_or_config
        else:
            # Processing in the case of configuration dictionary
            self._config = controller_or_config

    def all_consumed(self, experience_consumer_stage: str) -> int:
        """
        Check whether all the data have been consumed.

        Returns:
            int: 0 indicates that there is still data available, while 1 indicates that all data has been consumed.
        """
        # In the Verl mode, the consumption status is managed by the controller.
        return 0

    def get_data(
            self,
            experience_consumer_stage: str,
            experience_columns: Optional[List[str]],
            experience_count: int,
            get_n_samples: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        """
        Obtain training data (in Verl format)

        Args:
            experience_consumer_stage: Consumption stage identifier
            experience_columns: Required field list (None indicates all)
            experience_count: The requested sample quantity
            get_n_samples: Whether to obtain n samples

        Returns:
            tuple: (batch_dict, indices)
                - batch_dict: Verl format data dictionary
                - indices: Sample Index List
        """
        if self.controller is None:
            return {}, []

        # 1. Obtain the original data from the controller
        raw_batch, metric = self.controller.get_next_training_batch(False)

        # 2. Storage indicators
        if metric:
            self._metrics.update(metric)

        return raw_batch, [0]

    def put_data(self, output: Dict, index: List[int], metric: Optional[Dict] = None):
        """
        Write the training result data

        Args:
            output: Training result dictionary
            index: Sample index
            metric: Optional indicator data
        """
        # Convert the output to DataProto and merge them together
        data_proto = self._dict_to_dataproto(output)

        if self._current_batch is None:
            self._current_batch = data_proto
        else:
            self._current_batch = self._current_batch.union(data_proto)

        if metric:
            self._metrics.update(metric)

    def put_experience(self, batch_dict: Dict, indexes: List[int]):
        """Directly input the empirical data"""
        self.put_data(batch_dict, indexes)

    def update_metrics(self, k: str, value: Any, cumulate: bool):
        """Update training metrics"""
        if cumulate and k in self._metrics:
            if isinstance(self._metrics[k], list):
                self._metrics[k].extend(value if isinstance(value, list) else [value])
            elif isinstance(self._metrics[k], (int, float)):
                self._metrics[k] = self._metrics[k] + value
        else:
            self._metrics[k] = value

    def get_metrics(self) -> Dict:
        """Obtain all indicators"""
        return self._metrics.copy()

    def clear(self):
        """Clear the data and indicators"""
        self._current_batch = None
        self._metrics.clear()

    # ==================== verl 特有方法 ====================

    def get_current_batch(self) -> Optional[DataProto]:
        """Obtain the current batch of DataProto"""
        return self._current_batch

    def set_current_batch(self, batch: DataProto):
        """Set the current DataProto batch"""
        self._current_batch = batch

    def set_pad_token_id(self, pad_token_id: int):
        """Set padding token ID"""
        self._pad_token_id = pad_token_id

    def _dict_to_dataproto(self, data_dict: Dict) -> DataProto:
        """Convert the dict to DataProto"""
        tensors = {}
        non_tensors = {}

        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
            elif isinstance(value, np.ndarray):
                if value.dtype == object:
                    non_tensors[key] = value
                else:
                    tensors[key] = torch.from_numpy(value)
            elif isinstance(value, list):
                non_tensors[key] = np.array(value, dtype=object)
            else:
                non_tensors[key] = np.array([value], dtype=object)

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
