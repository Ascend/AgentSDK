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

import pytest
from datasets import Dataset

from agentic_rl.base.utils.data_loader import GRPODataLoader, RepeatSampler


class TestGRPODataLoader:

    @pytest.fixture
    def sample_dataset(self):
        # 创建简单的测试数据集
        data = {"input": [1, 2, 3], "label": [2, 3, 4]}
        return Dataset.from_dict(data)

    def test_init_with_empty_dataset(self, sample_dataset):
        # 测试空数据集
        empty_dataset = Dataset.from_dict({})
        with pytest.raises(ValueError, match="Provided dataset is None or contains no data!"):
            GRPODataLoader(empty_dataset, ["input", "label"], 2)

    def test_init_with_invalid_num_samples(self, sample_dataset):
        # 测试 num_samples 为空或者小于等于0
        GRPODataLoader(sample_dataset, ["input", "label"], 2, num_samples=None)
        GRPODataLoader(sample_dataset, ["input", "label"], 2, num_samples=0)

    def test_init_with_missing_keys(self, sample_dataset):
        # 测试缺少key的情况
        with pytest.raises(ValueError, match="Datasets must contain key"):
            GRPODataLoader(sample_dataset, ["input", "label", "missing_key"], 2)

    def test_init_with_invalid_inputs(self, sample_dataset):
        data_loader = GRPODataLoader(sample_dataset, ['input', 'label'], 2)

        data_iter = iter(data_loader)

        data = next(data_iter)
        assert data['input'][0] == 1
        assert data['input'][1] == 2
        assert data['label'][0] == 2
        assert data['label'][1] == 3

        with pytest.raises(StopIteration):
            next(data_iter)

    def test_init_with_custom_params(self, sample_dataset):
        data_loader = GRPODataLoader(sample_dataset,
                                     ["input", "label"],
                                     global_batch_size=2,
                                     no_shuffle=False,
                                     seed=52,
                                     num_workers=8)

        assert data_loader.batch_size == 2
        assert data_loader.num_workers == 8
        assert isinstance(data_loader.sampler, RepeatSampler)
