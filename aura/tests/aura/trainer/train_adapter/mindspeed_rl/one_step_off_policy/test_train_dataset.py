#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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

import sys
import os
import json
import pytest
import numpy as np
from unittest.mock import patch, mock_open


if 'mindspeed_rl.datasets.base_dataset' not in sys.modules:
    sys.modules['mindspeed_rl.datasets.base_dataset'] = type(sys)('mindspeed_rl.datasets.base_dataset')

class MockBaseDataset:
    def __init__(self, dataset, dataset_type):
        self.dataset = dataset
        self.dataset_type = dataset_type

class MockPackedDataset:
    def __init__(self, length, items=None):
        self._length = length
        self._items = items if items is not None else [{'input_ids': np.array([1, 2, 3, 4, 5])} for _ in range(length)]
        self.applied_filter = None

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return self._items[idx]

    @property
    def datasets(self):
        return {}

sys.modules['mindspeed_rl.datasets.base_dataset.BaseDataset'] = MockBaseDataset


class MockIndexedDatasetModule:
    _current_dataset = MockPackedDataset(100)

    @classmethod
    def set_dataset(cls, dataset):
        cls._current_dataset = dataset

    @classmethod
    def get_packed_indexed_dataset(cls, data_prefix, filter_length=None):
        return cls._current_dataset

sys.modules['mindspeed_rl.datasets.indexed_dataset'] = MockIndexedDatasetModule


class TestTrainDataset:

    def test_init_with_packed_data(self):
        mock_meta = {
            'batch_group': list(range(100)),
        }
        mock_packed_dataset = MockPackedDataset(100)
        MockIndexedDatasetModule.set_dataset(mock_packed_dataset)

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_meta))):
            with patch('os.path.exists', return_value=True):
                from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataset import TrainDataset

                mock_tokenizer = lambda x: x
                mock_extra_param = type('obj', (object,), {'max_prompt_length': 512})

                dataset = TrainDataset(
                    data_prefix='/path/to/data',
                    is_packed_data=True,
                    tokenizer=mock_tokenizer,
                    seq_length=2048,
                    num_samples=100,
                    name='test',
                    seed=42,
                    full_shuffle_instruction_dataset=False,
                    token_param=None,
                    preprocess_template=None,
                    pad_token=0,
                    eos_token=1,
                    extra_param=mock_extra_param,
                )

                assert dataset.data_prefix == '/path/to/data'
                assert dataset.is_packed_data == True
                assert dataset.seq_length == 2048
                assert dataset.num_samples == 100
                assert dataset.args == mock_extra_param

    def test_init_without_packed_data(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataset import TrainDataset

        mock_tokenizer = lambda x: x

        with pytest.raises(NotImplementedError, match="one step off Dataset currently supports only packed data"):
            TrainDataset(
                data_prefix='/path/to/data',
                is_packed_data=False,
                tokenizer=mock_tokenizer,
                seq_length=2048,
            )

    def test_init_with_meta(self):
        mock_meta = {
            'batch_group': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
            'prompt_id': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            'mini_batch_id': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
        }
        mock_packed_dataset = MockPackedDataset(10)
        MockIndexedDatasetModule.set_dataset(mock_packed_dataset)

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_meta))):
            with patch('os.path.exists', return_value=True):
                from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataset import TrainDataset

                mock_tokenizer = lambda x: x
                mock_extra_param = None

                dataset = TrainDataset(
                    data_prefix='/path/to/data',
                    is_packed_data=True,
                    tokenizer=mock_tokenizer,
                    seq_length=2048,
                    extra_param=mock_extra_param,
                )

                assert 'batch_group' in dataset.side_meta
                assert 'prompt_id' in dataset.side_meta
                assert 'mini_batch_id' in dataset.side_meta
                assert len(dataset.side_meta['batch_group']) == 10

    def test_len(self):
        mock_meta = {
            'batch_group': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
        }
        mock_packed_dataset = MockPackedDataset(10)
        MockIndexedDatasetModule.set_dataset(mock_packed_dataset)

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_meta))):
            with patch('os.path.exists', return_value=True):
                from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataset import TrainDataset

                mock_tokenizer = lambda x: x
                mock_extra_param = None

                dataset = TrainDataset(
                    data_prefix='/path/to/data',
                    is_packed_data=True,
                    tokenizer=mock_tokenizer,
                    seq_length=2048,
                    extra_param=mock_extra_param,
                )

                assert len(dataset) == 4

    def test_getitem(self):
        mock_meta = {
            'batch_group': [0, 0, 1, 1, 2, 2],
            'prompt_id': [0, 1, 0, 1, 0, 1],
            'mini_batch_id': [0, 0, 1, 1, 2, 2],
        }

        items = [
            {'input_ids': np.array([1, 2, 3, 4, 5])},
            {'input_ids': np.array([6, 7, 8, 9, 10])},
            {'input_ids': np.array([11, 12, 13, 14, 15])},
            {'input_ids': np.array([16, 17, 18, 19, 20])},
            {'input_ids': np.array([21, 22, 23, 24, 25])},
            {'input_ids': np.array([26, 27, 28, 29, 30])},
        ]
        mock_packed_dataset = MockPackedDataset(6, items)
        MockIndexedDatasetModule.set_dataset(mock_packed_dataset)

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_meta))):
            with patch('os.path.exists', return_value=True):
                from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataset import TrainDataset

                mock_tokenizer = lambda x: x
                mock_extra_param = type('obj', (object,), {'dataset_additional_keys': []})

                dataset = TrainDataset(
                    data_prefix='/path/to/data',
                    is_packed_data=True,
                    tokenizer=mock_tokenizer,
                    seq_length=2048,
                    extra_param=mock_extra_param,
                )

                group_0 = dataset[0]
                assert len(group_0) == 2
                assert 'input_ids' in group_0[0]
                assert 'attention_mask' in group_0[0]
                assert 'prompt_id' in group_0[0]
                assert 'mini_batch_id' in group_0[0]

    def test_get_field_array_from_side_meta(self):
        mock_meta = {
            'batch_group': [0, 0, 1, 1, 2],
        }
        mock_packed_dataset = MockPackedDataset(5)
        MockIndexedDatasetModule.set_dataset(mock_packed_dataset)

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_meta))):
            with patch('os.path.exists', return_value=True):
                from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataset import TrainDataset

                mock_tokenizer = lambda x: x
                mock_extra_param = None

                dataset = TrainDataset(
                    data_prefix='/path/to/data',
                    is_packed_data=True,
                    tokenizer=mock_tokenizer,
                    seq_length=2048,
                    extra_param=mock_extra_param,
                )

                result = dataset._get_field_array('batch_group')
                assert isinstance(result, np.ndarray)
                assert len(result) == 5

    def test_get_field_array_from_datasets(self):
        mock_meta = {'batch_group': [0, 1, 2]}

        items_with_custom_field = [
            {'input_ids': np.array([1, 2, 3]), 'custom_field': 0},
            {'input_ids': np.array([4, 5, 6]), 'custom_field': 1},
            {'input_ids': np.array([7, 8, 9]), 'custom_field': 2},
        ]

        class MockPackedDatasetWithCustomField(MockPackedDataset):
            def __init__(self):
                super().__init__(3, items_with_custom_field)

            @property
            def datasets(self):
                return {'custom_field': True}

        mock_packed_dataset = MockPackedDatasetWithCustomField()
        MockIndexedDatasetModule.set_dataset(mock_packed_dataset)

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_meta))):
            with patch('os.path.exists', return_value=True):
                from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataset import TrainDataset

                mock_tokenizer = lambda x: x
                mock_extra_param = None

                dataset = TrainDataset(
                    data_prefix='/path/to/data',
                    is_packed_data=True,
                    tokenizer=mock_tokenizer,
                    seq_length=2048,
                    extra_param=mock_extra_param,
                )

                result = dataset._get_field_array('custom_field')
                assert isinstance(result, np.ndarray)
                assert list(result) == [0, 1, 2]

    def test_get_field_array_not_found(self):
        mock_meta = {'batch_group': [0, 1, 2]}
        mock_packed_dataset = MockPackedDataset(3)
        MockIndexedDatasetModule.set_dataset(mock_packed_dataset)

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_meta))):
            with patch('os.path.exists', return_value=True):
                from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataset import TrainDataset

                mock_tokenizer = lambda x: x
                mock_extra_param = None

                dataset = TrainDataset(
                    data_prefix='/path/to/data',
                    is_packed_data=True,
                    tokenizer=mock_tokenizer,
                    seq_length=2048,
                    extra_param=mock_extra_param,
                )

                with pytest.raises(KeyError, match="Field 'nonexistent' not found"):
                    dataset._get_field_array('nonexistent')

    def test_merge_side_meta_into_item(self):
        mock_meta = {
            'batch_group': [0, 0, 1, 1, 2],
            'prompt_id': [0, 1, 2, 3, 4],
            'mini_batch_id': [0, 0, 1, 1, 2],
        }
        mock_packed_dataset = MockPackedDataset(5)
        MockIndexedDatasetModule.set_dataset(mock_packed_dataset)

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_meta))):
            with patch('os.path.exists', return_value=True):
                from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataset import TrainDataset

                mock_tokenizer = lambda x: x
                mock_extra_param = None

                dataset = TrainDataset(
                    data_prefix='/path/to/data',
                    is_packed_data=True,
                    tokenizer=mock_tokenizer,
                    seq_length=2048,
                    extra_param=mock_extra_param,
                )

                item = {'input_ids': np.array([1, 2, 3])}
                dataset._merge_side_meta_into_item(item, 2)

                assert item['prompt_id'] == 2
                assert item['mini_batch_id'] == 1

    def test_cut_instruction_token_with_labels(self):
        mock_meta = {'batch_group': [0]}
        mock_packed_dataset = MockPackedDataset(1)
        MockIndexedDatasetModule.set_dataset(mock_packed_dataset)

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_meta))):
            with patch('os.path.exists', return_value=True):
                from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataset import TrainDataset

                mock_tokenizer = lambda x: x
                mock_extra_param = type('obj', (object,), {'dataset_additional_keys': []})

                dataset = TrainDataset(
                    data_prefix='/path/to/data',
                    is_packed_data=True,
                    tokenizer=mock_tokenizer,
                    seq_length=10,
                    extra_param=mock_extra_param,
                )

                item = {
                    'input_ids': np.array([1, 2, 3, 4, 5]),
                    'labels': np.array([-100, -100, 2, 3, 4]),
                }
                result = dataset._cut_instruction_token(item, np.int64, 0)

                assert 'input_ids' in result
                assert 'attention_mask' in result
                assert 'labels' in result
                assert len(result['input_ids']) == 5

    def test_cut_instruction_token_truncate(self):
        mock_meta = {'batch_group': [0]}
        mock_packed_dataset = MockPackedDataset(1)
        MockIndexedDatasetModule.set_dataset(mock_packed_dataset)

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_meta))):
            with patch('os.path.exists', return_value=True):
                from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataset import TrainDataset

                mock_tokenizer = lambda x: x
                mock_extra_param = type('obj', (object,), {'dataset_additional_keys': []})

                dataset = TrainDataset(
                    data_prefix='/path/to/data',
                    is_packed_data=True,
                    tokenizer=mock_tokenizer,
                    seq_length=5,
                    extra_param=mock_extra_param,
                )

                item = {
                    'input_ids': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    'labels': np.array([-100] * 10),
                }
                result = dataset._cut_instruction_token(item, np.int64, 0)

                assert len(result['input_ids']) == 5
                assert len(result['labels']) == 5

    def test_cut_instruction_token_with_additional_keys(self):
        mock_meta = {'batch_group': [0]}
        mock_packed_dataset = MockPackedDataset(1)
        MockIndexedDatasetModule.set_dataset(mock_packed_dataset)

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_meta))):
            with patch('os.path.exists', return_value=True):
                from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataset import TrainDataset

                mock_tokenizer = lambda x: x
                mock_extra_param = type('obj', (object,), {'dataset_additional_keys': ['response_mask']})

                dataset = TrainDataset(
                    data_prefix='/path/to/data',
                    is_packed_data=True,
                    tokenizer=mock_tokenizer,
                    seq_length=10,
                    extra_param=mock_extra_param,
                )

                item = {
                    'input_ids': np.array([1, 2, 3, 4, 5]),
                    'response_mask': np.array([0, 0, 1, 1, 1]),
                }
                result = dataset._cut_instruction_token(item, np.int64, 0)

                assert 'response_mask' in result
                assert list(result['response_mask']) == [0, 0, 1, 1, 1]

    def test_cut_instruction_token_without_labels(self):
        mock_meta = {'batch_group': [0]}
        mock_packed_dataset = MockPackedDataset(1)
        MockIndexedDatasetModule.set_dataset(mock_packed_dataset)

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_meta))):
            with patch('os.path.exists', return_value=True):
                from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataset import TrainDataset

                mock_tokenizer = lambda x: x
                mock_extra_param = type('obj', (object,), {'dataset_additional_keys': []})

                dataset = TrainDataset(
                    data_prefix='/path/to/data',
                    is_packed_data=True,
                    tokenizer=mock_tokenizer,
                    seq_length=10,
                    extra_param=mock_extra_param,
                )

                item = {'input_ids': np.array([1, 2, 3, 4, 5])}
                result = dataset._cut_instruction_token(item, np.int64, 0)

                assert 'input_ids' in result
                assert 'attention_mask' in result
                assert 'labels' not in result
