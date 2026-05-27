# -*- coding: utf-8 -*-
import sys
import pytest
import torch
from unittest.mock import MagicMock, patch


class MockBaseDataset:
    def __init__(self, dataset, dataset_type):
        self.dataset = dataset
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


if 'mindspeed_rl.datasets.base_dataset' not in sys.modules:
    sys.modules['mindspeed_rl.datasets.base_dataset'] = MagicMock()
sys.modules['mindspeed_rl.datasets.base_dataset.BaseDataset'] = MockBaseDataset
sys.modules['mindspeed_rl.datasets.indexed_dataset'] = MagicMock()
sys.modules['mindspeed_rl.datasets.indexed_dataset.get_packed_indexed_dataset'] = MagicMock(return_value=MagicMock())


class TestTrainDataLoader:

    def test_train_data_loader_init(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataloader import TrainDataLoader

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__getitem__ = MagicMock(return_value=[
            {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1], 'prompt_id': 0, 'mini_batch_id': 0}
        ])

        loader = TrainDataLoader(
            dataset=mock_dataset,
            num_workers=2,
            seed=42,
            dataset_additional_keys=[],
            no_shuffle=False,
        )
        assert loader is not None
        assert loader.dataset_additional_keys == []
        assert loader.groups_per_step == 1

    def test_train_data_loader_with_shuffle(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataloader import TrainDataLoader

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__getitem__ = MagicMock(return_value=[
            {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1], 'prompt_id': 0, 'mini_batch_id': 0}
        ])

        loader = TrainDataLoader(
            dataset=mock_dataset,
            num_workers=2,
            seed=42,
            dataset_additional_keys=[],
            no_shuffle=False,
        )
        assert loader is not None

    def test_train_data_loader_without_shuffle(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataloader import TrainDataLoader

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__getitem__ = MagicMock(return_value=[
            {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1], 'prompt_id': 0, 'mini_batch_id': 0}
        ])

        loader = TrainDataLoader(
            dataset=mock_dataset,
            num_workers=2,
            seed=42,
            dataset_additional_keys=[],
            no_shuffle=True,
        )
        assert loader is not None

    def test_train_data_loader_collator_with_labels(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataloader import TrainDataLoader

        sample = {
            'input_ids': [1, 2, 3],
            'attention_mask': [1, 1, 1],
            'labels': [1, 2, 3],
            'prompt_id': 0,
            'mini_batch_id': 0,
        }
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__getitem__ = MagicMock(return_value=[sample])

        loader = TrainDataLoader(
            dataset=mock_dataset,
            num_workers=0,
            seed=42,
            dataset_additional_keys=[],
            no_shuffle=True,
        )

        data_iter = iter(loader)
        batch = next(data_iter)

        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch
        assert 'prompt_id' in batch
        assert 'mini_batch_id' in batch

    def test_train_data_loader_collator_without_labels(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataloader import TrainDataLoader

        sample = {
            'input_ids': [1, 2, 3],
            'attention_mask': [1, 1, 1],
            'prompt_id': 0,
            'mini_batch_id': 0,
        }
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__getitem__ = MagicMock(return_value=[sample])

        loader = TrainDataLoader(
            dataset=mock_dataset,
            num_workers=0,
            seed=42,
            dataset_additional_keys=[],
            no_shuffle=True,
        )

        data_iter = iter(loader)
        batch = next(data_iter)

        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' not in batch
        assert 'prompt_id' in batch
        assert 'mini_batch_id' in batch

    def test_train_data_loader_collator_with_additional_keys(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataloader import TrainDataLoader

        sample = {
            'input_ids': [1, 2, 3],
            'attention_mask': [1, 1, 1],
            'prompt_id': 0,
            'mini_batch_id': 0,
            'response_mask': [1, 1, 1],
        }
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__getitem__ = MagicMock(return_value=[sample])

        loader = TrainDataLoader(
            dataset=mock_dataset,
            num_workers=0,
            seed=42,
            dataset_additional_keys=['response_mask'],
            no_shuffle=True,
        )

        data_iter = iter(loader)
        batch = next(data_iter)

        assert 'response_mask' in batch

    def test_train_data_loader_collator_with_groups_per_step(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataloader import TrainDataLoader

        sample = {
            'input_ids': [1, 2, 3],
            'attention_mask': [1, 1, 1],
            'prompt_id': 0,
            'mini_batch_id': 0,
        }
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__getitem__ = MagicMock(return_value=[sample])

        loader = TrainDataLoader(
            dataset=mock_dataset,
            num_workers=0,
            seed=42,
            dataset_additional_keys=[],
            no_shuffle=True,
            groups_per_step=2,
        )

        assert loader.groups_per_step == 2

    def test_optimize_train_dataloader(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataloader import optimize_train_dataloader

        actor_config = MagicMock()
        actor_config.data_path = '/path/to/data'
        actor_config.split = '90,10,0'
        actor_config.seq_length = 2048
        actor_config.num_workers = 0
        actor_config.dataset_additional_keys = []
        actor_config.no_shuffle = False
        actor_config.global_batch_size = 8
        actor_config.seed = 42

        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=100)
        mock_train_ds.__getitem__ = MagicMock(return_value=[
            {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1], 'prompt_id': 0, 'mini_batch_id': 0}
        ])

        with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataloader.build_train_valid_test_datasets') as mock_build:
            mock_build.return_value = (mock_train_ds, None, None)
            result = optimize_train_dataloader(actor_config, 1000, 0)
            assert result is not None

    def test_optimize_train_dataloader_with_consumed_samples(self):
        from aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataloader import optimize_train_dataloader

        actor_config = MagicMock()
        actor_config.data_path = '/path/to/data'
        actor_config.split = '90,10,0'
        actor_config.seq_length = 2048
        actor_config.num_workers = 0
        actor_config.dataset_additional_keys = []
        actor_config.no_shuffle = False
        actor_config.global_batch_size = 8
        actor_config.seed = 42

        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=100)
        mock_train_ds.__getitem__ = MagicMock(return_value=[
            {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1], 'prompt_id': 0, 'mini_batch_id': 0}
        ])

        with patch('aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_dataloader.build_train_valid_test_datasets') as mock_build:
            mock_build.return_value = (mock_train_ds, None, None)
            result = optimize_train_dataloader(actor_config, 1000, 16)
            assert result is not None
