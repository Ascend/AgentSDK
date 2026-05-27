# -*- coding: utf-8 -*-
import pytest
from unittest.mock import MagicMock, patch


class TestDefaultTrainDataloader:

    def test_default_train_dataloader(self):
        from aura.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader import default_train_dataloader

        data_loader_config = MagicMock()
        data_loader_config.train_data_path = '/path/to/train'
        data_loader_config.split = '90,5,5'
        data_loader_config.seq_length = 2048
        data_loader_config.train_iters = 1000
        data_loader_config.global_batch_size = 8
        data_loader_config.seed = 42
        data_loader_config.num_workers = 2
        data_loader_config.dataset_additional_keys = []
        data_loader_config.no_shuffle = False
        data_loader_config.test_data_path = None

        mock_train_ds = MagicMock()
        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader.build_train_valid_test_datasets',
                   return_value=(mock_train_ds, None, None)):
            data_iters, val_dataloader, test_dataloader = default_train_dataloader(
                data_loader_config, 100, 0
            )
            assert data_iters is not None
            assert val_dataloader is None
            assert test_dataloader is None

    def test_default_train_dataloader_with_test_data(self):
        from aura.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader import default_train_dataloader

        data_loader_config = MagicMock()
        data_loader_config.train_data_path = '/path/to/train'
        data_loader_config.split = '90,5,5'
        data_loader_config.seq_length = 2048
        data_loader_config.train_iters = 1000
        data_loader_config.global_batch_size = 8
        data_loader_config.seed = 42
        data_loader_config.num_workers = 2
        data_loader_config.dataset_additional_keys = []
        data_loader_config.no_shuffle = False
        data_loader_config.test_data_path = '/path/to/test'

        mock_train_ds = MagicMock()
        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader.build_train_valid_test_datasets',
                   return_value=(mock_train_ds, None, None)):
            with patch('aura.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader.build_validate_test_dataloader',
                       return_value=(MagicMock(), MagicMock())):
                data_iters, val_dataloader, test_dataloader = default_train_dataloader(
                    data_loader_config, 100, 0
                )
                assert data_iters is not None
                assert val_dataloader is not None
                assert test_dataloader is not None

    def test_default_train_dataloader_with_resumption(self):
        from aura.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader import default_train_dataloader

        data_loader_config = MagicMock()
        data_loader_config.train_data_path = '/path/to/train'
        data_loader_config.split = '90,5,5'
        data_loader_config.seq_length = 2048
        data_loader_config.train_iters = 1000
        data_loader_config.global_batch_size = 8
        data_loader_config.seed = 42
        data_loader_config.num_workers = 2
        data_loader_config.dataset_additional_keys = []
        data_loader_config.no_shuffle = False
        data_loader_config.test_data_path = None

        mock_train_ds = MagicMock()
        mock_dataloader = MagicMock()
        mock_iter = iter([1, 2, 3])
        mock_dataloader.__iter__ = MagicMock(return_value=mock_iter)

        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader.build_train_valid_test_datasets',
                   return_value=(mock_train_ds, None, None)):
            with patch('aura.trainer.train_adapter.mindspeed_rl.utils.default_train_dataloader.PromptDataLoader',
                       return_value=mock_dataloader):
                data_iters, val_dataloader, test_dataloader = default_train_dataloader(
                    data_loader_config, 100, 16
                )
                assert data_iters is not None
