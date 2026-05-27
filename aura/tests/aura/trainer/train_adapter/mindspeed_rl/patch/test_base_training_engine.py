# -*- coding: utf-8 -*-
import pytest
import torch
from unittest.mock import patch, MagicMock


class TestBaseTrainingEngine:

    def test_split_batches_with_dynamic_bsz(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.base_training_engine import _split_batches_with_dynamic_bsz
        mock_self = MagicMock()
        batch = {
            'prompt_length': torch.tensor([10, 20, 30, 40]),
            'response_length': torch.tensor([5, 10, 15, 20]),
            'input_ids': torch.randn(4, 50),
        }
        max_packing_token = 100
        dynamic_max_batch_size = 2
        with patch('aura.trainer.train_adapter.mindspeed_rl.patch.base_training_engine.rearrange_micro_batches_patch') as mock_rearrange:
            mock_rearrange.return_value = [[0, 1], [2, 3]]
            result_batches, result_partitions = _split_batches_with_dynamic_bsz(mock_self, batch, max_packing_token, dynamic_max_batch_size)
            assert isinstance(result_batches, list)
            assert isinstance(result_partitions, list)
            assert len(result_batches) == 2
            assert result_partitions == [[0, 1], [2, 3]]

    def test_update_mini_batch_size(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.base_training_engine import update_mini_batch_size
        mock_self = MagicMock()
        mock_self.mini_batch_size_per_dp = 8
        update_mini_batch_size(mock_self, 2, 4)
        assert mock_self.mini_batch_size_per_dp_new_size == 8

        mock_self2 = MagicMock()
        mock_self2.mini_batch_size_per_dp = 8
        update_mini_batch_size(mock_self2, 2, 4, use_stepwise_advantage=True)
        assert mock_self2.mini_batch_size_per_dp_new_size == 16
