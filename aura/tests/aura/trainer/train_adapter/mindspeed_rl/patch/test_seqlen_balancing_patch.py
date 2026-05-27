# -*- coding: utf-8 -*-
import pytest
import torch
from unittest.mock import patch, MagicMock


class TestSeqlenBalancingPatch:

    def test_heapq_partition_with_max(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch import heapq_partition_with_max
        seqlen_list = [10, 20, 30, 40, 50]
        partitions = heapq_partition_with_max(seqlen_list, 2, 70)
        assert len(partitions) >= 2
        total_items = sum(len(p) for p in partitions)
        assert total_items == len(seqlen_list)

    def test_rearrange_micro_batches_raw(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch import rearrange_micro_batches_raw
        seqlen_list = [10, 20, 30, 40]
        with patch('aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch.dist') as mock_dist:
            mock_dist.is_initialized.return_value = False
            with patch('aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch.karmarkar_karp', return_value=[[0, 1], [2, 3]]):
                partitions = rearrange_micro_batches_raw(seqlen_list, 100)
                assert partitions is not None
                assert len(partitions) == 2

    def test_rearrange_micro_batches_raw_with_dynamic_bsz(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch import rearrange_micro_batches_raw
        seqlen_list = [10, 20, 30, 40, 50, 60]
        with patch('aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch.dist') as mock_dist:
            mock_dist.is_initialized.return_value = False
            with patch('aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch.karmarkar_karp', return_value=[[0], [1], [2], [3], [4], [5]]):
                partitions = rearrange_micro_batches_raw(seqlen_list, 100, dynamic_max_batch_size=2)
                assert partitions is not None
                assert len(partitions) >= 3

    def test_rearrange_micro_batches_raw_exceeds_max(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch import rearrange_micro_batches_raw
        seqlen_list = [10, 20, 30, 150]
        with pytest.raises(ValueError):
            rearrange_micro_batches_raw(seqlen_list, 100)

    def test_rearrange_micro_batches_patch(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch import rearrange_micro_batches_patch
        seqlen_list = [10, 20, 30, 40]
        with patch('aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch.dist') as mock_dist:
            mock_dist.is_initialized.return_value = False
            mock_dist.get_rank.return_value = 0
            with patch('aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch.karmarkar_karp', return_value=[[0, 1], [2, 3]]):
                partitions = rearrange_micro_batches_patch(seqlen_list, 100)
                assert partitions is not None
                assert len(partitions) >= 1

    def test_rearrange_micro_batches_patch_with_overflow(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch import rearrange_micro_batches_patch
        seqlen_list = [90, 80, 70, 60]
        with patch('aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch.dist') as mock_dist:
            mock_dist.is_initialized.return_value = False
            mock_dist.get_rank.return_value = 0
            with patch('aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch.karmarkar_karp', return_value=[[0], [1], [2], [3]]):
                partitions = rearrange_micro_batches_patch(seqlen_list, 100)
                assert partitions is not None

    def test_rearrange_micro_batches_patch_rep_zero(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch import rearrange_micro_batches_patch
        seqlen_list = [95, 95, 95, 5]
        with patch('aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch.dist') as mock_dist:
            mock_dist.is_initialized.return_value = False
            mock_dist.get_rank.return_value = 0
            with patch('aura.trainer.train_adapter.mindspeed_rl.patch.seqlen_balancing_patch.karmarkar_karp', return_value=[[0, 3], [1], [2]]):
                partitions = rearrange_micro_batches_patch(seqlen_list, 100)
                assert partitions is not None
