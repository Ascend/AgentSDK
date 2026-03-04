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
import sys
import unittest
from unittest.mock import patch, MagicMock

import pytest
import torch


class TestBaseTrainingEngine:
    @pytest.fixture(scope="class")
    def patch_modules(self):
        with patch.dict(sys.modules, {
            "mindspeed_rl": MagicMock(),
            "mindspeed_rl.utils": MagicMock(),
            "mindspeed_rl.utils.utils": MagicMock(),
            "mindspeed_rl.utils.compute": MagicMock(),
        }):
            yield

    @pytest.fixture(scope="class")
    def patch_target(self, patch_modules):
        with patch('mindspeed_rl.utils.utils.append_to_dict') as mock_append_to_dict, \
            patch('mindspeed_rl.utils.compute.get_parallel_state') as mock_get_parallel_state:
            patches = {
                "mock_append_to_dict": mock_append_to_dict,
                "mock_get_parallel_state": mock_get_parallel_state
            }

            def fake_append_to_dict(*args, **kwargs):
                pass

            def fake_get_parallel_state(*args, **kwargs):
                return MagicMock()

            mock_append_to_dict.side_effect = fake_append_to_dict
            mock_get_parallel_state.side_effect = fake_get_parallel_state

            yield patches

    def test_update_mini_batch_size_with_stepwise_advantage(self, patch_target):
        # Test the case where use_stepwise_advantage is True
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.base_training_engine import update_mini_batch_size
        obj = MagicMock()
        obj.mini_batch_size_per_dp_new_size = 0
        obj.mini_batch_size_per_dp = 100
        original_n_samples_per_prompt = 10
        new_samples_per_prompt = 20
        update_mini_batch_size(obj, original_n_samples_per_prompt, new_samples_per_prompt, True)
        assert obj.mini_batch_size_per_dp_new_size == 200

    def test_update_mini_batch_size_without_stepwise_advantage(self, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.base_training_engine import update_mini_batch_size
        # Test the case where use_stepwise_advantage is False
        obj = MagicMock()
        obj.mini_batch_size_per_dp_new_size = 0
        obj.mini_batch_size_per_dp = 100
        update_mini_batch_size(obj, 1, 2, False)
        assert obj.mini_batch_size_per_dp_new_size == 100

    def test_update_mini_batch_size_with_zero_samples(self, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.base_training_engine import update_mini_batch_size
        # Test the case where original_n_samples_per_prompt is 0
        obj = MagicMock()
        obj.mini_batch_size_per_dp_new_size = 0
        obj.mini_batch_size_per_dp = 100
        with pytest.raises(ValueError):
            update_mini_batch_size(obj, 0, 1, True)

    def test_update_with_valid_data(self, patch_target):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.base_training_engine import update
        mock_get_parallel_state = patch_target["mock_get_parallel_state"]
        mock_get_parallel_state.get_data_parallel_world_size = MagicMock()
        mock_get_parallel_state.get_data_parallel_world_size.return_value = 2
        # Mock input data
        test_data = {
            'tensor1': MagicMock(),
            'tensor2': [MagicMock(), MagicMock()],
            'none_value': None
        }

        mock_engine = MagicMock()
        mock_engine.kl_ctrl = None
        mock_engine.mini_batch_size_per_dp_new_size = 32
        mock_engine.shuffle_mini_batch = False
        mock_engine.epochs = 1
        mock_engine.optimizer = MagicMock()
        mock_engine.opt_param_scheduler = MagicMock()
        mock_engine.role = "trainer"
        # Mock model parameters
        mock_param = MagicMock()

        def f():
            return iter([torch.tensor([1.0])])

        mock_param.parameters.side_effect = f
        mock_engine.model = [mock_param]

        # Configure mock return values
        mock_engine._split_batches.return_value = [test_data]  # Simulate batch splitting result
        mock_engine._forward_backward_batch.return_value = {"loss": 0.5}
        mock_engine.optimizer.step.return_value = (True, 1.0, 0)  # (success, grad_norm, zero_grads)
        result = update(mock_engine, test_data)

        # Verify expected behavior
        mock_engine._split_batches.assert_called_once()
        mock_engine.optimizer.zero_grad.assert_called_once()
        assert result['trainer/grad_norm'] == [1.0]
        mock_engine.opt_param_scheduler.step.assert_called_once()


if __name__ == '__main__':
    unittest.main()
