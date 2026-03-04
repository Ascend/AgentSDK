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
from unittest.mock import patch, MagicMock, Mock

import pytest
import torch


class Mock1:
    def __init__(self):
        self.experience_columns = ["key1"]
        self.experience_consumers = ["key1"]
        self.prompts_num = 1


class MockGRPOTransferDock:
    __ray_metadata__ = MagicMock()
    __ray_metadata__.modified_class = Mock1
    __ray_metadata__.modified_class.__name__ = "MockGRPOTransferDock"


class TestGRPOTransferDock:
    @pytest.fixture(scope="class")
    def patch_modules(self):
        # Patch modules to avoid importing actual dependencies
        with patch.dict(sys.modules, {
            "mindspeed_rl": MagicMock(),
            "mindspeed_rl.trainer": MagicMock(),
            "mindspeed_rl.trainer.grpo_trainer_hybrid": MagicMock(),
        }):
            yield

    @pytest.fixture(scope="class")
    def patch_target(self, patch_modules):
        # Patch GRPOTransferDock and ray.remote for testing
        with patch('mindspeed_rl.trainer.grpo_trainer_hybrid.GRPOTransferDock', MockGRPOTransferDock), \
                patch("ray.remote") as mock_remote:

            def fake_remote(*args, **kwargs):
                if len(args) == 1 and callable(args[0]) and not kwargs:
                    obj = args[0]
                    obj.remote = obj
                    return obj
                else:
                    def decorator(obj):
                        obj.remote = obj
                        return obj

                    return decorator

            mock_remote.side_effect = fake_remote

            yield

    @patch('ray.get')
    def test_reset_experience_len_with_valid_int(self, mock_ray_get, patch_target):
        # Test using a valid integer as max_len
        max_len = 10
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.grpo_transfer_dock import GRPOTransferDock
        dock = GRPOTransferDock()
        dock.reset_experience_len(max_len)

        # Check that max_len is set correctly
        assert dock.max_len == max_len

        # Check that experience_data is initialized correctly
        experience_data = dock.experience_data
        for key in dock.experience_columns:
            assert len(experience_data[key]) == max_len
            assert experience_data[key] == [None] * max_len

        # Check that experience_data_status is initialized correctly
        experience_data_status = dock.experience_data_status
        for key in dock.experience_columns:
            assert experience_data_status[key].shape == (max_len,)
            assert torch.all(experience_data_status[key] == 0)

        # Check that experience_consumer_status is initialized correctly
        experience_consumer_status = dock.experience_consumer_status
        for key in dock.experience_consumers:
            assert experience_consumer_status[key].shape == (max_len,)
            assert torch.all(experience_consumer_status[key] == 0)

        # Check that n_samples_per_prompt is computed correctly
        assert dock.n_samples_per_prompt == max_len // dock.prompts_num

    def test_reset_experience_len_with_invalid_type(self, patch_target):
        # Test using an invalid type (string) as max_len
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.grpo_transfer_dock import GRPOTransferDock
        dock = GRPOTransferDock()
        max_len = "10"  # string type
        with pytest.raises(ValueError):
            dock.reset_experience_len(max_len)


if __name__ == '__main__':
    unittest.main()
