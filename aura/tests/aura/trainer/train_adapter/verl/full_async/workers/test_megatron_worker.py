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


import unittest
import sys
from unittest.mock import patch, MagicMock


# Define a simple base class replacement
class MockDetachActorWorker:
    pass

# Create mock objects
mock_ray = MagicMock()
mock_torch = MagicMock()
mock_verl = MagicMock()
mock_transformers = MagicMock()
mock_transformers_config = MagicMock()

# Set necessary mock values
mock_verl.Dispatch = MagicMock()
mock_verl.Dispatch.ONE_TO_ALL = 'ONE_TO_ALL'
mock_verl.register = MagicMock(return_value=lambda func: func)

# Set torch submodules
mock_torch.distributed = MagicMock()

# Set transformers submodules
mock_transformers.configuration_utils = mock_transformers_config
mock_transformers_config.PretrainedConfig = MagicMock()

# Create a proper mock module for engine_workers
mock_engine_workers = MagicMock()
mock_engine_workers.DetachActorWorker = MockDetachActorWorker

# Mock required dependency modules before importing the module under test
with patch.dict('sys.modules', {
    'ray': mock_ray,
    'torch': mock_torch,
    'torch.distributed': mock_torch.distributed,
    'transformers': mock_transformers,
    'transformers.configuration_utils': mock_transformers_config,
    'verl': mock_verl,
    'verl.experimental': MagicMock(),
    'verl.experimental.separation': MagicMock(),
    'verl.experimental.separation.engine_workers': mock_engine_workers,
    'verl.single_controller.base.decorator': mock_verl,
    'megatron': MagicMock(),
    'megatron.bridge': MagicMock(),
    'megatron.bridge.models': MagicMock(),
    'megatron.bridge.models.conversion': MagicMock(),
    'megatron.bridge.models.hf_pretrained': MagicMock(),
    'verl.utils': MagicMock(),
    'verl.utils.megatron_utils': MagicMock()
}):
    # Now we can safely import the class under test
    from aura.trainer.train_adapter.verl.full_async.workers.megatron_worker import MegatronDetachActorWorker


class TestMegatronDetachActorWorker(unittest.TestCase):
    def setUp(self):
        # Create test instance
        self.worker = MegatronDetachActorWorker()

        # Set necessary properties
        self.worker.actor = MagicMock()
        self.worker.actor.engine = MagicMock()
        self.worker.actor.engine.module = MagicMock()
        self.worker.actor.engine.engine_config = MagicMock()
        self.worker.actor.engine.engine_config.param_offload = False
        self.worker.rank = 0

        # Mock _save_weights method instead of _save_hf_weights
        self.worker._save_weights = MagicMock()

    def tearDown(self):
        # Reset all mocks
        mock_ray.reset_mock()
        mock_torch.reset_mock()

    def test_prepare_infer_params_to_cpu(self):
        # Set up environment
        weight_save_dir = '/tmp/test_weights'

        # Mock weight_updater actor
        mock_weight_actor = MagicMock()
        mock_ray.get_actor.return_value = mock_weight_actor

        # Call the test method
        self.worker.prepare_infer_params_to_cpu(weight_save_dir)

        # Verify calls
        self.worker._save_weights.assert_called_once()
        mock_ray.get_actor.assert_called_once_with("weight_updater", namespace="controller_raygroup")
        mock_weight_actor.weight_saved.remote.assert_called_once_with(weight_save_dir)

    def test_prepare_infer_params_to_cpu_get_actor_exception(self):
        # Set up environment
        weight_save_dir = '/tmp/test_weights'

        # Mock ray.get_actor to raise exception
        error_msg = "Actor not found"
        mock_ray.get_actor.side_effect = Exception(error_msg)

        # Call the test method and verify exception
        with self.assertRaises(Exception) as context:
            self.worker.prepare_infer_params_to_cpu(weight_save_dir)

        self.assertEqual(str(context.exception), error_msg)
        self.worker._save_weights.assert_called_once()

    def test_prepare_infer_params_to_cpu_save_weights_exception(self):
        # Set up environment
        weight_save_dir = '/tmp/test_weights'

        # Mock _save_weights to raise exception
        error_msg = "Save weights error"
        self.worker._save_weights.side_effect = Exception(error_msg)

        # Call the test method and verify exception
        with self.assertRaises(Exception) as context:
            self.worker.prepare_infer_params_to_cpu(weight_save_dir)

        self.assertEqual(str(context.exception), error_msg)
        mock_ray.get_actor.assert_not_called()

if __name__ == '__main__':
    unittest.main()
