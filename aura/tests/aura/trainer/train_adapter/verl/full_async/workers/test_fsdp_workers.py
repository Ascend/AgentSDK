#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of MulanPSL2 at:
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

# Create mock objects
mock_ray = MagicMock()
mock_dcp_module = MagicMock()
mock_verl = MagicMock()

# Set necessary mock values
mock_verl.Dispatch = MagicMock()
mock_verl.Dispatch.ONE_TO_ALL = 'ONE_TO_ALL'
mock_verl.register = MagicMock(return_value=lambda func: func)

# Set dcp module properties - dcp.save is called
mock_dcp_module.save = MagicMock()
mock_dcp_module.state_dict = MagicMock()
mock_dcp_module.state_dict.get_model_state_dict = MagicMock()

# Define a simple base class replacement
class MockDetachActorWorker:
    pass

# Create a proper mock module for engine_workers
mock_engine_workers = MagicMock()
mock_engine_workers.DetachActorWorker = MockDetachActorWorker

# Mock required dependency modules
sys.modules['ray'] = mock_ray
sys.modules['verl'] = mock_verl
sys.modules['verl.experimental'] = MagicMock()
sys.modules['verl.experimental.separation'] = MagicMock()
sys.modules['verl.experimental.separation.engine_workers'] = mock_engine_workers
sys.modules['verl.single_controller.base.decorator'] = mock_verl

# Now import the class under test
from aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers import FsdpDetachActorWorker


class TestFsdpDetachActorWorker(unittest.TestCase):
    def setUp(self):
        # Reset side_effects from previous tests
        mock_dcp_module.save.side_effect = None
        mock_dcp_module.state_dict.get_model_state_dict.side_effect = None
        mock_ray.get_actor.side_effect = None

        # Create test instance
        self.worker = FsdpDetachActorWorker()

        # Set necessary properties
        self.worker.actor = MagicMock()
        self.worker.actor.engine = MagicMock()
        self.worker.actor.engine.module = MagicMock()

    def tearDown(self):
        # Clean up mock modules to avoid affecting other tests
        modules_to_remove = ['ray', 'verl', 'verl.experimental', 'verl.experimental.separation',
                            'verl.experimental.separation.engine_workers', 'verl.single_controller.base.decorator']
        for module_name in modules_to_remove:
            if module_name in sys.modules:
                del sys.modules[module_name]

    def test_prepare_infer_params_to_cpu(self):
        # Set up environment
        weight_save_dir = '/tmp/test_weights'
        mock_state_dict = {'param1': MagicMock(), 'param2': MagicMock()}

        # Reset mocks
        mock_dcp_module.state_dict.get_model_state_dict.reset_mock()
        mock_dcp_module.save.reset_mock()
        mock_ray.get_actor.reset_mock()

        # Configure mocks
        mock_dcp_module.state_dict.get_model_state_dict.return_value = mock_state_dict
        mock_weight_actor = MagicMock()
        mock_ray.get_actor.return_value = mock_weight_actor

        # Use patch to mock torch.distributed.get_rank and dcp functions
        with patch('aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers.torch.distributed.get_rank', return_value=0):
            with patch('aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers.dcp.save', mock_dcp_module.save):
                with patch('aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers.get_model_state_dict', mock_dcp_module.state_dict.get_model_state_dict):
                    # Call the test method
                    self.worker.prepare_infer_params_to_cpu(weight_save_dir)

        # Verify calls
        mock_dcp_module.state_dict.get_model_state_dict.assert_called_once_with(self.worker.actor.engine.module)
        mock_dcp_module.save.assert_called_once_with(state_dict=mock_state_dict, checkpoint_id=weight_save_dir)
        mock_ray.get_actor.assert_called_once_with("weight_updater", namespace="controller_raygroup")
        mock_weight_actor.weight_saved.remote.assert_called_once_with(weight_save_dir)

    def test_prepare_infer_params_to_cpu_get_model_state_dict_exception(self):
        # Set up environment
        weight_save_dir = '/tmp/test_weights'

        # Reset mocks
        mock_dcp_module.state_dict.get_model_state_dict.reset_mock()
        mock_dcp_module.save.reset_mock()

        # Mock get_model_state_dict to raise exception
        error_msg = "Get model state dict error"
        mock_dcp_module.state_dict.get_model_state_dict.side_effect = Exception(error_msg)

        # Use patch to mock torch.distributed.get_rank and dcp functions
        with patch('aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers.torch.distributed.get_rank', return_value=0):
            with patch('aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers.dcp.save', mock_dcp_module.save):
                with patch('aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers.get_model_state_dict', mock_dcp_module.state_dict.get_model_state_dict):
                    # Call the test method and verify exception
                    with self.assertRaises(Exception) as context:
                        self.worker.prepare_infer_params_to_cpu(weight_save_dir)

        self.assertEqual(str(context.exception), error_msg)
        mock_dcp_module.save.assert_not_called()

    def test_prepare_infer_params_to_cpu_save_exception(self):
        # Set up environment
        weight_save_dir = '/tmp/test_weights'
        mock_state_dict = {'param1': MagicMock(), 'param2': MagicMock()}

        # Reset mocks
        mock_dcp_module.state_dict.get_model_state_dict.reset_mock()
        mock_dcp_module.save.reset_mock()

        # Configure mocks
        mock_dcp_module.state_dict.get_model_state_dict.return_value = mock_state_dict

        # Mock dcp.save to raise exception
        error_msg = "Save error"
        mock_dcp_module.save.side_effect = Exception(error_msg)

        # Use patch to mock torch.distributed.get_rank and dcp functions
        with patch('aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers.torch.distributed.get_rank', return_value=0):
            with patch('aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers.dcp.save', mock_dcp_module.save):
                with patch('aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers.get_model_state_dict', mock_dcp_module.state_dict.get_model_state_dict):
                    # Call the test method and verify exception
                    with self.assertRaises(Exception) as context:
                        self.worker.prepare_infer_params_to_cpu(weight_save_dir)

        self.assertEqual(str(context.exception), error_msg)
        mock_dcp_module.state_dict.get_model_state_dict.assert_called_once()

    def test_prepare_infer_params_to_cpu_get_actor_exception(self):
        # Set up environment
        weight_save_dir = '/tmp/test_weights'
        mock_state_dict = {'param1': MagicMock(), 'param2': MagicMock()}

        # Reset mocks
        mock_dcp_module.state_dict.get_model_state_dict.reset_mock()
        mock_dcp_module.save.reset_mock()
        mock_ray.get_actor.reset_mock()

        # Configure mocks
        mock_dcp_module.state_dict.get_model_state_dict.return_value = mock_state_dict

        # Mock ray.get_actor to raise exception
        error_msg = "Actor not found"
        mock_ray.get_actor.side_effect = Exception(error_msg)

        # Use patch to mock torch.distributed.get_rank and dcp functions
        with patch('aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers.torch.distributed.get_rank', return_value=0):
            with patch('aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers.dcp.save', mock_dcp_module.save):
                with patch('aura.trainer.train_adapter.verl.full_async.workers.fsdp_workers.get_model_state_dict', mock_dcp_module.state_dict.get_model_state_dict):
                    # Call the test method and verify exception
                    with self.assertRaises(Exception) as context:
                        self.worker.prepare_infer_params_to_cpu(weight_save_dir)

        self.assertEqual(str(context.exception), error_msg)
        mock_dcp_module.state_dict.get_model_state_dict.assert_called_once()
        mock_dcp_module.save.assert_called_once_with(state_dict=mock_state_dict, checkpoint_id=weight_save_dir)

if __name__ == '__main__':
    unittest.main()
