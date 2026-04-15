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


import os
import time
import unittest
import sys
from unittest.mock import patch, MagicMock


# Create mock objects
mock_ray = MagicMock()
mock_recipe = MagicMock()
mock_verl = MagicMock()

# Set necessary mock values
mock_verl.Dispatch = MagicMock()
mock_verl.Dispatch.ONE_TO_ALL = 'ONE_TO_ALL'
mock_verl.register = MagicMock(return_value=lambda func: func)

# Define a simple base class replacement
class MockDetachActorWorker:
    pass

# Set recipe module properties
mock_recipe.DetachActorWorker = MockDetachActorWorker

# Mock required dependency modules before importing the module under test
with patch.dict('sys.modules', {
    'ray': mock_ray,
    'recipe.fully_async_policy.megatron_worker': mock_recipe,
    'verl.single_controller.base.decorator': mock_verl
}):
    # Now we can safely import the class under test
    from agentic_rl.trainer.train_adapter.verl.full_async.workers.megatron_worker import MegatronDetachActorWorker


class TestMegatronDetachActorWorker(unittest.TestCase):
    def setUp(self):
        # Create test instance
        self.worker = MegatronDetachActorWorker()
        
        # Set necessary properties
        self.worker.bridge = MagicMock()
        self.worker.actor = MagicMock()
        self.worker.actor.actor_module = MagicMock()
        
        # Mock time.time()
        self.time_patch = patch('time.time', return_value=12345.6789)
        self.time_patch.start()
        
        # Mock os.makedirs
        self.makedirs_patch = patch('os.makedirs')
        self.mock_makedirs = self.makedirs_patch.start()
    
    def tearDown(self):
        # Stop all patches
        self.time_patch.stop()
        self.makedirs_patch.stop()
        
        # Reset all mocks
        mock_ray.reset_mock()
        self.worker.bridge.reset_mock()
        self.worker.actor.actor_module.reset_mock()
    
    def test_prepare_infer_params_to_cpu_normal(self):
        # Set up environment
        weight_save_dir = '/tmp/test_weights'
        sync_group_name = 'test_sync_group'
        
        # Mock weight_updater actor
        mock_weight_actor = MagicMock()
        mock_ray.get_actor.side_effect = None  # Clear any possible side effects
        mock_ray.get_actor.return_value = mock_weight_actor
        
        # Call the test method
        self.worker.prepare_infer_params_to_cpu(weight_save_dir, sync_group_name)
        
        # Verify calls
        # 1. Verify os.makedirs is called
        self.mock_makedirs.assert_called_once_with(weight_save_dir, exist_ok=True)
        
        # 3. Verify bridge.save_weights is called
        self.worker.bridge.save_weights.assert_called_once_with(
            self.worker.actor.actor_module, weight_save_dir
        )
        
        # 4. Verify ray.get_actor is called
        mock_ray.get_actor.assert_called_once_with("weight_updater", namespace="controller_raygroup")
        
        # 5. Verify weight_saved.remote method is called (Ray remote call)
        mock_weight_actor.weight_saved.remote.assert_called_once_with(weight_save_dir)
    
    def test_prepare_infer_params_to_cpu_default_sync_group(self):
        # Set up environment
        weight_save_dir = '/tmp/test_weights'
        
        # Mock weight_updater actor
        mock_weight_actor = MagicMock()
        mock_ray.get_actor.side_effect = None  # Clear any possible side effects
        mock_ray.get_actor.return_value = mock_weight_actor
        
        # Call the test method without specifying sync_group_name
        self.worker.prepare_infer_params_to_cpu(weight_save_dir)
        
        # Verify bridge.save_weights is called with default sync_group_name
        self.worker.bridge.save_weights.assert_called_once_with(
            self.worker.actor.actor_module, weight_save_dir
        )
        
        # Verify other calls
        self.mock_makedirs.assert_called_once_with(weight_save_dir, exist_ok=True)
        mock_ray.get_actor.assert_called_once()
    
    def test_prepare_infer_params_to_cpu_makedirs_exception(self):
        # Set up environment
        weight_save_dir = '/tmp/test_weights'
        
        # Mock os.makedirs to raise exception
        error_msg = "Permission denied"
        self.mock_makedirs.side_effect = PermissionError(error_msg)
        
        # Call the test method and verify exception
        with self.assertRaises(PermissionError) as context:
            self.worker.prepare_infer_params_to_cpu(weight_save_dir)
        
        self.assertEqual(str(context.exception), error_msg)
        
        # Verify bridge.save_weights is not called
        self.worker.bridge.save_weights.assert_not_called()
        
        # Verify ray.get_actor is not called
        mock_ray.get_actor.assert_not_called()
    
    def test_prepare_infer_params_to_cpu_save_weights_exception(self):
        # Set up environment
        weight_save_dir = '/tmp/test_weights'
        
        # Mock bridge.save_weights to raise exception
        error_msg = "Save weights failed"
        self.worker.bridge.save_weights.side_effect = Exception(error_msg)
        
        # Mock weight_updater actor
        mock_weight_actor = MagicMock()
        mock_ray.get_actor.side_effect = None  # Clear any possible side effects
        mock_ray.get_actor.return_value = mock_weight_actor
        
        # Call the test method and verify exception
        with self.assertRaises(Exception) as context:
            self.worker.prepare_infer_params_to_cpu(weight_save_dir)
        
        self.assertEqual(str(context.exception), error_msg)
        
        # Verify os.makedirs is called
        self.mock_makedirs.assert_called_once_with(weight_save_dir, exist_ok=True)
        
        # Verify ray.get_actor is not called
        mock_ray.get_actor.assert_not_called()
    
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
        
        # Verify os.makedirs is called
        self.mock_makedirs.assert_called_once_with(weight_save_dir, exist_ok=True)
        
        # Verify bridge.save_weights is called
        self.worker.bridge.save_weights.assert_called_once_with(
            self.worker.actor.actor_module, weight_save_dir
        )


if __name__ == '__main__':
    unittest.main()