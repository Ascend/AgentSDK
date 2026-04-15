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
from unittest.mock import patch, MagicMock, Mock


# Create mock objects
mock_ray = MagicMock()
mock_torch = MagicMock()
mock_fsdp_module = MagicMock()
mock_torch_distributed = MagicMock()
mock_safetensors = MagicMock()
mock_recipe = MagicMock()
mock_verl = MagicMock()

# Set necessary mock values
mock_verl.Dispatch = MagicMock()
mock_verl.Dispatch.ONE_TO_ALL = 'ONE_TO_ALL'
mock_verl.register = MagicMock(return_value=lambda func: func)

# Set torch submodules
mock_torch.distributed = mock_torch_distributed

# Set fsdp module properties
mock_fsdp_module.FullyShardedDataParallel = MagicMock()
mock_fsdp_module.StateDictType = MagicMock()
mock_fsdp_module.FullStateDictConfig = MagicMock()

# Add state_dict_type attribute to FullyShardedDataParallel
mock_fsdp_module.FullyShardedDataParallel.state_dict_type = MagicMock()

# Ensure StateDictType.FULL_STATE_DICT has a value
mock_fsdp_module.StateDictType.FULL_STATE_DICT = 'FULL_STATE_DICT'

# Define a simple base class replacement
class MockDetachActorWorker:
    pass

# Set recipe module properties
mock_recipe.DetachActorWorker = MockDetachActorWorker

# Mock required dependency modules before importing the module under test
with patch.dict('sys.modules', {
    'ray': mock_ray,
    'torch': mock_torch,
    'torch.distributed.fsdp': mock_fsdp_module,
    'torch.distributed': mock_torch_distributed,
    'safetensors.torch': mock_safetensors,
    'recipe.fully_async_policy.fsdp_workers': mock_recipe,
    'verl.single_controller.base.decorator': mock_verl
}):
    # Now we can safely import the class under test
    from agentic_rl.trainer.train_adapter.verl.full_async.workers.fsdp_workers import FsdpDetachActorWorker

# Since the module under test dynamically imports safetensors.torch inside methods, we need to ensure this mock is valid throughout all tests
# Assign mock_safetensors.save_file to a global variable to ensure all test methods can use it
mock_save_file = mock_safetensors.save_file

class TestFsdpDetachActorWorker(unittest.TestCase):
    def setUp(self):
        # Create test instance
        self.worker = FsdpDetachActorWorker()
        
        # Set necessary properties
        self.worker.actor_module_fsdp = MagicMock()
        
        # Mock time.time()
        self.time_patch = patch('time.time', return_value=12345.6789)
        self.time_patch.start()
        
        # Mock os.makedirs
        self.makedirs_patch = patch('os.makedirs')
        self.mock_makedirs = self.makedirs_patch.start()
        
        # Mock safetensors.torch module (resolve dynamic import issue)
        self.safetensors_patch = patch.dict('sys.modules', {'safetensors.torch': mock_safetensors})
        self.safetensors_patch.start()
    
    def tearDown(self):
        # Stop all patches
        self.time_patch.stop()
        self.makedirs_patch.stop()
        self.safetensors_patch.stop()
        
        # Reset all mocks
        mock_ray.reset_mock()
        mock_torch.reset_mock()
        mock_fsdp_module.reset_mock()
        mock_torch_distributed.reset_mock()
        mock_safetensors.reset_mock()
    
    def test_prepare_infer_params_to_cpu_rank0(self):
        # Set up environment
        weight_save_dir = '/tmp/test_weights'
        mock_state_dict = {'param1': MagicMock(), 'param2': MagicMock()}
        
        # Mock rank as 0
        mock_torch_distributed.get_rank.return_value = 0
        
        # Mock state_dict call
        self.worker.actor_module_fsdp.state_dict.return_value = mock_state_dict
        
        # Mock weight_updater actor
        mock_weight_actor = MagicMock()
        mock_ray.get_actor.return_value = mock_weight_actor
        
        # Call the test method (safetensors module is mocked during import)
        self.worker.prepare_infer_params_to_cpu(weight_save_dir)
        
        # Verify calls
        # 1. Verify FSDP.state_dict_type context manager is used correctly
        self.assertTrue(mock_fsdp_module.FullyShardedDataParallel.state_dict_type.called)
        args, kwargs = mock_fsdp_module.FullyShardedDataParallel.state_dict_type.call_args
        self.assertEqual(args[0], self.worker.actor_module_fsdp)
        self.assertEqual(args[1], mock_fsdp_module.StateDictType.FULL_STATE_DICT)
        
        # 2. Verify state_dict is called
        self.worker.actor_module_fsdp.state_dict.assert_called_once()
        
        # 3. Verify os.makedirs is called
        self.mock_makedirs.assert_called_once_with(weight_save_dir, exist_ok=True)
        
        # 4. Verify save_file is called
        mock_safetensors.save_file.assert_called_once_with(
            mock_state_dict,
            os.path.join(weight_save_dir, "model.safetensors")
        )
        
        # 5. Verify ray.get_actor is called
        mock_ray.get_actor.assert_called_once_with("weight_updater", namespace="controller_raygroup")
        
        # 6. Verify weight_saved.remote method is called (Ray remote call)
        mock_weight_actor.weight_saved.remote.assert_called_once_with(weight_save_dir)
    
    def test_prepare_infer_params_to_cpu_non_rank0(self):
        # Set up environment
        weight_save_dir = '/tmp/test_weights'
        mock_state_dict = {'param1': MagicMock(), 'param2': MagicMock()}
        
        # Mock rank as non-zero
        mock_torch_distributed.get_rank.return_value = 1
        
        # Mock state_dict call
        self.worker.actor_module_fsdp.state_dict.return_value = mock_state_dict
        
        # Call the test method (safetensors module is mocked during import)
        self.worker.prepare_infer_params_to_cpu(weight_save_dir)
        
        # Verify calls
        # 1. Verify FSDP.state_dict_type context manager is used correctly
        self.assertTrue(mock_fsdp_module.FullyShardedDataParallel.state_dict_type.called)
        
        # 2. Verify state_dict is called
        self.worker.actor_module_fsdp.state_dict.assert_called_once()
        
        # 3. Verify os.makedirs is not called (non-zero rank)
        self.mock_makedirs.assert_not_called()
        
        # 4. Verify save_file is not called (non-zero rank)
        mock_safetensors.save_file.assert_not_called()
        
        # 5. Verify ray.get_actor is called
        mock_ray.get_actor.assert_called_once_with("weight_updater", namespace="controller_raygroup")
        
        # 6. Verify weight_saved.remote method is called (Ray remote call)
        weight_actor = mock_ray.get_actor.return_value
        weight_actor.weight_saved.remote.assert_called_once_with(weight_save_dir)
    
    def test_prepare_infer_params_to_cpu_state_dict_exception(self):
        # Set up environment
        weight_save_dir = '/tmp/test_weights'
        
        # Mock rank as 0
        mock_torch_distributed.get_rank.return_value = 0
        
        # Mock state_dict to raise exception
        error_msg = "State dict error"
        self.worker.actor_module_fsdp.state_dict.side_effect = Exception(error_msg)
        
        # Call the test method and verify exception (safetensors module is mocked during import)
        with self.assertRaises(Exception) as context:
            self.worker.prepare_infer_params_to_cpu(weight_save_dir)
        
        self.assertEqual(str(context.exception), error_msg)
        
        # Verify os.makedirs is not called (exception occurred)
        self.mock_makedirs.assert_not_called()
        
        # Verify save_file is not called (exception occurred)
        mock_safetensors.save_file.assert_not_called()
    
    def test_prepare_infer_params_to_cpu_save_exception(self):
        # Set up environment
        weight_save_dir = '/tmp/test_weights'
        mock_state_dict = {'param1': MagicMock(), 'param2': MagicMock()}
        
        # Mock rank as 0
        mock_torch_distributed.get_rank.return_value = 0
        
        # Mock state_dict call
        self.worker.actor_module_fsdp.state_dict.return_value = mock_state_dict
        
        # Mock save_file to raise exception
        error_msg = "Save error"
        mock_safetensors.save_file.side_effect = Exception(error_msg)
        
        # Call the test method and verify exception (safetensors module is mocked during import)
        with self.assertRaises(Exception) as context:
            self.worker.prepare_infer_params_to_cpu(weight_save_dir)
        
        self.assertEqual(str(context.exception), error_msg)
        
        # Verify os.makedirs is called
        self.mock_makedirs.assert_called_once_with(weight_save_dir, exist_ok=True)

if __name__ == '__main__':
    unittest.main()