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
import sys
import unittest
from unittest.mock import patch, MagicMock


# Create mock objects
mock_ray = MagicMock()
mock_collective = MagicMock()
mock_get_nccl_backend = MagicMock()

# Create mock for ray.remote decorator
mock_ray.remote = MagicMock(side_effect=lambda cls: cls)  # Directly return the original class without remote processing

# Mock required dependency modules before importing the module under test
with patch.dict('sys.modules', {
    'ray': mock_ray,
    'ray.util.collective': MagicMock(collective=mock_collective),
    'verl.utils.device': MagicMock(get_nccl_backend=mock_get_nccl_backend)
}):
    # Now we can safely import the class under test
    from agentic_rl.trainer.train_adapter.verl.full_async.param_sync import ParameterSynchronizer

class TestParameterSynchronizer(unittest.TestCase):
    def setUp(self):
        # Create test instance
        self.synchronizer = ParameterSynchronizer()
    
    def test_init(self):
        # Verify initialization
        self.assertIsInstance(self.synchronizer, ParameterSynchronizer)
    
    def test_get_current_param_version(self):
        # Call the method
        result = self.synchronizer.get_current_param_version()
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)
    
    def test_get_weights_info(self):
        # Call the method
        result = self.synchronizer.get_weights_info()
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)
    
    def test_sync_weights(self):
        # Prepare test data
        version = 1
        validate = False
        global_steps = 100
        
        # Call the method
        result = self.synchronizer.sync_weights(version, validate, global_steps)
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)
    
    def test_wait_last_valid(self):
        # Call the method
        result = self.synchronizer.wait_last_valid()
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)
    
    def test_rollouter_save_checkpoint(self):
        # Prepare test data
        local_global_step_folder = '/tmp/test_ckpt/global_step_100'
        
        # Call the method
        result = self.synchronizer.rollouter_save_checkpoint(local_global_step_folder)
        
        # Since the original method has no return value, verify result is None
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()