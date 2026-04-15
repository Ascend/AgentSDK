#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import unittest
from unittest.mock import MagicMock
import sys

class TestPatchScheduleConfig(unittest.TestCase):
    """Test patch_schedule_config.py module"""
    
    def setUp(self):
        """Set up test environment with comprehensive mocks"""
        # Create a clean state for each test
        self.original_modules = {}
        
        # Save original modules to restore later
        for module_name in ['vllm_ascend', 'vllm_ascend.core.schedule_config']:
            if module_name in sys.modules:
                self.original_modules[module_name] = sys.modules[module_name]
        
        # Mock vllm_ascend and its submodules
        self.mock_vllm_ascend = MagicMock()
        sys.modules['vllm_ascend'] = self.mock_vllm_ascend
        
        # Create a mock AscendSchedulerConfig class
        class MockAscendSchedulerConfig:
            def __init__(self, max_num_batched_tokens=1024, max_model_len=2048, 
                        enable_chunked_prefill=False, policy="fcfs",
                        is_multimodal_model=False, num_scheduler_steps=1,
                        send_delta_data=False, delay_factor=0):
                self.max_num_batched_tokens = max_num_batched_tokens
                self.max_model_len = max_model_len
                self.enable_chunked_prefill = enable_chunked_prefill
                self.policy = policy
                self.is_multimodal_model = is_multimodal_model
                self.num_scheduler_steps = num_scheduler_steps
                self.send_delta_data = send_delta_data
                self.delay_factor = delay_factor
                
                # These attributes will be set by __post_init__
                self.max_num_encoder_input_tokens = None
                self.encoder_cache_size = None
                self.chunked_prefill_enabled = None
                
                # Manually call __post_init__ since this isn't a dataclass
                self.__post_init__()
            
            def __post_init__(self):
                self.original_post_init_called = True
        
        # Mock vllm_ascend.core.schedule_config
        self.mock_schedule_config = MagicMock()
        self.mock_schedule_config.AscendSchedulerConfig = MockAscendSchedulerConfig
        sys.modules['vllm_ascend.core.schedule_config'] = self.mock_schedule_config
        
        # Now import the patch module
        import agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_schedule_config as patch_module
        self.patch_module = patch_module
    
    def tearDown(self):
        """Clean up after each test by restoring original modules"""
        # Remove the patch module from sys.modules to ensure fresh import for each test
        if 'agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_schedule_config' in sys.modules:
            del sys.modules['agentic_rl.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_schedule_config']
        
        # Restore original modules
        for module_name, module in self.original_modules.items():
            sys.modules[module_name] = module
        
        # Remove any other mocked modules
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('vllm_ascend'):
                if module_name not in self.original_modules:
                    del sys.modules[module_name]
    
    def test_ascend_scheduler_config_post_init_success(self):
        """Test ascend_scheduler_config_post_init function with valid configuration"""
        # Create an instance with valid configuration
        config = self.mock_schedule_config.AscendSchedulerConfig(
            max_num_batched_tokens=1024,
            max_model_len=2048,
            enable_chunked_prefill=False,
            policy="fcfs",
            is_multimodal_model=False,
            num_scheduler_steps=1,
            send_delta_data=False,
            delay_factor=0
        )
        
        # Verify the attributes were set correctly
        self.assertEqual(config.max_num_encoder_input_tokens, 1024)
        self.assertEqual(config.encoder_cache_size, 1024)
        self.assertEqual(config.chunked_prefill_enabled, False)
    
    def test_ascend_scheduler_config_post_init_non_fcfs_policy(self):
        """Test ascend_scheduler_config_post_init function with non-fcfs policy"""
        # Attempt to create an instance with non-fcfs policy
        with self.assertRaises(NotImplementedError) as context:
            self.mock_schedule_config.AscendSchedulerConfig(
                max_num_batched_tokens=1024,
                policy="priority"
            )
        
        # Verify the error message
        self.assertIn("currently AscendScheduler only supports fcfs policy", str(context.exception))
    
    def test_ascend_scheduler_config_post_init_multimodal_model(self):
        """Test ascend_scheduler_config_post_init function with multimodal model"""
        # Attempt to create an instance with multimodal model
        with self.assertRaises(NotImplementedError) as context:
            self.mock_schedule_config.AscendSchedulerConfig(
                max_num_batched_tokens=1024,
                is_multimodal_model=True
            )
        
        # Verify the error message
        self.assertIn("currently AscendScheduler only supports LLM models", str(context.exception))
    
    def test_ascend_scheduler_config_post_init_multi_step(self):
        """Test ascend_scheduler_config_post_init function with num_scheduler_steps > 1"""
        # Attempt to create an instance with num_scheduler_steps > 1
        with self.assertRaises(NotImplementedError) as context:
            self.mock_schedule_config.AscendSchedulerConfig(
                max_num_batched_tokens=1024,
                num_scheduler_steps=2
            )
        
        # Verify the error message
        self.assertIn("currently AscendScheduler doesn't support multi-step", str(context.exception))
    
    def test_ascend_scheduler_config_post_init_send_delta_data(self):
        """Test ascend_scheduler_config_post_init function with send_delta_data=True"""
        # Attempt to create an instance with send_delta_data=True
        with self.assertRaises(NotImplementedError) as context:
            self.mock_schedule_config.AscendSchedulerConfig(
                max_num_batched_tokens=1024,
                send_delta_data=True
            )
        
        # Verify the error message
        self.assertIn("currently AscendScheduler doesn't support send_delta_data", str(context.exception))
    
    def test_ascend_scheduler_config_post_init_delay_factor(self):
        """Test ascend_scheduler_config_post_init function with delay_factor > 0"""
        # Attempt to create an instance with delay_factor > 0
        with self.assertRaises(NotImplementedError) as context:
            self.mock_schedule_config.AscendSchedulerConfig(
                max_num_batched_tokens=1024,
                delay_factor=1.5
            )
        
        # Verify the error message
        self.assertIn("currently AscendScheduler doesn't support scheduler_delay_factor", str(context.exception))
    
    def test_ascend_scheduler_config_post_init_chunked_prefill(self):
        """Test ascend_scheduler_config_post_init function with enable_chunked_prefill=True"""
        # Create an instance with enable_chunked_prefill=True
        config = self.mock_schedule_config.AscendSchedulerConfig(
            max_num_batched_tokens=1024,
            enable_chunked_prefill=True
        )
        
        # Verify chunked_prefill_enabled is set correctly
        self.assertEqual(config.chunked_prefill_enabled, True)
    
    def test_ascend_scheduler_config_post_init_different_batch_size(self):
        """Test ascend_scheduler_config_post_init function with different batch size"""
        # Create an instance with different batch size
        config = self.mock_schedule_config.AscendSchedulerConfig(
            max_num_batched_tokens=2048
        )
        
        # Verify the attributes were set correctly
        self.assertEqual(config.max_num_encoder_input_tokens, 2048)
        self.assertEqual(config.encoder_cache_size, 2048)


if __name__ == '__main__':
    unittest.main()
    