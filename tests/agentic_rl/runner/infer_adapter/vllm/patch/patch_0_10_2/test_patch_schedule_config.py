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
from unittest.mock import MagicMock, patch
import sys
import os
import importlib
import importlib.util


class TestPatchScheduleConfig(unittest.TestCase):
    """Test patch_schedule_config.py module"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment for the entire test class"""
        cls._setup_mocks()
        cls._import_module_under_test()

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment for the entire test class"""
        cls._cleanup_mocks()

    @classmethod
    def _setup_mocks(cls):
        """Setup mock objects for vllm_ascend"""
        cls.mock_vllm_ascend = MagicMock()

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

                self.max_num_encoder_input_tokens = None
                self.encoder_cache_size = None
                self.chunked_prefill_enabled = None

                self.__post_init__()

            def __post_init__(self):
                self.original_post_init_called = True

        cls.mock_schedule_config = MagicMock()
        cls.mock_schedule_config.AscendSchedulerConfig = MockAscendSchedulerConfig

        cls.modules_patcher = patch.dict('sys.modules', {
            'vllm_ascend': cls.mock_vllm_ascend,
            'vllm_ascend.core': MagicMock(),
            'vllm_ascend.core.schedule_config': cls.mock_schedule_config,
            'vllm_ascend.patch': MagicMock(),
            'vllm_ascend.patch.platform': MagicMock(),
            'vllm_ascend.patch.worker': MagicMock(),
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        spec = importlib.util.spec_from_file_location(
            'patch_schedule_config',
            os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2',
                         'patch_schedule_config.py')
        )
        cls.patch_schedule_config = importlib.util.module_from_spec(spec)
        sys.modules['patch_schedule_config'] = cls.patch_schedule_config
        spec.loader.exec_module(cls.patch_schedule_config)

    @classmethod
    def _cleanup_mocks(cls):
        """Clean up mock patches"""
        cls.modules_patcher.stop()

    def test_ascend_scheduler_config_post_init_success(self):
        """Test ascend_scheduler_config_post_init function with valid configuration"""
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

        self.assertEqual(config.max_num_encoder_input_tokens, 1024)
        self.assertEqual(config.encoder_cache_size, 1024)
        self.assertEqual(config.chunked_prefill_enabled, False)

    def test_ascend_scheduler_config_post_init_non_fcfs_policy(self):
        """Test ascend_scheduler_config_post_init function with non-fcfs policy"""
        with self.assertRaises(NotImplementedError) as context:
            self.mock_schedule_config.AscendSchedulerConfig(
                max_num_batched_tokens=1024,
                policy="priority"
            )

        self.assertIn("currently AscendScheduler only supports fcfs policy", str(context.exception))

    def test_ascend_scheduler_config_post_init_multimodal_model(self):
        """Test ascend_scheduler_config_post_init function with multimodal model"""
        with self.assertRaises(NotImplementedError) as context:
            self.mock_schedule_config.AscendSchedulerConfig(
                max_num_batched_tokens=1024,
                is_multimodal_model=True
            )

        self.assertIn("currently AscendScheduler only supports LLM models", str(context.exception))

    def test_ascend_scheduler_config_post_init_multi_step(self):
        """Test ascend_scheduler_config_post_init function with num_scheduler_steps > 1"""
        with self.assertRaises(NotImplementedError) as context:
            self.mock_schedule_config.AscendSchedulerConfig(
                max_num_batched_tokens=1024,
                num_scheduler_steps=2
            )

        self.assertIn("currently AscendScheduler doesn't support multi-step", str(context.exception))

    def test_ascend_scheduler_config_post_init_send_delta_data(self):
        """Test ascend_scheduler_config_post_init function with send_delta_data=True"""
        with self.assertRaises(NotImplementedError) as context:
            self.mock_schedule_config.AscendSchedulerConfig(
                max_num_batched_tokens=1024,
                send_delta_data=True
            )

        self.assertIn("currently AscendScheduler doesn't support send_delta_data", str(context.exception))

    def test_ascend_scheduler_config_post_init_delay_factor(self):
        """Test ascend_scheduler_config_post_init function with delay_factor > 0"""
        with self.assertRaises(NotImplementedError) as context:
            self.mock_schedule_config.AscendSchedulerConfig(
                max_num_batched_tokens=1024,
                delay_factor=1.5
            )

        self.assertIn("currently AscendScheduler doesn't support scheduler_delay_factor", str(context.exception))

    def test_ascend_scheduler_config_post_init_chunked_prefill(self):
        """Test ascend_scheduler_config_post_init function with enable_chunked_prefill=True"""
        config = self.mock_schedule_config.AscendSchedulerConfig(
            max_num_batched_tokens=1024,
            enable_chunked_prefill=True
        )

        self.assertEqual(config.chunked_prefill_enabled, True)

    def test_ascend_scheduler_config_post_init_different_batch_size(self):
        """Test ascend_scheduler_config_post_init function with different batch size"""
        config = self.mock_schedule_config.AscendSchedulerConfig(
            max_num_batched_tokens=2048
        )

        self.assertEqual(config.max_num_encoder_input_tokens, 2048)
        self.assertEqual(config.encoder_cache_size, 2048)


if __name__ == '__main__':
    unittest.main()
