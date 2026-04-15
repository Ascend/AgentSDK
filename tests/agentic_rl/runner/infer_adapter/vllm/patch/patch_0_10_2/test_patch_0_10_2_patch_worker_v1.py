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

import importlib
import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock, patch


class TestPatchWorkerV1(unittest.TestCase):
    """Test patch_worker_v1.py module"""

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
        """Setup mock objects for vllm and vllm_ascend"""
        cls.mock_vllm = MagicMock()

        cls.mock_logger = MagicMock()
        cls.mock_logger.info = MagicMock()
        cls.mock_vllm_logger_module = MagicMock()
        cls.mock_vllm_logger_module.logger = cls.mock_logger

        cls.mock_utils = MagicMock()
        cls.mock_utils.STR_DTYPE_TO_TORCH_DTYPE = {}
        cls.mock_utils.GiB_bytes = 1024 * 1024 * 1024

        cls.mock_config = MagicMock()
        cls.mock_config.CUDAGraphMode = MagicMock()
        cls.mock_config.CUDAGraphMode.FULL_DECODE_ONLY = "full_decode_only"
        cls.mock_config.VllmConfig = MagicMock()

        cls.mock_vllm_ascend = MagicMock()

        cls.mock_device_allocator = MagicMock()

        cls.mock_camem = MagicMock()
        cls.mock_camem_allocator = MagicMock()
        cls.mock_camem.CaMemAllocator = cls.mock_camem_allocator
        cls.mock_camem_allocator.get_instance.return_value = cls.mock_camem_allocator
        cls.mock_camem_allocator.sleep = MagicMock()

        cls.mock_platform = MagicMock()
        cls.mock_platform.NPUPlatform = MagicMock()
        cls.mock_platform.NPUPlatform.set_device = MagicMock()
        cls.mock_platform.NPUPlatform.mem_get_info.return_value = (500 * 1024 * 1024, 1000 * 1024 * 1024)

        class MockNPUWorker:
            def __init__(self):
                self.device = 0
                self.model_runner = MagicMock()
                self.compilation_config = MagicMock()
                self.compilation_config.cudagraph_mode = "normal"

        cls.MockNPUWorker = MockNPUWorker

        cls.mock_worker = MagicMock()

        cls.mock_worker_v1 = MagicMock()
        cls.mock_worker_v1.NPUWorker = MockNPUWorker

        cls.modules_patcher = patch.dict('sys.modules', {
            'vllm': cls.mock_vllm,
            'vllm.logger': cls.mock_vllm_logger_module,
            'vllm.utils': cls.mock_utils,
            'vllm.config': cls.mock_config,
            'vllm_ascend': cls.mock_vllm_ascend,
            'vllm_ascend.device_allocator': cls.mock_device_allocator,
            'vllm_ascend.device_allocator.camem': cls.mock_camem,
            'vllm_ascend.platform': cls.mock_platform,
            'vllm_ascend.worker': cls.mock_worker,
            'vllm_ascend.worker.worker_v1': cls.mock_worker_v1,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        spec = importlib.util.spec_from_file_location(
            'patch_worker_v1',
            os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2',
                         'patch_worker_v1.py')
        )
        cls.patch_worker_v1 = importlib.util.module_from_spec(spec)
        sys.modules['patch_worker_v1'] = cls.patch_worker_v1
        spec.loader.exec_module(cls.patch_worker_v1)

    @classmethod
    def _cleanup_mocks(cls):
        """Clean up mock patches"""
        cls.modules_patcher.stop()

    def setUp(self):
        """Set up test environment"""
        self.mock_platform.NPUPlatform.set_device.reset_mock()
        self.mock_platform.NPUPlatform.mem_get_info.reset_mock()
        self.mock_camem_allocator.sleep.reset_mock()
        self.mock_logger.info.reset_mock()

        self.worker_instance = self.MockNPUWorker()

    def test_multi_level_sleep_level_1(self):
        """Test multi_level_sleep function with level 1"""
        self.worker_instance.sleep(level=1)

        self.mock_platform.NPUPlatform.set_device.assert_called_once_with(0)
        self.mock_platform.NPUPlatform.mem_get_info.assert_called()
        self.mock_camem_allocator.sleep.assert_called_once_with(offload_tags=("weights",))
        self.mock_logger.info.assert_called_once()

    def test_multi_level_sleep_level_2(self):
        """Test multi_level_sleep function with level 2"""
        self.worker_instance.sleep(level=2)

        self.mock_platform.NPUPlatform.set_device.assert_called_once_with(0)
        self.mock_platform.NPUPlatform.mem_get_info.assert_called()
        self.mock_camem_allocator.sleep.assert_called_once_with(offload_tags=("kv_cache",))
        self.mock_logger.info.assert_called_once()

    def test_multi_level_sleep_level_3(self):
        """Test multi_level_sleep function with level 3"""
        self.worker_instance.sleep(level=3)

        self.mock_platform.NPUPlatform.set_device.assert_called_once_with(0)
        self.mock_platform.NPUPlatform.mem_get_info.assert_called()
        self.mock_camem_allocator.sleep.assert_called_once_with(offload_tags=tuple())
        self.mock_logger.info.assert_called_once()

    def test_multi_level_sleep_default_level(self):
        """Test multi_level_sleep function with default level"""
        self.worker_instance.sleep()

        self.mock_platform.NPUPlatform.set_device.assert_called_once_with(0)
        self.mock_platform.NPUPlatform.mem_get_info.assert_called()
        self.mock_camem_allocator.sleep.assert_called_once_with(offload_tags=("weights",))
        self.mock_logger.info.assert_called_once()

    def test_worker_execute_dummy_batch_normal_mode(self):
        """Test worker_execute_dummy_batch function with normal cudagraph mode"""
        self.worker_instance.execute_dummy_batch()

        self.worker_instance.model_runner._dummy_run.assert_called_once_with(
            num_tokens=1,
            uniform_decode=True,
            force_attention=False
        )

    def test_worker_execute_dummy_batch_full_decode_mode(self):
        """Test worker_execute_dummy_batch function with FULL_DECODE_ONLY cudagraph mode"""
        self.worker_instance.compilation_config.cudagraph_mode = self.mock_config.CUDAGraphMode.FULL_DECODE_ONLY

        self.worker_instance.execute_dummy_batch()

        self.worker_instance.model_runner._dummy_run.assert_called_once_with(
            num_tokens=1,
            uniform_decode=True,
            force_attention=True
        )

    def test_multi_level_sleep_memory_increase_assertion(self):
        """Test that multi_level_sleep asserts when memory usage increases"""
        self.mock_platform.NPUPlatform.mem_get_info.side_effect = [
            (500 * 1024 * 1024, 1000 * 1024 * 1024),
            (400 * 1024 * 1024, 1000 * 1024 * 1024)
        ]

        with self.assertRaises(RuntimeError):
            self.worker_instance.sleep(level=1)


if __name__ == '__main__':
    unittest.main()
