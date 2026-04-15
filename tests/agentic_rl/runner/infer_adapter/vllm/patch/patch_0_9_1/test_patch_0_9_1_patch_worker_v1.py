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
    """Test patch_worker_v1.py"""

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
        """Setup mock objects for vllm_ascend and vllm"""
        cls.mock_vllm_ascend = MagicMock()
        cls.mock_vllm_ascend.device_allocator = MagicMock()
        cls.mock_vllm_ascend.device_allocator.camem = MagicMock()
        cls.mock_vllm_ascend.platform = MagicMock()
        cls.mock_vllm_ascend.worker = MagicMock()
        cls.mock_vllm_ascend.worker.worker_v1 = MagicMock()

        cls.mock_vllm = MagicMock()
        cls.mock_vllm.logger = MagicMock()
        cls.mock_vllm.utils = MagicMock()

        cls.vllm_ascend_patcher = patch.dict('sys.modules', {'vllm_ascend': cls.mock_vllm_ascend})
        cls.vllm_ascend_device_allocator_patcher = patch.dict('sys.modules', {
            'vllm_ascend.device_allocator': cls.mock_vllm_ascend.device_allocator})
        cls.vllm_ascend_camem_patcher = patch.dict('sys.modules', {
            'vllm_ascend.device_allocator.camem': cls.mock_vllm_ascend.device_allocator.camem})
        cls.vllm_ascend_platform_patcher = patch.dict('sys.modules',
                                                      {'vllm_ascend.platform': cls.mock_vllm_ascend.platform})
        cls.vllm_ascend_worker_patcher = patch.dict('sys.modules', {'vllm_ascend.worker': cls.mock_vllm_ascend.worker})
        cls.vllm_ascend_worker_v1_patcher = patch.dict('sys.modules', {
            'vllm_ascend.worker.worker_v1': cls.mock_vllm_ascend.worker.worker_v1})
        cls.vllm_patcher = patch.dict('sys.modules', {'vllm': cls.mock_vllm})
        cls.vllm_logger_patcher = patch.dict('sys.modules', {'vllm.logger': cls.mock_vllm.logger})
        cls.vllm_utils_patcher = patch.dict('sys.modules', {'vllm.utils': cls.mock_vllm.utils})

        cls.vllm_ascend_patcher.start()
        cls.vllm_ascend_device_allocator_patcher.start()
        cls.vllm_ascend_camem_patcher.start()
        cls.vllm_ascend_platform_patcher.start()
        cls.vllm_ascend_worker_patcher.start()
        cls.vllm_ascend_worker_v1_patcher.start()
        cls.vllm_patcher.start()
        cls.vllm_logger_patcher.start()
        cls.vllm_utils_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        patch_worker_v1_path = os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch',
                                            'patch_0_9_1', 'patch_worker_v1.py')
        spec = importlib.util.spec_from_file_location('patch_worker_v1', patch_worker_v1_path)
        cls.patch_worker_v1 = importlib.util.module_from_spec(spec)
        sys.modules['patch_worker_v1'] = cls.patch_worker_v1
        spec.loader.exec_module(cls.patch_worker_v1)

    @classmethod
    def _cleanup_mocks(cls):
        """Clean up mock patches"""
        cls.vllm_ascend_patcher.stop()
        cls.vllm_ascend_device_allocator_patcher.stop()
        cls.vllm_ascend_camem_patcher.stop()
        cls.vllm_ascend_platform_patcher.stop()
        cls.vllm_ascend_worker_patcher.stop()
        cls.vllm_ascend_worker_v1_patcher.stop()
        cls.vllm_patcher.stop()
        cls.vllm_logger_patcher.stop()
        cls.vllm_utils_patcher.stop()

    def setUp(self):
        """Set up test environment"""
        self.mock_npu_worker = MagicMock()
        self.mock_camem_allocator = MagicMock()
        self.mock_npu_platform = MagicMock()
        self.mock_logger = MagicMock()

        self.patch_worker_v1.NPUPlatform = self.mock_npu_platform
        self.patch_worker_v1.CaMemAllocator = self.mock_camem_allocator
        self.patch_worker_v1.logger = self.mock_logger
        self.patch_worker_v1.NPUWorker = self.mock_npu_worker

        self.test_worker = MagicMock()
        self.test_worker.device = 0

    def test_multi_level_sleep_level_1(self):
        """Test sleep mode with level=1"""
        self.mock_npu_platform.reset_mock()
        self.mock_camem_allocator.reset_mock()
        self.mock_logger.reset_mock()

        mock_allocator_instance = MagicMock()
        self.mock_camem_allocator.get_instance.return_value = mock_allocator_instance

        self.mock_npu_platform.mem_get_info.side_effect = [(10 * 1024 ** 3, 20 * 1024 ** 3),
                                                           (15 * 1024 ** 3, 20 * 1024 ** 3)]

        test_worker = MagicMock()
        test_worker.device = 0

        self.patch_worker_v1.GiB_bytes = 1024 ** 3

        self.patch_worker_v1.multi_level_sleep(test_worker, level=1)

        self.mock_npu_platform.set_device.assert_called_once_with(0)
        self.mock_camem_allocator.get_instance.assert_called_once()
        mock_allocator_instance.sleep.assert_called_once_with(offload_tags=("weights",))

        self.assertEqual(self.mock_npu_platform.mem_get_info.call_count, 2)

        self.mock_logger.info.assert_called_once_with(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", 5.0, 5.0
        )

    def test_multi_level_sleep_level_2(self):
        """Test sleep mode with level=2"""
        self.mock_npu_platform.reset_mock()
        self.mock_camem_allocator.reset_mock()
        self.mock_logger.reset_mock()

        mock_allocator_instance = MagicMock()
        self.mock_camem_allocator.get_instance.return_value = mock_allocator_instance

        self.mock_npu_platform.mem_get_info.side_effect = [(10 * 1024 ** 3, 20 * 1024 ** 3),
                                                           (15 * 1024 ** 3, 20 * 1024 ** 3)]

        test_worker = MagicMock()
        test_worker.device = 0

        self.patch_worker_v1.multi_level_sleep(test_worker, level=2)

        mock_allocator_instance.sleep.assert_called_once_with(offload_tags=("kv_cache",))

        self.mock_logger.info.assert_called_once()

    def test_multi_level_sleep_default_level(self):
        """Test sleep mode with default level"""
        self.mock_npu_platform.reset_mock()
        self.mock_camem_allocator.reset_mock()
        self.mock_logger.reset_mock()

        mock_allocator_instance = MagicMock()
        self.mock_camem_allocator.get_instance.return_value = mock_allocator_instance

        self.mock_npu_platform.mem_get_info.side_effect = [(10 * 1024 ** 3, 20 * 1024 ** 3),
                                                           (15 * 1024 ** 3, 20 * 1024 ** 3)]

        test_worker = MagicMock()
        test_worker.device = 0

        self.patch_worker_v1.multi_level_sleep(test_worker)

        mock_allocator_instance.sleep.assert_called_once_with(offload_tags=("weights",))

    def test_multi_level_sleep_invalid_level(self):
        """Test sleep mode with an invalid level"""
        self.mock_npu_platform.reset_mock()
        self.mock_camem_allocator.reset_mock()
        self.mock_logger.reset_mock()

        mock_allocator_instance = MagicMock()
        self.mock_camem_allocator.get_instance.return_value = mock_allocator_instance

        self.mock_npu_platform.mem_get_info.side_effect = [(10 * 1024 ** 3, 20 * 1024 ** 3),
                                                           (15 * 1024 ** 3, 20 * 1024 ** 3)]

        test_worker = MagicMock()
        test_worker.device = 0

        self.patch_worker_v1.multi_level_sleep(test_worker, level=3)

        mock_allocator_instance.sleep.assert_called_once_with(offload_tags=tuple())

    def test_multi_level_sleep_memory_increase(self):
        """Test case where memory usage increases after sleeping (should raise assertion error)"""
        self.mock_npu_platform.reset_mock()
        self.mock_camem_allocator.reset_mock()
        self.mock_logger.reset_mock()

        mock_allocator_instance = MagicMock()
        self.mock_camem_allocator.get_instance.return_value = mock_allocator_instance

        self.mock_npu_platform.mem_get_info.side_effect = [(10 * 1024 ** 3, 20 * 1024 ** 3),
                                                           (5 * 1024 ** 3, 20 * 1024 ** 3)]

        test_worker = MagicMock()
        test_worker.device = 0

        with self.assertRaises(RuntimeError) as context:
            self.patch_worker_v1.multi_level_sleep(test_worker)

        self.assertIn("Memory usage increased after sleeping.", str(context.exception))

    def test_npu_worker_patch(self):
        """Test that the module attempts to patch NPUWorker.sleep to multi_level_sleep"""
        self.assertTrue(hasattr(self.patch_worker_v1, 'NPUWorker'))
        self.assertTrue(hasattr(self.patch_worker_v1, 'multi_level_sleep'))
        self.assertTrue(callable(self.patch_worker_v1.multi_level_sleep))


if __name__ == '__main__':
    unittest.main()
