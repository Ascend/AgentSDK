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
from unittest.mock import MagicMock, patch, call


class TestPatchCamem(unittest.TestCase):
    """Test patch_camem.py module"""

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
        """Setup mock objects for torch, acl, vllm, and vllm_ascend"""
        cls.mock_torch = MagicMock()
        cls.mock_torch.empty = MagicMock(return_value="cpu_backup_tensor")

        cls.mock_acl = MagicMock()
        cls.mock_acl_rt = MagicMock()
        cls.mock_acl.rt = cls.mock_acl_rt
        cls.mock_acl_rt.memcpy = MagicMock()

        cls.mock_vllm = MagicMock()
        cls.mock_vllm_utils = MagicMock()
        cls.mock_vllm.utils = cls.mock_vllm_utils
        cls.mock_vllm_utils.is_pin_memory_available = MagicMock(return_value=True)

        cls.mock_vllm_logger = MagicMock()
        cls.mock_vllm.logger = cls.mock_vllm_logger

        cls.mock_vllm_ascend = MagicMock()
        cls.mock_camem = MagicMock()
        cls.mock_vllm_ascend.device_allocator = MagicMock()
        cls.mock_vllm_ascend.device_allocator.camem = cls.mock_camem

        class MockCaMemAllocator:
            def __init__(self):
                self.pointer_to_data = {}

        cls.mock_camem.CaMemAllocator = MockCaMemAllocator
        cls.mock_camem.CaMemAllocator.default_tag = "default_tag"
        cls.mock_camem.unmap_and_release = MagicMock()

        cls.modules_patcher = patch.dict('sys.modules', {
            'torch': cls.mock_torch,
            'acl': cls.mock_acl,
            'acl.rt': cls.mock_acl_rt,
            'vllm': cls.mock_vllm,
            'vllm.logger': cls.mock_vllm_logger,
            'vllm.utils': cls.mock_vllm_utils,
            'vllm_ascend': cls.mock_vllm_ascend,
            'vllm_ascend.device_allocator': cls.mock_vllm_ascend.device_allocator,
            'vllm_ascend.device_allocator.camem': cls.mock_camem,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        spec = importlib.util.spec_from_file_location(
            'patch_camem',
            os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2',
                         'patch_camem.py')
        )
        cls.patch_camem = importlib.util.module_from_spec(spec)
        sys.modules['patch_camem'] = cls.patch_camem
        spec.loader.exec_module(cls.patch_camem)

    @classmethod
    def _cleanup_mocks(cls):
        """Clean up mock patches"""
        cls.modules_patcher.stop()

    def setUp(self):
        """Set up test environment"""
        self.mock_camem.unmap_and_release.reset_mock()
        self.mock_vllm_utils.is_pin_memory_available.reset_mock(return_value=True)
        self.mock_acl_rt.memcpy.reset_mock()
        self.mock_torch.empty.reset_mock()

        self.mock_allocator = self.mock_camem.CaMemAllocator()

        self.mock_data1 = MagicMock()
        self.mock_data1.handle = (0x1234, 1024)
        self.mock_data1.tag = self.mock_camem.CaMemAllocator.default_tag

        self.mock_data2 = MagicMock()
        self.mock_data2.handle = (0x5678, 2048)
        self.mock_data2.tag = "other_tag"

        self.mock_allocator.pointer_to_data = {
            0x1234: self.mock_data1,
            0x5678: self.mock_data2
        }

        self.mock_cpu_tensor = MagicMock()
        self.mock_cpu_tensor.data_ptr.return_value = 0x9012

    def test_camem_sleep_with_default_tags(self):
        """Test camem_sleep with default tags"""
        self.mock_torch.empty.return_value = self.mock_cpu_tensor

        self.patch_camem.camem_sleep(self.mock_allocator)

        self.assertEqual(self.mock_torch.empty.call_count, 1)

        self.mock_acl_rt.memcpy.assert_called_once()

        self.mock_camem.unmap_and_release.assert_called_once_with(self.mock_data1.handle)

    def test_camem_sleep_with_specific_tag(self):
        """Test camem_sleep with a specific tag"""
        self.mock_torch.empty.return_value = self.mock_cpu_tensor

        self.patch_camem.camem_sleep(self.mock_allocator, offload_tags="other_tag")

        self.assertEqual(self.mock_torch.empty.call_count, 1)

        self.mock_acl_rt.memcpy.assert_called_once()

        self.mock_camem.unmap_and_release.assert_called_once_with(self.mock_data2.handle)

    def test_camem_sleep_with_multiple_tags(self):
        """Test camem_sleep with multiple tags"""
        self.mock_torch.empty.return_value = self.mock_cpu_tensor

        self.patch_camem.camem_sleep(self.mock_allocator,
                                     offload_tags=(self.mock_camem.CaMemAllocator.default_tag, "other_tag"))

        self.assertEqual(self.mock_data1.cpu_backup_tensor, self.mock_cpu_tensor)
        self.assertEqual(self.mock_data2.cpu_backup_tensor, self.mock_cpu_tensor)

        self.assertEqual(self.mock_acl_rt.memcpy.call_count, 2)

        self.assertEqual(self.mock_camem.unmap_and_release.call_count, 2)
        self.mock_camem.unmap_and_release.assert_has_calls([
            call(self.mock_data1.handle),
            call(self.mock_data2.handle)
        ])

    def test_camem_sleep_with_pin_memory_unavailable(self):
        """Test camem_sleep when pin_memory is not available"""
        self.mock_vllm_utils.is_pin_memory_available.return_value = False

        self.mock_torch.empty.return_value = self.mock_cpu_tensor

        self.patch_camem.camem_sleep(self.mock_allocator)

        self.mock_torch.empty.assert_called_once()
        call_args = self.mock_torch.empty.call_args
        self.assertFalse(call_args[1]['pin_memory'])

    def test_camem_sleep_with_empty_pointer_data(self):
        """Test camem_sleep with empty pointer_to_data"""
        self.mock_allocator.pointer_to_data = {}

        self.patch_camem.camem_sleep(self.mock_allocator)

        self.mock_torch.empty.assert_not_called()
        self.mock_acl_rt.memcpy.assert_not_called()
        self.mock_camem.unmap_and_release.assert_not_called()


if __name__ == '__main__':
    unittest.main()
