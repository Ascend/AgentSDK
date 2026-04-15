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

import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock, patch


class TestPatchCaMem(unittest.TestCase):
    """Test the patch_camem.py module"""

    @classmethod
    def setUpClass(cls):
        cls._setup_mocks()
        cls._import_module_under_test()

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_mocks()

    @classmethod
    def _setup_mocks(cls):
        cls.mock_torch = MagicMock()
        cls.mock_torch.uint8 = "uint8"

        cls.mock_acl = MagicMock()
        cls.mock_acl_rt = MagicMock()
        cls.mock_acl.rt = cls.mock_acl_rt
        cls.mock_acl_rt.memcpy = MagicMock()

        cls.mock_vllm = MagicMock()
        cls.mock_vllm_logger = MagicMock()
        cls.mock_vllm.logger = cls.mock_vllm_logger
        cls.mock_vllm_logger.logger = MagicMock()

        cls.mock_vllm_utils = MagicMock()
        cls.mock_vllm.utils = cls.mock_vllm_utils
        cls.mock_vllm_utils.is_pin_memory_available.return_value = True

        cls.mock_vllm_ascend = MagicMock()
        cls.mock_vllm_ascend_camem = MagicMock()
        cls.mock_vllm_ascend.device_allocator = MagicMock()
        cls.mock_vllm_ascend.device_allocator.camem = cls.mock_vllm_ascend_camem

        cls.modules_patcher = patch.dict('sys.modules', {
            'torch': cls.mock_torch,
            'acl': cls.mock_acl,
            'acl.rt': cls.mock_acl_rt,
            'vllm': cls.mock_vllm,
            'vllm.logger': cls.mock_vllm_logger,
            'vllm.utils': cls.mock_vllm_utils,
            'vllm_ascend': cls.mock_vllm_ascend,
            'vllm_ascend.device_allocator': cls.mock_vllm_ascend.device_allocator,
            'vllm_ascend.device_allocator.camem': cls.mock_vllm_ascend_camem,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        module_file_path = os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch',
                                        'patch_0_9_1',
                                        'patch_camem.py')

        spec = importlib.util.spec_from_file_location('patch_camem', module_file_path)
        cls.patch_camem = importlib.util.module_from_spec(spec)
        sys.modules['patch_camem'] = cls.patch_camem

        spec.loader.exec_module(cls.patch_camem)

    @classmethod
    def _cleanup_mocks(cls):
        cls.modules_patcher.stop()

    def setUp(self):
        self.mock_allocator = MagicMock()
        self.mock_allocator.pointer_to_data = {}

        class MockData:
            def __init__(self, tag, handle):
                self.tag = tag
                self.handle = handle

        self.mock_data1 = MockData("weights", ("handle1", 1024))
        self.mock_data2 = MockData("kv_cache", ("handle2", 2048))
        self.mock_data3 = MockData("default", ("handle3", 512))

        self.mock_vllm_ascend_camem.CaMemAllocator.default_tag = "default"

        self.mock_allocator.pointer_to_data = {
            100: self.mock_data1,
            200: self.mock_data2,
            300: self.mock_data3
        }

    def test_camem_sleep_with_none_offload_tags(self):
        self.mock_vllm_utils.is_pin_memory_available.reset_mock()
        self.mock_torch.empty.reset_mock()
        self.mock_acl_rt.memcpy.reset_mock()
        self.mock_vllm_ascend_camem.unmap_and_release.reset_mock()

        self.patch_camem.camem_sleep(self.mock_allocator, offload_tags=None)

        self.assertFalse(hasattr(self.mock_data1, 'cpu_backup_tensor'))
        self.assertFalse(hasattr(self.mock_data2, 'cpu_backup_tensor'))
        self.assertTrue(hasattr(self.mock_data3, 'cpu_backup_tensor'))

        self.assertEqual(self.mock_vllm_ascend_camem.unmap_and_release.call_count, 1)

    def test_camem_sleep_with_string_offload_tags(self):
        self.mock_vllm_utils.is_pin_memory_available.reset_mock()
        self.mock_torch.empty.reset_mock()
        self.mock_acl_rt.memcpy.reset_mock()
        self.mock_vllm_ascend_camem.unmap_and_release.reset_mock()

        self.patch_camem.camem_sleep(self.mock_allocator, offload_tags="weights")

        self.assertTrue(hasattr(self.mock_data1, 'cpu_backup_tensor'))
        self.assertFalse(hasattr(self.mock_data2, 'cpu_backup_tensor'))
        self.assertFalse(hasattr(self.mock_data3, 'cpu_backup_tensor'))

        self.assertEqual(self.mock_vllm_ascend_camem.unmap_and_release.call_count, 1)

    def test_camem_sleep_with_tuple_offload_tags(self):
        self.mock_vllm_utils.is_pin_memory_available.reset_mock()
        self.mock_torch.empty.reset_mock()
        self.mock_acl_rt.memcpy.reset_mock()
        self.mock_vllm_ascend_camem.unmap_and_release.reset_mock()

        self.patch_camem.camem_sleep(self.mock_allocator, offload_tags=("weights", "kv_cache"))

        self.assertTrue(hasattr(self.mock_data1, 'cpu_backup_tensor'))
        self.assertTrue(hasattr(self.mock_data2, 'cpu_backup_tensor'))
        self.assertFalse(hasattr(self.mock_data3, 'cpu_backup_tensor'))

        self.assertEqual(self.mock_vllm_ascend_camem.unmap_and_release.call_count, 2)

    def test_camem_sleep_tensor_creation(self):
        self.mock_vllm_utils.is_pin_memory_available.reset_mock()
        self.mock_torch.empty.reset_mock()

        self.mock_vllm_utils.is_pin_memory_available.return_value = False

        self.patch_camem.camem_sleep(self.mock_allocator, offload_tags="weights")

        self.mock_torch.empty.assert_called_once_with(
            1024,
            dtype=self.mock_torch.uint8,
            device='cpu',
            pin_memory=False
        )

    def test_camem_sleep_memcpy_call(self):
        self.mock_torch.empty.reset_mock()
        self.mock_acl_rt.memcpy.reset_mock()

        mock_tensor = MagicMock()
        mock_tensor.data_ptr.return_value = 1000
        self.mock_torch.empty.return_value = mock_tensor

        self.patch_camem.camem_sleep(self.mock_allocator, offload_tags="weights")

        self.mock_acl_rt.memcpy.assert_called_once_with(
            1000,
            1000 + 1024 * 2,
            100,
            1024,
            2
        )

    def test_camem_sleep_unmap_and_release(self):
        self.mock_vllm_ascend_camem.unmap_and_release.reset_mock()

        self.patch_camem.camem_sleep(self.mock_allocator, offload_tags="weights")

        self.mock_vllm_ascend_camem.unmap_and_release.assert_called_once_with(
            ("handle1", 1024)
        )


if __name__ == '__main__':
    unittest.main()
