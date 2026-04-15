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
from unittest.mock import patch, MagicMock
import sys
import os
import importlib
import importlib.util


class TestPatchAttentionMask(unittest.TestCase):
    """Test patch_attention_mask.py module"""

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
        """Setup mock objects for torch and vllm_ascend"""
        cls.mock_torch = MagicMock()
        cls.mock_torch.float16 = "float16"
        cls.mock_torch.float32 = "float32"

        cls.mock_vllm_ascend = MagicMock()
        cls.mock_vllm_ascend.attention = MagicMock()
        cls.mock_vllm_ascend_attention_mask = MagicMock()
        cls.mock_vllm_ascend.attention.attention_mask = cls.mock_vllm_ascend_attention_mask

        cls.mock_vllm_ascend_attention_mask.AttentionMaskBuilder = MagicMock()
        cls.mock_vllm_ascend_attention_mask.AttentionMaskBuilder.get_mask_scale_factor = MagicMock(return_value=1.0)

        cls.modules_patcher = patch.dict('sys.modules', {
            'torch': cls.mock_torch,
            'vllm_ascend': cls.mock_vllm_ascend,
            'vllm_ascend.attention': cls.mock_vllm_ascend.attention,
            'vllm_ascend.attention.attention_mask': cls.mock_vllm_ascend_attention_mask,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        spec = importlib.util.spec_from_file_location(
            'patch_attention_mask',
            os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2', 'patch_attention_mask.py')
        )
        cls.patch_attention_mask = importlib.util.module_from_spec(spec)
        sys.modules['patch_attention_mask'] = cls.patch_attention_mask
        spec.loader.exec_module(cls.patch_attention_mask)

    @classmethod
    def _cleanup_mocks(cls):
        """Clean up mock patches"""
        cls.modules_patcher.stop()

    def setUp(self):
        """Set up test environment"""
        self.mock_self = MagicMock()
        self.mock_self.attn_mask_cache = MagicMock()
        self.mock_self._update_attn_cache = MagicMock()

        self.mock_seq_lens = MagicMock()
        self.mock_seq_lens.max = MagicMock(return_value=1024)

        self.mock_position = MagicMock()
        self.mock_dtype = self.mock_torch.float16
        self.mock_device = MagicMock()

    def test_get_splitfuse_attn_mask_supported_dtype(self):
        """Test get_splitfuse_attn_mask_patch with supported dtype"""
        mock_attn_mask = MagicMock()
        mock_sliced_attn_mask = MagicMock()
        mock_scaled_attn_mask = MagicMock()
        mock_contiguous_attn_mask = MagicMock()
        mock_final_attn_mask = MagicMock()

        self.mock_torch.index_select = MagicMock(return_value=mock_attn_mask)
        mock_attn_mask.__getitem__ = MagicMock(return_value=mock_sliced_attn_mask)
        mock_sliced_attn_mask.__mul__ = MagicMock(return_value=mock_scaled_attn_mask)
        mock_scaled_attn_mask.contiguous = MagicMock(return_value=mock_contiguous_attn_mask)
        mock_contiguous_attn_mask.to = MagicMock(return_value=mock_final_attn_mask)

        self.mock_seq_lens.__iter__ = MagicMock(return_value=iter([1024]))

        result = self.patch_attention_mask.get_splitfuse_attn_mask_patch(
            self.mock_self, self.mock_seq_lens, self.mock_position,
            self.mock_dtype, self.mock_device
        )

        self.mock_self._update_attn_cache.assert_called_once_with(1024, self.mock_dtype)

        self.mock_torch.index_select.assert_called_once_with(
            self.mock_self.attn_mask_cache, dim=0, index=self.mock_position
        )
        mock_attn_mask.__getitem__.assert_called_once_with((slice(None), slice(None, 1024)))
        mock_sliced_attn_mask.__mul__.assert_called_once()
        mock_scaled_attn_mask.contiguous.assert_called_once()
        mock_contiguous_attn_mask.to.assert_called_once_with(self.mock_device, non_blocking=True)

        self.assertEqual(result, mock_final_attn_mask)

    def test_get_splitfuse_attn_mask_unsupported_dtype(self):
        """Test get_splitfuse_attn_mask_patch with unsupported dtype"""
        unsupported_dtype = self.mock_torch.float32

        with self.assertRaises(ValueError) as context:
            self.patch_attention_mask.get_splitfuse_attn_mask_patch(
                self.mock_self, self.mock_seq_lens, self.mock_position,
                unsupported_dtype, self.mock_device
            )

        self.assertIn("splitfuse_attn_mask now only supports bf16 and fp16", str(context.exception))

        self.mock_self._update_attn_cache.assert_not_called()

    def test_get_splitfuse_attn_mask_empty_seq_lens(self):
        """Test get_splitfuse_attn_mask_patch with empty seq_lens"""
        empty_seq_lens = MagicMock()
        empty_seq_lens.max = MagicMock(return_value=0)

        mock_attn_mask = MagicMock()
        mock_sliced_attn_mask = MagicMock()
        mock_scaled_attn_mask = MagicMock()
        mock_contiguous_attn_mask = MagicMock()
        mock_final_attn_mask = MagicMock()

        self.mock_torch.index_select = MagicMock(return_value=mock_attn_mask)
        mock_attn_mask.__getitem__ = MagicMock(return_value=mock_sliced_attn_mask)
        mock_sliced_attn_mask.__mul__ = MagicMock(return_value=mock_scaled_attn_mask)
        mock_scaled_attn_mask.contiguous = MagicMock(return_value=mock_contiguous_attn_mask)
        mock_contiguous_attn_mask.to = MagicMock(return_value=mock_final_attn_mask)

        empty_seq_lens.__iter__ = MagicMock(return_value=iter([]))

        result = self.patch_attention_mask.get_splitfuse_attn_mask_patch(
            self.mock_self, empty_seq_lens, self.mock_position,
            self.mock_dtype, self.mock_device
        )

        self.mock_self._update_attn_cache.assert_called_once_with(0, self.mock_dtype)

        self.mock_torch.index_select.assert_called_once_with(
            self.mock_self.attn_mask_cache, dim=0, index=self.mock_position
        )
        mock_attn_mask.__getitem__.assert_called_once_with((slice(None), slice(None, 0)))
        mock_sliced_attn_mask.__mul__.assert_called_once()
        mock_scaled_attn_mask.contiguous.assert_called_once()
        mock_contiguous_attn_mask.to.assert_called_once_with(self.mock_device, non_blocking=True)

        self.assertEqual(result, mock_final_attn_mask)


if __name__ == '__main__':
    unittest.main()
