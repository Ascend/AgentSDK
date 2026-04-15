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


class TestPatchAttentionV1(unittest.TestCase):
    """Test patch_attention_v1.py module"""

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
        """Setup mock objects for torch, torch_npu, and vllm_ascend"""
        cls.mock_torch = MagicMock()
        cls.mock_torch.float16 = "float16"

        cls.mock_torch_npu = MagicMock()
        cls.mock_torch_npu._npu_flash_attention = MagicMock()
        cls.mock_torch_npu.npu_fused_infer_attention_score = MagicMock(return_value=(MagicMock(), None))
        cls.mock_torch_npu.npu_format_cast = MagicMock()
        cls.mock_torch_npu.npu = MagicMock()

        cls.mock_vllm_ascend = MagicMock()
        cls.mock_vllm_ascend_utils = MagicMock()
        cls.mock_vllm_ascend.utils = cls.mock_vllm_ascend_utils
        cls.mock_vllm_ascend_utils.ACL_FORMAT_FRACTAL_NZ = "ACL_FORMAT_FRACTAL_NZ"
        cls.mock_vllm_ascend_utils.aligned_16 = MagicMock(side_effect=lambda x: x)
        cls.mock_vllm_ascend_utils.is_310p = MagicMock(return_value=False)
        cls.mock_vllm_ascend_utils.nd_to_nz_2d = MagicMock()
        cls.mock_vllm_ascend_utils.nd_to_nz_spec = MagicMock()

        cls.mock_vllm_ascend_attention = MagicMock()
        cls.mock_vllm_ascend.attention = cls.mock_vllm_ascend_attention

        cls.mock_vllm_ascend_attention_v1 = MagicMock()
        cls.mock_vllm_ascend.attention.attention_v1 = cls.mock_vllm_ascend_attention_v1
        cls.mock_vllm_ascend_attention_v1.AscendAttentionBackendImpl = MagicMock()
        cls.mock_vllm_ascend_attention_v1.AscendMetadata = MagicMock()

        cls.modules_patcher = patch.dict('sys.modules', {
            'torch': cls.mock_torch,
            'torch_npu': cls.mock_torch_npu,
            'vllm_ascend': cls.mock_vllm_ascend,
            'vllm_ascend.utils': cls.mock_vllm_ascend_utils,
            'vllm_ascend.attention': cls.mock_vllm_ascend_attention,
            'vllm_ascend.attention.attention_v1': cls.mock_vllm_ascend_attention_v1,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        spec = importlib.util.spec_from_file_location(
            'patch_attention_v1',
            os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2',
                         'patch_attention_v1.py')
        )
        cls.patch_attention_v1 = importlib.util.module_from_spec(spec)
        sys.modules['patch_attention_v1'] = cls.patch_attention_v1
        spec.loader.exec_module(cls.patch_attention_v1)

    @classmethod
    def _cleanup_mocks(cls):
        """Clean up mock patches"""
        cls.modules_patcher.stop()

    def setUp(self):
        """Set up test environment"""
        self.mock_torch_npu._npu_flash_attention.reset_mock()
        self.mock_torch_npu.npu_fused_infer_attention_score.reset_mock()
        self.mock_torch_npu.npu_format_cast.reset_mock()
        self.mock_vllm_ascend_utils.aligned_16.reset_mock(side_effect=lambda x: x)
        self.mock_vllm_ascend_utils.is_310p.reset_mock(return_value=False)

        self.mock_self = MagicMock()
        self.mock_self.num_heads = 8
        self.mock_self.num_kv_heads = 4
        self.mock_self.head_size = 64
        self.mock_self.sliding_window = None
        self.mock_self.scale = 0.125
        self.mock_self._repeat_kv = MagicMock(side_effect=lambda x, _: x)

        self.mock_query = MagicMock()
        self.mock_query.dtype = self.mock_torch.float16
        self.mock_key = MagicMock()
        self.mock_value = MagicMock()
        self.mock_output = MagicMock()

        self.mock_attn_metadata = MagicMock()

        self.mock_mask = MagicMock()
        self.mock_mask_repeated = MagicMock()
        self.mock_mask_contiguous = MagicMock()
        self.mock_mask_boolean = MagicMock()
        self.mock_mask_long = MagicMock()

        self.mock_mask.__ne__ = MagicMock(return_value=self.mock_mask_boolean)
        self.mock_mask_boolean.long = MagicMock(return_value=self.mock_mask_long)
        self.mock_mask_long.to = MagicMock(return_value=self.mock_mask_long)

        self.mock_mask.repeat = MagicMock(return_value=self.mock_mask_repeated)
        self.mock_mask_repeated.contiguous = MagicMock(return_value=self.mock_mask_contiguous)

        self.mock_attn_metadata.attn_mask = self.mock_mask
        self.mock_attn_metadata.seq_lens = MagicMock()

    def test_forward_prefill_no_cache_basic(self):
        """Test _forward_prefill_no_cache_patch with basic configuration"""
        num_tokens = 10
        self.mock_query.shape = (num_tokens, 8, 64)

        result = self.patch_attention_v1._forward_prefill_no_cache_patch(
            self.mock_self, self.mock_query, self.mock_key, self.mock_value,
            self.mock_attn_metadata, self.mock_output, num_tokens
        )

        self.mock_torch_npu._npu_flash_attention.assert_called_once()

        self.mock_output.__getitem__.assert_called_once_with((slice(None, num_tokens), slice(None), slice(None)))

        self.assertEqual(result, self.mock_output.__getitem__.return_value)

    def test_forward_prefill_no_cache_with_sliding_window(self):
        """Test _forward_prefill_no_cache_patch with sliding window"""
        num_tokens = 2048
        self.mock_self.sliding_window = 1024
        self.mock_attn_metadata.attn_mask.shape = (2048, 1, 1, 2048)

        mock_fused_output = MagicMock()
        mock_viewed_output = MagicMock()
        mock_sliced_output = MagicMock()

        mock_fused_output.view = MagicMock(return_value=mock_viewed_output)
        mock_viewed_output.__getitem__ = MagicMock(return_value=mock_sliced_output)
        self.mock_torch_npu.npu_fused_infer_attention_score = MagicMock(return_value=(mock_fused_output, None))

        result = self.patch_attention_v1._forward_prefill_no_cache_patch(
            self.mock_self, self.mock_query, self.mock_key, self.mock_value,
            self.mock_attn_metadata, self.mock_output, num_tokens
        )

        self.mock_self._repeat_kv.assert_any_call(self.mock_key, 2)
        self.mock_self._repeat_kv.assert_any_call(self.mock_value, 2)

        self.mock_torch_npu.npu_fused_infer_attention_score.assert_called_once()

        mock_fused_output.view.assert_called_once_with(num_tokens, self.mock_self.num_heads, self.mock_self.head_size)

        mock_viewed_output.__getitem__.assert_called_once_with((slice(None, num_tokens), slice(None), slice(None)))

        self.assertEqual(result, mock_sliced_output)

    def test_forward_prefill_no_cache_310p(self):
        """Test _forward_prefill_no_cache_patch on 310p device"""
        num_tokens = 10
        self.mock_query.shape = (num_tokens, 8, 64)

        self.mock_vllm_ascend_utils.is_310p.return_value = True

        self.mock_attn_metadata.seq_lens.size = MagicMock(return_value=1)

        mock_formatted_mask = MagicMock()
        mock_formatted_mask_boolean = MagicMock()
        mock_formatted_mask_long = MagicMock()
        mock_formatted_mask.__ne__ = MagicMock(return_value=mock_formatted_mask_boolean)
        mock_formatted_mask_boolean.long = MagicMock(return_value=mock_formatted_mask_long)
        mock_formatted_mask_long.to = MagicMock(return_value=mock_formatted_mask_long)

        self.mock_torch_npu.npu_format_cast = MagicMock(return_value=mock_formatted_mask)

        result = self.patch_attention_v1._forward_prefill_no_cache_patch(
            self.mock_self, self.mock_query, self.mock_key, self.mock_value,
            self.mock_attn_metadata, self.mock_output, num_tokens
        )

        self.mock_vllm_ascend_utils.aligned_16.assert_any_call(self.mock_query)
        self.mock_vllm_ascend_utils.aligned_16.assert_any_call(self.mock_key)
        self.mock_vllm_ascend_utils.aligned_16.assert_any_call(self.mock_value)
        self.mock_vllm_ascend_utils.aligned_16.assert_any_call(self.mock_output)

        self.mock_attn_metadata.attn_mask.repeat.assert_called_once_with(1, 1, 1, 1)
        self.mock_mask_repeated.contiguous.assert_called_once()
        self.mock_torch_npu.npu_format_cast.assert_called_once()

        self.mock_vllm_ascend_utils.is_310p.return_value = False

        self.mock_torch_npu._npu_flash_attention.assert_called_once()

        aligned_output = self.mock_vllm_ascend_utils.aligned_16.return_value
        self.assertEqual(result, aligned_output.__getitem__.return_value)


if __name__ == '__main__':
    unittest.main()
