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


class TestPatchAttention(unittest.TestCase):
    """Test the patch_attention.py module"""

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
        """Setup mock objects for numpy, torch, torchair, vllm, and vllm_ascend"""
        cls.mock_numpy = MagicMock()

        cls.mock_torch = MagicMock()
        cls.mock_torch.float32 = "float32"
        cls.mock_torch.tensor = MagicMock()

        cls.mock_torchair = MagicMock()
        cls.mock_torchair._contrib = MagicMock()
        cls.mock_torchair._contrib.custom_torch_ops = MagicMock()

        cls.mock_vllm = MagicMock()
        cls.mock_vllm.attention = MagicMock()
        cls.mock_vllm.attention.backends = MagicMock()
        cls.mock_vllm_attention_backends_utils = MagicMock()
        cls.mock_vllm.attention.backends.utils = cls.mock_vllm_attention_backends_utils
        cls.mock_vllm_attention_backends_utils.PAD_SLOT_ID = -1
        cls.mock_vllm_attention_backends_utils.is_block_tables_empty = MagicMock(return_value=False)

        cls.mock_vllm_utils = MagicMock()
        cls.mock_vllm.utils = cls.mock_vllm_utils
        cls.mock_vllm_utils.async_tensor_h2d = MagicMock(
            side_effect=lambda x, dtype, device, pin_memory: cls.mock_torch.tensor(x, dtype=dtype, device=device))
        cls.mock_vllm_utils.make_tensor_with_pad = MagicMock(
            side_effect=lambda x, pad, dtype, device: cls.mock_torch.tensor(x, dtype=dtype, device=device))

        cls.mock_vllm_ascend = MagicMock()
        cls.mock_vllm_ascend.attention = MagicMock()
        cls.mock_vllm_ascend_attention = MagicMock()
        cls.mock_vllm_ascend.attention.attention = cls.mock_vllm_ascend_attention

        class MockAscendMetadata:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        cls.mock_vllm_ascend_attention.AscendMetadata = MockAscendMetadata

        cls.modules_patcher = patch.dict('sys.modules', {
            'numpy': cls.mock_numpy,
            'torch': cls.mock_torch,
            'torchair': cls.mock_torchair,
            'torchair._contrib': cls.mock_torchair._contrib,
            'torchair._contrib.custom_torch_ops': cls.mock_torchair._contrib.custom_torch_ops,
            'vllm': cls.mock_vllm,
            'vllm.attention': cls.mock_vllm.attention,
            'vllm.attention.backends': cls.mock_vllm.attention.backends,
            'vllm.attention.backends.utils': cls.mock_vllm_attention_backends_utils,
            'vllm.utils': cls.mock_vllm_utils,
            'vllm_ascend': cls.mock_vllm_ascend,
            'vllm_ascend.attention': cls.mock_vllm_ascend.attention,
            'vllm_ascend.attention.attention': cls.mock_vllm_ascend_attention,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        module_file_path = os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch',
                                        'patch_0_9_1',
                                        'patch_attention.py')

        spec = importlib.util.spec_from_file_location('patch_attention', module_file_path)
        cls.patch_attention = importlib.util.module_from_spec(spec)
        sys.modules['patch_attention'] = cls.patch_attention

        spec.loader.exec_module(cls.patch_attention)

    @classmethod
    def _cleanup_mocks(cls):
        """Clean up mock patches"""
        cls.modules_patcher.stop()

    def setUp(self):
        """Set up test environment"""

        class MockRunner:
            def __init__(self, torch_float32):
                self.device = "npu:0"
                self.pin_memory = False
                self.model_config = MagicMock()
                self.model_config.dtype = torch_float32

        class MockInputBuilder:
            def __init__(self):
                self.inter_data_list = []
                self.chunked_prefill_enabled = False

        self.mock_runner = MockRunner(self.mock_torch.float32)
        self.mock_input_builder = MockInputBuilder()

        self.AttentionMaskBuilder = MagicMock()
        self.mock_attn_mask_builder = self.AttentionMaskBuilder()
        self.mock_attn_mask_builder._seq_len_cached = 0
        self.mock_attn_mask_builder.splitfuse_mask_value = 1.0

        self.mock_attn_mask_cache = MagicMock()
        self.mock_attn_mask_cache.numel = MagicMock(return_value=100)
        self.mock_attn_mask_cache.__getitem__ = MagicMock(return_value=MagicMock(__getitem__=MagicMock(return_value=0)))
        self.mock_attn_mask_builder.attn_mask_cache = self.mock_attn_mask_cache
        self.mock_attn_mask_builder.get_attn_mask = MagicMock(return_value=self.mock_attn_mask_cache)
        self.mock_attn_mask_builder.update_attn_cache = MagicMock()

        self.AscendMetadataBuilder = MagicMock()
        self.mock_metadata_builder = self.AscendMetadataBuilder()
        self.mock_metadata_builder.runner = self.mock_runner
        self.mock_metadata_builder.input_builder = self.mock_input_builder
        self.mock_metadata_builder.num_prefills = 0
        self.mock_metadata_builder.num_decode_tokens = 0
        self.mock_metadata_builder.num_prefill_tokens = 0
        self.mock_metadata_builder.prefill_seq_lens = []
        self.mock_metadata_builder.curr_seq_lens = []
        self.mock_metadata_builder.slot_mapping = []
        self.mock_metadata_builder.block_tables = []
        self.mock_metadata_builder.context_lens = []
        self.mock_metadata_builder.attn_mask = None
        self.mock_metadata_builder.compress_mask = None
        self.mock_metadata_builder.chunk_mask = None
        self.mock_metadata_builder.multimodal_placeholder_maps = {}

        self.mock_metadata_builder._attn_mask_builder = self.mock_attn_mask_builder

        self.original_get_splitfuse_attn_mask = getattr(self.AttentionMaskBuilder, 'get_splitfuse_attn_mask', None)
        self.original_build = getattr(self.AscendMetadataBuilder, 'build', None)

    def tearDown(self):
        """Clean up test environment"""
        if self.original_get_splitfuse_attn_mask is not None:
            self.AttentionMaskBuilder.get_splitfuse_attn_mask = self.original_get_splitfuse_attn_mask
        if self.original_build is not None:
            self.AscendMetadataBuilder.build = self.original_build

    def test_get_splitfuse_attn_mask_cache_hit(self):
        """Test get_splitfuse_attn_mask method when cache hits"""
        self.mock_attn_mask_builder._seq_len_cached = 10
        max_seq_len = 5

        result = self.patch_attention.get_splitfuse_attn_mask_patch(
            self.mock_attn_mask_builder, [5], [3], self.mock_torch.tensor([0, 1, 2]), self.mock_torch.float32, "npu:0"
        )

        self.mock_attn_mask_builder.update_attn_cache.assert_called_once_with(5, self.mock_torch.float32, "npu:0")

        self.assertIsNotNone(result)

    def test_get_splitfuse_attn_mask_cache_miss(self):
        """Test get_splitfuse_attn_mask method when cache misses"""
        self.mock_attn_mask_builder._seq_len_cached = 5
        max_seq_len = 10

        result = self.patch_attention.get_splitfuse_attn_mask_patch(
            self.mock_attn_mask_builder, [5, 7], [3, 4], self.mock_torch.tensor([0, 1, 2, 3, 4, 5, 6]),
            self.mock_torch.float32, "npu:0"
        )

        self.mock_attn_mask_builder.update_attn_cache.assert_not_called()

        self.assertIsNotNone(result)

    def test_build_with_decode_only(self):
        """Test build method with decode-only scenario"""
        self.mock_metadata_builder.num_prefills = 0
        self.mock_metadata_builder.num_decode_tokens = 5
        self.mock_metadata_builder.slot_mapping = [1, 2, 3, 4, 5]
        self.mock_metadata_builder.block_tables = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        self.mock_metadata_builder.context_lens = [10, 15, 20, 25, 30]

        result = self.patch_attention.build_patch(
            self.mock_metadata_builder, [10, 15, 20, 25, 30], [1, 1, 1, 1, 1], -1
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.num_prefills, 0)
        self.assertEqual(result.num_decode_tokens, 5)

    def test_build_with_prefill(self):
        """Test build method with prefill scenario"""
        self.mock_metadata_builder.num_prefills = 2
        self.mock_metadata_builder.num_prefill_tokens = 10
        self.mock_metadata_builder.slot_mapping = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.mock_metadata_builder.block_tables = [[0, 1], [2, 3]]
        self.mock_metadata_builder.context_lens = [0, 0]
        self.mock_metadata_builder.prefill_seq_lens = [5, 5]
        self.mock_metadata_builder.chunked_prefill_enabled = False

        result = self.patch_attention.build_patch(
            self.mock_metadata_builder, [5, 5], [5, 5], -1
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.num_prefills, 2)
        self.assertEqual(result.num_prefill_tokens, 10)


if __name__ == '__main__':
    unittest.main()
