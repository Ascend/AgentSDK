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


class TestPatchBase(unittest.TestCase):
    """Test patch_base.py module"""

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
        """Setup mock objects for torch, torch_npu, and vllm"""
        cls.mock_torch = MagicMock()
        cls.mock_torch.float16 = "float16"

        cls.mock_torch_npu = MagicMock()

        cls.mock_vllm = MagicMock()
        cls.mock_vllm.model_executor = MagicMock()
        cls.mock_vllm.model_executor.layers = MagicMock()
        cls.mock_vllm.model_executor.layers.rotary_embedding = MagicMock()
        cls.mock_vllm.model_executor.layers.rotary_embedding.base = MagicMock()

        class MockRotaryEmbedding:
            def __init__(self):
                self.rotary_dim = 64
                self.head_size = 64
                self.is_neox_style = False
                self.cos_sin_cache = MagicMock()

        cls.MockRotaryEmbedding = MockRotaryEmbedding
        cls.original_forward_native_mock = MagicMock(return_value=("original_query", "original_key"))

        cls.mock_vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding = MockRotaryEmbedding
        cls.mock_vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding.forward_native = cls.original_forward_native_mock

        cls.modules_patcher = patch.dict('sys.modules', {
            'torch': cls.mock_torch,
            'torch_npu': cls.mock_torch_npu,
            'vllm': cls.mock_vllm,
            'vllm.model_executor': cls.mock_vllm.model_executor,
            'vllm.model_executor.layers': cls.mock_vllm.model_executor.layers,
            'vllm.model_executor.layers.rotary_embedding': cls.mock_vllm.model_executor.layers.rotary_embedding,
            'vllm.model_executor.layers.rotary_embedding.base': cls.mock_vllm.model_executor.layers.rotary_embedding.base,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        spec = importlib.util.spec_from_file_location(
            'patch_base',
            os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2',
                         'patch_base.py')
        )

        cls.patch_base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.patch_base)

    @classmethod
    def _cleanup_mocks(cls):
        """Clean up mock patches"""
        cls.modules_patcher.stop()

    def setUp(self):
        """Set up test environment"""
        self.original_forward_native_mock.reset_mock(return_value=("original_query", "original_key"))
        self.mock_torch.cat.reset_mock(return_value="concatenated_tensor")
        self.mock_torch.Tensor.view.reset_mock(return_value="viewed_tensor")
        self.mock_torch.Tensor.contiguous.reset_mock(return_value="contiguous_tensor")
        self.mock_torch_npu._npu_rotary_embedding.reset_mock()

        self.mock_self = self.MockRotaryEmbedding()

        self.mock_positions = MagicMock()
        self.mock_query = MagicMock()
        self.mock_key = MagicMock()
        self.mock_query.shape = (10, 8, 64)
        self.mock_key.shape = (10, 8, 64)

        self.mock_query.device = "npu:0"
        self.mock_query.dtype = self.mock_torch.float16
        self.mock_key.device = "npu:0"
        self.mock_key.dtype = self.mock_torch.float16

        self.mock_self.cos_sin_cache.device = "npu:0"
        self.mock_self.cos_sin_cache.dtype = self.mock_torch.float16

        self.mock_viewed_query = MagicMock()
        self.mock_viewed_key = MagicMock()
        self.mock_contiguous_query = MagicMock()
        self.mock_contiguous_key = MagicMock()

        self.mock_query.view = MagicMock(return_value=self.mock_viewed_query)
        self.mock_key.view = MagicMock(return_value=self.mock_viewed_key)
        self.mock_query.contiguous = MagicMock(return_value=self.mock_contiguous_query)
        self.mock_key.contiguous = MagicMock(return_value=self.mock_contiguous_key)

        self.mock_viewed_query.__getitem__ = MagicMock(return_value=self.mock_viewed_query)
        self.mock_viewed_query.contiguous = MagicMock(return_value=self.mock_contiguous_query)
        self.mock_contiguous_query.view = MagicMock(return_value=self.mock_contiguous_query)
        self.mock_viewed_key.__getitem__ = MagicMock(return_value=self.mock_viewed_key)
        self.mock_viewed_key.contiguous = MagicMock(return_value=self.mock_contiguous_key)
        self.mock_contiguous_key.view = MagicMock(return_value=self.mock_contiguous_key)

        self.mock_contiguous_query.reshape = MagicMock(return_value=self.mock_contiguous_query)
        self.mock_contiguous_key.reshape = MagicMock(return_value=self.mock_contiguous_key)

    def test_forward_native_patch_with_offsets(self):
        """Test forward_native_patch when offsets is not None"""
        mock_offsets = MagicMock()

        expected_result = ("original_query", "original_key")
        self.original_forward_native_mock.return_value = expected_result

        result = self.patch_base.forward_native_patch(
            self.mock_self, self.mock_positions, self.mock_query, self.mock_key, mock_offsets
        )

        self.original_forward_native_mock.assert_called_once()

        self.assertEqual(result, expected_result)

    def test_forward_native_patch_with_disabled_atb_rope(self):
        """Test forward_native_patch when ENABLE_ATB_ROPE is False"""
        with patch.dict('os.environ', {'ENABLE_ATB_ROPE': 'false'}):
            expected_result = ("original_query", "original_key")
            self.original_forward_native_mock.return_value = expected_result

            result = self.patch_base.forward_native_patch(
                self.mock_self, self.mock_positions, self.mock_query, self.mock_key, None
            )

            self.original_forward_native_mock.assert_called_once()

            self.assertEqual(result, expected_result)

    def test_forward_native_patch_with_full_rotary_dim(self):
        """Test forward_native_patch when rotary_dim equals head_size"""
        mock_torch_npu_fresh = MagicMock()

        with patch.dict('os.environ', {'ENABLE_ATB_ROPE': 'true'}), \
                patch.dict('sys.modules', {'torch_npu': mock_torch_npu_fresh}):
            result = self.patch_base.forward_native_patch(
                self.mock_self, self.mock_positions, self.mock_query, self.mock_key, None
            )

            mock_torch_npu_fresh._npu_rotary_embedding.assert_called_once()

            self.original_forward_native_mock.assert_not_called()

    def test_forward_native_patch_with_partial_rotary_dim(self):
        """Test forward_native_patch when rotary_dim is less than head_size"""
        self.mock_self.rotary_dim = 32
        self.mock_self.head_size = 64
        self.mock_self.is_neox_style = False

        mock_torch_npu_fresh = MagicMock()

        mock_torch_fresh = MagicMock()
        mock_q_concat = MagicMock()
        mock_k_concat = MagicMock()
        mock_torch_fresh.cat.side_effect = [mock_q_concat, mock_k_concat]
        mock_q_concat.reshape.return_value = mock_q_concat
        mock_k_concat.reshape.return_value = mock_k_concat

        mock_vllm = MagicMock()
        mock_vllm.model_executor = MagicMock()
        mock_vllm.model_executor.layers = MagicMock()
        mock_vllm.model_executor.layers.rotary_embedding = MagicMock()
        mock_vllm.model_executor.layers.rotary_embedding.base = MagicMock()
        mock_vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding = MagicMock()

        with patch.dict('os.environ', {'ENABLE_ATB_ROPE': 'true'}), \
                patch.dict('sys.modules', {
                    'torch_npu': mock_torch_npu_fresh,
                    'torch': mock_torch_fresh,
                    'vllm': mock_vllm,
                    'vllm.model_executor': mock_vllm.model_executor,
                    'vllm.model_executor.layers': mock_vllm.model_executor.layers,
                    'vllm.model_executor.layers.rotary_embedding': mock_vllm.model_executor.layers.rotary_embedding,
                    'vllm.model_executor.layers.rotary_embedding.base': mock_vllm.model_executor.layers.rotary_embedding.base
                }):
            test_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))

            spec_fresh = importlib.util.spec_from_file_location(
                'patch_base_fresh',
                os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2',
                             'patch_base.py')
            )

            patch_base_fresh = importlib.util.module_from_spec(spec_fresh)
            spec_fresh.loader.exec_module(patch_base_fresh)

            mock_positions = MagicMock()
            mock_query = MagicMock()
            mock_key = MagicMock()
            mock_query.shape = (10, 8, 64)
            mock_key.shape = (10, 8, 64)
            mock_query.device = "npu:0"
            mock_query.dtype = mock_torch_fresh.float16
            mock_key.device = "npu:0"
            mock_key.dtype = mock_torch_fresh.float16

            mock_query_view = MagicMock()
            mock_key_view = MagicMock()
            mock_query.view.return_value = mock_query_view
            mock_key.view.return_value = mock_key_view

            mock_q_rot = MagicMock()
            mock_q_pass = MagicMock()
            mock_k_rot = MagicMock()
            mock_k_pass = MagicMock()

            def mock_query_getitem(slice_obj):
                if hasattr(slice_obj, '__len__') and len(slice_obj) > 1:
                    return mock_q_rot
                return mock_q_pass

            def mock_key_getitem(slice_obj):
                if hasattr(slice_obj, '__len__') and len(slice_obj) > 1:
                    return mock_k_rot
                return mock_k_pass

            mock_query_view.__getitem__.side_effect = mock_query_getitem
            mock_key_view.__getitem__.side_effect = mock_key_getitem

            mock_q_rot.contiguous.return_value = mock_q_rot
            mock_q_rot.view.return_value = mock_q_rot
            mock_k_rot.contiguous.return_value = mock_k_rot
            mock_k_rot.view.return_value = mock_k_rot

            self.original_forward_native_mock.reset_mock()

            result = patch_base_fresh.forward_native_patch(
                self.mock_self, mock_positions, mock_query, mock_key, None
            )

            mock_torch_npu_fresh._npu_rotary_embedding.assert_called_once()

            self.assertEqual(mock_torch_fresh.cat.call_count, 2)

            for call_args in mock_torch_fresh.cat.call_args_list:
                self.assertEqual(call_args[1]['dim'], -1)

            self.original_forward_native_mock.assert_not_called()

    def test_forward_native_patch_with_device_and_dtype_conversion(self):
        """Test forward_native_patch handles device and dtype conversion"""
        mock_cos_sin_cache = MagicMock()
        self.mock_self.cos_sin_cache = mock_cos_sin_cache
        mock_cos_sin_cache.device = "cpu"
        mock_cos_sin_cache.dtype = "float32"
        self.mock_query.device = "npu"
        self.mock_query.dtype = "float16"

        with patch.dict('os.environ', {'ENABLE_ATB_ROPE': 'true'}):
            result = self.patch_base.forward_native_patch(
                self.mock_self, self.mock_positions, self.mock_query, self.mock_key, None
            )

            mock_cos_sin_cache.to.assert_called()

            call_args_list = mock_cos_sin_cache.to.call_args_list
            self.assertGreater(len(call_args_list), 0)


if __name__ == '__main__':
    unittest.main()
