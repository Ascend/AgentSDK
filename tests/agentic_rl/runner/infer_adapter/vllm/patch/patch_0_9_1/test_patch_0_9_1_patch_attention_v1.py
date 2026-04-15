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


class TestPatchAttentionV1(unittest.TestCase):
    """Test suite for the patch_attention_v1.py module."""

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
        cls.mock_torch.float32 = "float32"
        cls.mock_torch.tensor = MagicMock(return_value=MagicMock())
        cls.mock_torch.zeros = MagicMock(return_value=MagicMock())
        cls.mock_torch.empty = MagicMock(return_value=MagicMock())
        cls.mock_torch.cumsum = MagicMock(return_value=MagicMock())
        cls.mock_torch.max = MagicMock(return_value=MagicMock())
        cls.mock_torch.npu = MagicMock()
        cls.mock_torch.npu.ExternalEvent = MagicMock()
        cls.mock_torch.npu.current_stream = MagicMock(return_value="mock_stream")
        cls.mock_torch.npu.graph_task_group_begin = MagicMock()
        cls.mock_torch.npu.graph_task_group_end = MagicMock(return_value="mock_handle")
        cls.mock_torch.ops = MagicMock()
        cls.mock_torch.ops.vllm = MagicMock()
        cls.mock_torch.ops.vllm.unified_ascend_attention_with_output = MagicMock()

        cls.mock_torch_npu = MagicMock()
        cls.mock_torch_npu._npu_flash_attention = MagicMock()
        cls.mock_torch_npu._npu_flash_attention_qlens = MagicMock()
        cls.mock_torch_npu._npu_paged_attention = MagicMock()
        cls.mock_torch_npu._npu_paged_attention_splitfuse = MagicMock()
        cls.mock_torch_npu._npu_reshape_and_cache = MagicMock()

        class MockAttentionType:
            DECODER = "DECODER"

        cls.MockAttentionType = MockAttentionType

        cls.mock_vllm = MagicMock()
        cls.mock_vllm.attention = MagicMock()
        cls.mock_vllm.attention.backends = MagicMock()
        cls.mock_vllm.attention.backends.abstract = MagicMock()
        cls.mock_vllm.attention.backends.abstract.AttentionType = MockAttentionType
        cls.mock_vllm.forward_context = MagicMock()
        cls.mock_vllm.forward_context.get_forward_context = MagicMock()

        class MockAscendAttentionState:
            PrefillNoCache = "PrefillNoCache"
            PrefillCacheHit = "PrefillCacheHit"
            DecodeOnly = "DecodeOnly"

        cls.AscendAttentionState = MockAscendAttentionState

        cls.mock_vllm_ascend = MagicMock()
        cls.mock_vllm_ascend.attention = MagicMock()
        cls.mock_vllm_ascend.attention.attention_v1 = MagicMock()
        cls.mock_vllm_ascend.attention.attention_v1.AscendAttentionState = MockAscendAttentionState
        cls.mock_vllm_ascend.ops = MagicMock()
        cls.mock_vllm_ascend.ops.attention = MagicMock()
        cls.mock_vllm_ascend.ops.attention.vanilla_chunked_prefill = MagicMock()
        cls.mock_vllm_ascend.utils = MagicMock()
        cls.mock_vllm_ascend.utils.get_graph_params = MagicMock()

        cls.mock_forward_context = MagicMock()
        cls.mock_forward_context.capturing = False
        cls.mock_vllm.forward_context.get_forward_context.return_value = cls.mock_forward_context

        cls.mock_graph_params = MagicMock()
        cls.mock_graph_params.events = {}
        cls.mock_graph_params.attn_params = {}
        cls.mock_graph_params.handles = {}
        cls.mock_vllm_ascend.utils.get_graph_params.return_value = cls.mock_graph_params

        cls.mock_ascend_impl = MagicMock()
        cls.mock_vllm_ascend.attention.attention_v1.AscendAttentionBackendImpl = cls.mock_ascend_impl

        cls.modules_patcher = patch.dict('sys.modules', {
            'torch': cls.mock_torch,
            'torch_npu': cls.mock_torch_npu,
            'vllm': cls.mock_vllm,
            'vllm.attention': cls.mock_vllm.attention,
            'vllm.attention.backends': cls.mock_vllm.attention.backends,
            'vllm.attention.backends.abstract': cls.mock_vllm.attention.backends.abstract,
            'vllm.forward_context': cls.mock_vllm.forward_context,
            'vllm_ascend': cls.mock_vllm_ascend,
            'vllm_ascend.attention': cls.mock_vllm_ascend.attention,
            'vllm_ascend.attention.attention_v1': cls.mock_vllm_ascend.attention.attention_v1,
            'vllm_ascend.ops': cls.mock_vllm_ascend.ops,
            'vllm_ascend.ops.attention': cls.mock_vllm_ascend.ops.attention,
            'vllm_ascend.utils': cls.mock_vllm_ascend.utils,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        module_file_path = os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch',
                                        'patch_0_9_1',
                                        'patch_attention_v1.py')

        spec = importlib.util.spec_from_file_location("patch_attention_v1", module_file_path)
        cls.patch_attention_v1 = importlib.util.module_from_spec(spec)
        sys.modules['patch_attention_v1'] = cls.patch_attention_v1

        spec.loader.exec_module(cls.patch_attention_v1)

    @classmethod
    def _cleanup_mocks(cls):
        cls.modules_patcher.stop()

    def setUp(self):
        self.mock_torch.reset_mock()
        self.mock_torch_npu.reset_mock()
        self.mock_vllm.reset_mock()
        self.mock_vllm_ascend.reset_mock()

        self.mock_self = MagicMock()
        self.mock_self.num_heads = 8
        self.mock_self.num_kv_heads = 8
        self.mock_self.head_size = 64
        self.mock_self.hidden_size = 512
        self.mock_self.attn_type = self.MockAttentionType.DECODER
        self.mock_self.scale = 0.125
        self.mock_self.key_cache = None
        self.mock_self.value_cache = None

        class MockLayer:
            def __init__(self):
                self.layer_name = "test_layer"
                self._k_scale_float = 1.0
                self._v_scale_float = 1.0

        self.mock_layer = MockLayer()

        self.mock_query = MagicMock()
        self.mock_query.shape = (10, 512)
        self.mock_query.dtype = self.mock_torch.float32
        self.mock_query.device = "npu:0"
        self.mock_query.view.return_value = self.mock_torch.zeros(10, 8, 64)

        self.mock_key = MagicMock()
        self.mock_key.view.return_value = self.mock_torch.zeros(10, 8, 64)

        self.mock_value = MagicMock()
        self.mock_value.view.return_value = self.mock_torch.zeros(10, 8, 64)
        self.mock_value.contiguous.return_value = self.mock_torch.zeros(10, 8, 64)

        self.mock_attn_metadata = MagicMock()
        self.mock_attn_metadata.num_actual_tokens = 10
        self.mock_attn_metadata.slot_mapping = self.mock_torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.mock_attn_metadata.seq_lens = self.mock_torch.tensor([10])
        self.mock_attn_metadata.query_lens = self.mock_torch.tensor([10])
        self.mock_attn_metadata.block_tables = self.mock_torch.tensor([[0, 1]])

        mock_mask = MagicMock()
        mock_mask_ne = MagicMock()
        mock_mask_ne.long.return_value = MagicMock()
        mock_mask_ne.long.return_value.to.return_value = MagicMock()
        mock_mask.__ne__.return_value = mock_mask_ne
        self.mock_attn_metadata.attn_mask = mock_mask

        self.mock_output = MagicMock()
        self.mock_output.view.return_value = self.mock_torch.zeros(10, 512)

        self.mock_kv_cache = []

    def test_forward_patch_trace_flag(self):
        result = self.patch_attention_v1.forward_patch(
            self.mock_self, self.mock_layer, self.mock_query, self.mock_key, self.mock_value,
            self.mock_kv_cache, self.mock_attn_metadata, self.mock_output, trace_flag=True
        )

        self.mock_torch.ops.vllm.unified_ascend_attention_with_output.assert_called_once()
        self.mock_output.view.assert_called_once_with(10, 512)
        self.assertEqual(result, self.mock_output.view.return_value)

    def test_forward_patch_prefill_no_cache(self):
        self.mock_attn_metadata.attn_state = self.AscendAttentionState.PrefillNoCache

        result = self.patch_attention_v1.forward_patch(
            self.mock_self, self.mock_layer, self.mock_query, self.mock_key, self.mock_value,
            [], self.mock_attn_metadata, self.mock_output, trace_flag=False
        )

        self.mock_torch_npu._npu_flash_attention.assert_called_once()
        self.mock_output.view.assert_called_once_with(10, 512)

    def test_forward_patch_prefill_cache_hit(self):
        self.mock_attn_metadata.attn_state = self.AscendAttentionState.PrefillCacheHit
        self.mock_self.key_cache = self.mock_torch.zeros(10, 8, 64)
        self.mock_self.value_cache = self.mock_torch.zeros(10, 8, 64)

        result = self.patch_attention_v1.forward_patch(
            self.mock_self, self.mock_layer, self.mock_query, self.mock_key, self.mock_value,
            self.mock_kv_cache, self.mock_attn_metadata, self.mock_output, trace_flag=False
        )

        self.mock_torch_npu._npu_flash_attention_qlens.assert_called_once()
        self.mock_output.view.assert_called_once_with(10, 512)

    def test_forward_patch_decode_only(self):
        self.mock_attn_metadata.attn_state = self.AscendAttentionState.DecodeOnly
        self.mock_self.key_cache = self.mock_torch.zeros(10, 8, 64)
        self.mock_self.value_cache = self.mock_torch.zeros(10, 8, 64)
        self.mock_forward_context.capturing = False

        result = self.patch_attention_v1.forward_patch(
            self.mock_self, self.mock_layer, self.mock_query, self.mock_key, self.mock_value,
            self.mock_kv_cache, self.mock_attn_metadata, self.mock_output, trace_flag=False
        )

        self.mock_torch_npu._npu_paged_attention.assert_called_once()
        self.mock_output.view.assert_called_once_with(10, 512)

    def test_forward_patch_decode_only_with_capturing(self):
        self.mock_attn_metadata.attn_state = self.AscendAttentionState.DecodeOnly
        self.mock_self.key_cache = self.mock_torch.zeros(10, 8, 64)
        self.mock_self.value_cache = self.mock_torch.zeros(10, 8, 64)
        self.mock_forward_context.capturing = True

        self.mock_graph_params.events[10] = []
        self.mock_graph_params.attn_params[10] = []
        self.mock_graph_params.handles[10] = []

        mock_event = MagicMock()
        self.mock_torch.npu.ExternalEvent.return_value = mock_event

        result = self.patch_attention_v1.forward_patch(
            self.mock_self, self.mock_layer, self.mock_query, self.mock_key, self.mock_value,
            self.mock_kv_cache, self.mock_attn_metadata, self.mock_output, trace_flag=False
        )

        self.mock_torch.npu.ExternalEvent.assert_called_once()
        mock_event.wait.assert_called_once()
        mock_event.reset.assert_called_once()
        self.mock_torch.npu.graph_task_group_begin.assert_called_once()
        self.mock_torch.npu.graph_task_group_end.assert_called_once()
        self.mock_torch_npu._npu_paged_attention.assert_called_once()

        self.assertEqual(len(self.mock_graph_params.events[10]), 1)
        self.assertEqual(len(self.mock_graph_params.attn_params[10]), 1)
        self.assertEqual(len(self.mock_graph_params.handles[10]), 1)

        self.mock_output.view.assert_called_once_with(10, 512)

    def test_forward_patch_default_case(self):
        self.mock_attn_metadata.attn_state = "OtherState"
        self.mock_self.key_cache = self.mock_torch.zeros(10, 8, 64)
        self.mock_self.value_cache = self.mock_torch.zeros(10, 8, 64)

        result = self.patch_attention_v1.forward_patch(
            self.mock_self, self.mock_layer, self.mock_query, self.mock_key, self.mock_value,
            self.mock_kv_cache, self.mock_attn_metadata, self.mock_output, trace_flag=False
        )

        self.mock_torch_npu._npu_paged_attention_splitfuse.assert_called_once()
        self.mock_output.view.assert_called_once_with(10, 512)

    def test_forward_patch_large_head_size(self):
        self.mock_attn_metadata.attn_state = "OtherState"
        self.mock_self.head_size = 192
        self.mock_self.hidden_size = 1536
        self.mock_self.key_cache = self.mock_torch.zeros(10, 8, 192)
        self.mock_self.value_cache = self.mock_torch.zeros(10, 8, 192)

        mock_tensor = MagicMock()
        mock_tensor.cumsum.return_value = mock_tensor
        original_tensor = self.mock_torch.tensor
        self.mock_torch.tensor = MagicMock(return_value=mock_tensor)
        self.mock_torch.max.return_value = mock_tensor

        try:
            result = self.patch_attention_v1.forward_patch(
                self.mock_self, self.mock_layer, self.mock_query, self.mock_key, self.mock_value,
                self.mock_kv_cache, self.mock_attn_metadata, self.mock_output, trace_flag=False
            )

            self.mock_vllm_ascend.ops.attention.vanilla_chunked_prefill.assert_called_once()
            self.mock_output.view.assert_called_once_with(10, 1536)
        finally:
            self.mock_torch.tensor = original_tensor

    def test_forward_patch_with_kv_cache(self):
        self.mock_attn_metadata.attn_state = "OtherState"
        mock_key_cache = self.mock_torch.zeros(10, 8, 64)
        mock_value_cache = self.mock_torch.zeros(10, 8, 64)
        self.mock_kv_cache = [mock_key_cache, mock_value_cache]

        result = self.patch_attention_v1.forward_patch(
            self.mock_self, self.mock_layer, self.mock_query, self.mock_key, self.mock_value,
            self.mock_kv_cache, self.mock_attn_metadata, self.mock_output, trace_flag=False
        )

        self.assertEqual(self.mock_self.key_cache, mock_key_cache)
        self.assertEqual(self.mock_self.value_cache, mock_value_cache)
        self.mock_torch_npu._npu_reshape_and_cache.assert_called_once()
        self.mock_output.view.assert_called_once_with(10, 512)


if __name__ == '__main__':
    unittest.main()
