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


class TestPatchQwen3Moe(unittest.TestCase):
    """Test patch_qwen3_moe.py module"""

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
        """Setup mock objects for vllm, vllm_ascend"""
        cls.mock_vllm = MagicMock()

        cls.mock_vllm_forward_context = MagicMock()
        cls.mock_forward_context = MagicMock()
        cls.mock_forward_context.attn_metadata = MagicMock()
        cls.mock_forward_context.in_profile_run = False
        cls.mock_forward_context.with_prefill = True
        cls.mock_vllm_forward_context.get_forward_context.return_value = cls.mock_forward_context

        cls.mock_vllm_ascend = MagicMock()

        cls.mock_sequence_parallel = MagicMock()
        cls.mock_sequence_parallel.MetadataForPadding = MagicMock()
        cls.mock_sequence_parallel.init_metadata_for_sp = MagicMock()

        class MockCustomSparseMoeBlock:
            def __init__(self):
                self.gate = MagicMock(return_value=(MagicMock(), None))
                self.experts = MagicMock(return_value=MagicMock())
                self.top_k = 2

            def forward(self, hidden_states, attn_metadata=None, _metadata_for_padding=None):
                self.original_forward_called = True
                return MagicMock()

        cls.mock_qwen3_moe = MagicMock()
        cls.mock_qwen3_moe.CustomSparseMoeBlock = MockCustomSparseMoeBlock

        cls.modules_patcher = patch.dict('sys.modules', {
            'vllm': cls.mock_vllm,
            'vllm.forward_context': cls.mock_vllm_forward_context,
            'vllm_ascend': cls.mock_vllm_ascend,
            'vllm_ascend.ops.sequence_parallel': cls.mock_sequence_parallel,
            'vllm_ascend.models.qwen3_moe': cls.mock_qwen3_moe,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        spec = importlib.util.spec_from_file_location(
            'patch_qwen3_moe',
            os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2', 'patch_qwen3_moe.py')
        )
        cls.patch_qwen3_moe = importlib.util.module_from_spec(spec)
        sys.modules['patch_qwen3_moe'] = cls.patch_qwen3_moe
        spec.loader.exec_module(cls.patch_qwen3_moe)

    @classmethod
    def _cleanup_mocks(cls):
        """Clean up mock patches"""
        cls.modules_patcher.stop()

    def setUp(self):
        """Set up test environment"""
        self.mock_vllm_forward_context.get_forward_context.reset_mock()

        self.mock_hidden_states = MagicMock()
        self.mock_attn_metadata = MagicMock()
        self.mock_metadata_for_padding = MagicMock()

    def test_custom_sparse_moe_block_forward_with_attn_metadata(self):
        """Test custom_sparse_moe_block_forward function with attn_metadata provided"""
        moe_block = self.mock_qwen3_moe.CustomSparseMoeBlock()

        moe_block.forward(
            self.mock_hidden_states,
            attn_metadata=self.mock_attn_metadata,
            _metadata_for_padding=self.mock_metadata_for_padding
        )

        moe_block.gate.assert_called_once_with(self.mock_hidden_states)
        moe_block.experts.assert_called_once()
        call_args = moe_block.experts.call_args
        self.assertEqual(call_args[1]['hidden_states'], self.mock_hidden_states)
        self.assertIsNotNone(call_args[1]['router_logits'])
        self.assertEqual(call_args[1]['is_prefill'], True)
        self.assertEqual(call_args[1]['top_k'], 2)
        self.assertEqual(call_args[1]['enable_force_load_balance'], False)
        self.assertIsNone(call_args[1]['shared_experts'])
        self.assertEqual(call_args[1]['_metadata_for_padding'], self.mock_metadata_for_padding)
        self.assertEqual(self.mock_vllm_forward_context.get_forward_context.call_count, 1)

    def test_custom_sparse_moe_block_forward_without_attn_metadata(self):
        """Test custom_sparse_moe_block_forward function without attn_metadata provided"""
        moe_block = self.mock_qwen3_moe.CustomSparseMoeBlock()

        moe_block.forward(
            self.mock_hidden_states,
            _metadata_for_padding=self.mock_metadata_for_padding
        )

        moe_block.gate.assert_called_once_with(self.mock_hidden_states)
        moe_block.experts.assert_called_once()
        call_args = moe_block.experts.call_args
        self.assertEqual(call_args[1]['hidden_states'], self.mock_hidden_states)
        self.assertIsNotNone(call_args[1]['router_logits'])
        self.assertEqual(call_args[1]['is_prefill'], True)
        self.assertEqual(call_args[1]['top_k'], 2)
        self.assertEqual(call_args[1]['enable_force_load_balance'], False)
        self.assertIsNone(call_args[1]['shared_experts'])
        self.assertEqual(call_args[1]['_metadata_for_padding'], self.mock_metadata_for_padding)
        self.assertEqual(self.mock_vllm_forward_context.get_forward_context.call_count, 2)

    def test_custom_sparse_moe_block_forward_without_metadata_for_padding(self):
        """Test custom_sparse_moe_block_forward function without _metadata_for_padding provided"""
        moe_block = self.mock_qwen3_moe.CustomSparseMoeBlock()

        moe_block.forward(
            self.mock_hidden_states,
            attn_metadata=self.mock_attn_metadata
        )

        moe_block.gate.assert_called_once_with(self.mock_hidden_states)
        moe_block.experts.assert_called_once()
        call_args = moe_block.experts.call_args
        self.assertEqual(call_args[1]['hidden_states'], self.mock_hidden_states)
        self.assertIsNotNone(call_args[1]['router_logits'])
        self.assertEqual(call_args[1]['is_prefill'], True)
        self.assertEqual(call_args[1]['top_k'], 2)
        self.assertEqual(call_args[1]['enable_force_load_balance'], False)
        self.assertIsNone(call_args[1]['shared_experts'])
        self.assertIsNone(call_args[1]['_metadata_for_padding'])
        self.assertEqual(self.mock_vllm_forward_context.get_forward_context.call_count, 1)

    def test_custom_sparse_moe_block_forward_default_params(self):
        """Test custom_sparse_moe_block_forward function with default parameters"""
        moe_block = self.mock_qwen3_moe.CustomSparseMoeBlock()

        moe_block.forward(self.mock_hidden_states)

        moe_block.gate.assert_called_once_with(self.mock_hidden_states)
        moe_block.experts.assert_called_once()
        call_args = moe_block.experts.call_args
        self.assertEqual(call_args[1]['hidden_states'], self.mock_hidden_states)
        self.assertIsNotNone(call_args[1]['router_logits'])
        self.assertEqual(call_args[1]['is_prefill'], True)
        self.assertEqual(call_args[1]['top_k'], 2)
        self.assertEqual(call_args[1]['enable_force_load_balance'], False)
        self.assertIsNone(call_args[1]['shared_experts'])
        self.assertIsNone(call_args[1]['_metadata_for_padding'])
        self.assertEqual(self.mock_vllm_forward_context.get_forward_context.call_count, 2)


if __name__ == '__main__':
    unittest.main()
