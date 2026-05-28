#!/usr/bin/env python3S
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import pytest
import torch
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

_PATCH_MODULE = 'aura.trainer.train_adapter.mindspeed_rl.cp_opt.dot_product_attention'


class TestDotProductAttention:

    def _make_args(self, context_parallel_algo='megatron_cp_algo', use_cp_send_recv_overlap=False):
        mock_args = MagicMock()
        mock_args.shape_order = "TND"
        mock_args.cp_attention_mask_type = 'causal'
        mock_args.context_parallel_algo = context_parallel_algo
        mock_args.use_cp_send_recv_overlap = use_cp_send_recv_overlap
        return mock_args

    def _patch_mpu(self, stack, cp_world_size=1, cp_rank=0, cp_global_ranks=None, send_recv_overlap_group=None):
        mock_mpu = stack.enter_context(patch(f'{_PATCH_MODULE}.mpu'))
        mock_mpu.get_context_parallel_group.return_value = MagicMock()
        mock_mpu.get_context_parallel_world_size.return_value = cp_world_size
        mock_mpu.get_context_parallel_rank.return_value = cp_rank
        mock_mpu.get_context_parallel_global_ranks.return_value = cp_global_ranks or [0]
        mock_mpu.get_context_parallel_group_for_send_recv_overlap.return_value = send_recv_overlap_group
        return mock_mpu

    def _patch_ring_ranks(self, stack, intra_ranks=None, inter_kv_ranks=None, inter_dkv_ranks=None):
        stack.enter_context(patch(f'{_PATCH_MODULE}.get_ring_ranks_for_intra_window', return_value=intra_ranks or [0]))
        stack.enter_context(patch(f'{_PATCH_MODULE}.get_ring_ranks_for_inter_window_kv', return_value=inter_kv_ranks or [0]))
        stack.enter_context(patch(f'{_PATCH_MODULE}.get_ring_ranks_for_inter_window_dkv', return_value=inter_dkv_ranks or [0]))

    def _patch_ring_groups(self, stack, intra_group=None, send_recv_overlap_group=None):
        stack.enter_context(patch(f'{_PATCH_MODULE}.get_ring_group_for_intra_window', return_value=intra_group or MagicMock()))
        stack.enter_context(patch(f'{_PATCH_MODULE}.get_ring_group_for_intra_window_send_recv_overlap', return_value=send_recv_overlap_group))

    def test_do_ring_context_parallel_megatron_cp_algo(self):
        from aura.trainer.train_adapter.mindspeed_rl.cp_opt.dot_product_attention import do_ring_context_parallel

        mock_q = MagicMock()
        mock_k = MagicMock()
        mock_v = MagicMock()
        head_num = 16
        softmax_scale = 1.0
        attn_mask = None

        with ExitStack() as stack:
            stack.enter_context(patch(f'{_PATCH_MODULE}.get_args', return_value=self._make_args()))
            stack.enter_context(patch(f'{_PATCH_MODULE}.get_context_parallel_group_for_hybrid_ring', return_value=None))
            self._patch_mpu(stack, send_recv_overlap_group=None)
            self._patch_ring_ranks(stack)
            self._patch_ring_groups(stack, send_recv_overlap_group=None)
            mock_ringattn = stack.enter_context(patch(f'{_PATCH_MODULE}.ringattn_context_parallel', return_value=MagicMock()))

            result = do_ring_context_parallel(mock_q, mock_k, mock_v, head_num, softmax_scale, attn_mask)

            assert result is not None
            mock_ringattn.assert_called_once()

    def test_do_ring_context_parallel_hybrid_cp_algo(self):
        from aura.trainer.train_adapter.mindspeed_rl.cp_opt.dot_product_attention import do_ring_context_parallel

        mock_q = MagicMock()
        mock_k = MagicMock()
        mock_v = MagicMock()
        head_num = 16
        softmax_scale = 1.0
        attn_mask = None

        with ExitStack() as stack:
            stack.enter_context(patch(f'{_PATCH_MODULE}.get_args', return_value=self._make_args(
                context_parallel_algo='hybrid_cp_algo', use_cp_send_recv_overlap=True)))
            stack.enter_context(patch(f'{_PATCH_MODULE}.get_context_parallel_group_for_hybrid_ring', return_value=None))
            self._patch_mpu(stack, cp_world_size=2, cp_global_ranks=[0, 1], send_recv_overlap_group=MagicMock())
            self._patch_ring_ranks(stack, intra_ranks=[0, 1], inter_kv_ranks=[0, 1], inter_dkv_ranks=[0, 1])
            self._patch_ring_groups(stack, send_recv_overlap_group=MagicMock())
            mock_ringattn = stack.enter_context(patch(f'{_PATCH_MODULE}.ringattn_context_parallel', return_value=MagicMock()))

            result = do_ring_context_parallel(mock_q, mock_k, mock_v, head_num, softmax_scale, attn_mask)

            assert result is not None
            mock_ringattn.assert_called_once()

    def test_do_ring_context_parallel_adaptive_algo(self):
        from aura.trainer.train_adapter.mindspeed_rl.cp_opt.dot_product_attention import do_ring_context_parallel

        mock_q = MagicMock()
        mock_k = MagicMock()
        mock_v = MagicMock()
        head_num = 16
        softmax_scale = 1.0
        attn_mask = None

        with ExitStack() as stack:
            stack.enter_context(patch(f'{_PATCH_MODULE}.get_args', return_value=self._make_args(context_parallel_algo='adaptive')))
            stack.enter_context(patch(f'{_PATCH_MODULE}.get_context_parallel_group_for_hybrid_ring', return_value=None))
            self._patch_mpu(stack)
            stack.enter_context(patch(f'{_PATCH_MODULE}.get_scheduling_info', return_value=MagicMock()))
            mock_adaptive = stack.enter_context(patch(f'{_PATCH_MODULE}.adaptive_attn_context_parallel', return_value=MagicMock()))

            result = do_ring_context_parallel(mock_q, mock_k, mock_v, head_num, softmax_scale, attn_mask)

            assert result is not None
            mock_adaptive.assert_called_once()

    def test_do_ring_context_parallel_hybrid_mode(self):
        from aura.trainer.train_adapter.mindspeed_rl.cp_opt.dot_product_attention import do_ring_context_parallel

        mock_q = MagicMock()
        mock_k = MagicMock()
        mock_v = MagicMock()
        head_num = 16
        softmax_scale = 1.0
        attn_mask = None

        hybrid_group = MagicMock()

        with ExitStack() as stack:
            stack.enter_context(patch(f'{_PATCH_MODULE}.get_args', return_value=self._make_args()))
            mock_hybrid_group = stack.enter_context(patch(f'{_PATCH_MODULE}.get_context_parallel_group_for_hybrid_ring'))
            mock_hybrid_group.side_effect = [hybrid_group, hybrid_group]
            stack.enter_context(patch(f'{_PATCH_MODULE}.get_context_parallel_for_hybrid_ring_world_size', return_value=2))
            stack.enter_context(patch(f'{_PATCH_MODULE}.get_context_parallel_for_hybrid_ring_rank', return_value=0))
            stack.enter_context(patch(f'{_PATCH_MODULE}.get_context_parallel_for_hybrid_ring_global_ranks', return_value=[0, 1]))
            self._patch_ring_ranks(stack, intra_ranks=[0, 1], inter_kv_ranks=[0, 1], inter_dkv_ranks=[0, 1])
            self._patch_ring_groups(stack, send_recv_overlap_group=None)
            mock_ringattn = stack.enter_context(patch(f'{_PATCH_MODULE}.ringattn_context_parallel', return_value=MagicMock()))

            result = do_ring_context_parallel(mock_q, mock_k, mock_v, head_num, softmax_scale, attn_mask)

            assert result is not None
            mock_ringattn.assert_called_once()

    def test_do_ring_context_parallel_packed_seq_params(self):
        dp_module = __import__('aura.trainer.train_adapter.mindspeed_rl.cp_opt.dot_product_attention', fromlist=['do_ring_context_parallel'])
        dp_module.torch = torch

        mock_q = MagicMock()
        mock_k = MagicMock()
        mock_v = MagicMock()
        head_num = 16
        softmax_scale = 1.0
        attn_mask = None

        mock_packed_seq_params = MagicMock()
        mock_packed_seq_params.cu_seqlens_q = torch.tensor([0, 10, 20])
        mock_packed_seq_params.cu_seqlens_kv = torch.tensor([0, 10, 20])

        with ExitStack() as stack:
            stack.enter_context(patch(f'{_PATCH_MODULE}.get_args', return_value=self._make_args()))
            stack.enter_context(patch(f'{_PATCH_MODULE}.get_context_parallel_group_for_hybrid_ring', return_value=None))
            self._patch_mpu(stack, send_recv_overlap_group=None)
            self._patch_ring_ranks(stack)
            self._patch_ring_groups(stack, send_recv_overlap_group=None)
            mock_ringattn = stack.enter_context(patch(f'{_PATCH_MODULE}.ringattn_context_parallel', return_value=MagicMock()))

            result = dp_module.do_ring_context_parallel(mock_q, mock_k, mock_v, head_num, softmax_scale, attn_mask, packed_seq_params=mock_packed_seq_params)

            assert result is not None
            mock_ringattn.assert_called_once()
