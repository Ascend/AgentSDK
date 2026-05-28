#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

        http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import pytest
import torch
import sys
from unittest.mock import MagicMock, patch


class TestCpOptUtils:

    def test_accumulate_list_empty(self):
        utils_module = __import__('aura.trainer.train_adapter.mindspeed_rl.cp_opt.utils', fromlist=['accumulate_list', '_ACCUMULATE_LIST_CACHE_LRU'])

        utils_module.torch = torch
        mock_prev_attn_out = MagicMock()
        mock_prev_attn_out.device = torch.device('cpu')
        utils_module.prev_attn_out = mock_prev_attn_out

        utils_module._ACCUMULATE_LIST_CACHE_LRU.clear()

        result = utils_module.accumulate_list([])

        assert result is not None
        assert result.dtype == torch.int64
        assert result.device == torch.device('cpu')
        assert len(result) == 1
        assert result[0].item() == 0

    def test_accumulate_list_with_values(self):
        utils_module = __import__('aura.trainer.train_adapter.mindspeed_rl.cp_opt.utils', fromlist=['accumulate_list', '_ACCUMULATE_LIST_CACHE_LRU'])

        utils_module.torch = torch
        mock_prev_attn_out = MagicMock()
        mock_prev_attn_out.device = torch.device('cpu')
        utils_module.prev_attn_out = mock_prev_attn_out

        utils_module._ACCUMULATE_LIST_CACHE_LRU.clear()

        result = utils_module.accumulate_list([1, 2, 3])

        assert result is not None
        assert result.dtype == torch.int64
        assert torch.equal(result, torch.tensor([0, 1, 3, 6], dtype=torch.int64))

    def test_accumulate_list_cache(self):
        utils_module = __import__('aura.trainer.train_adapter.mindspeed_rl.cp_opt.utils', fromlist=['accumulate_list', '_ACCUMULATE_LIST_CACHE_LRU'])

        utils_module.torch = torch
        mock_prev_attn_out = MagicMock()
        mock_prev_attn_out.device = torch.device('cpu')
        utils_module.prev_attn_out = mock_prev_attn_out

        utils_module._ACCUMULATE_LIST_CACHE_LRU.clear()

        result1 = utils_module.accumulate_list([1, 2, 3])
        result2 = utils_module.accumulate_list([1, 2, 3])

        assert result1 is result2
        assert len(utils_module._ACCUMULATE_LIST_CACHE_LRU) == 1

    def test_accumulate_list_cache_clear(self):
        utils_module = __import__('aura.trainer.train_adapter.mindspeed_rl.cp_opt.utils', fromlist=['accumulate_list', '_ACCUMULATE_LIST_CACHE_LRU'])

        utils_module.torch = torch
        mock_prev_attn_out = MagicMock()
        mock_prev_attn_out.device = torch.device('cpu')
        utils_module.prev_attn_out = mock_prev_attn_out

        utils_module._ACCUMULATE_LIST_CACHE_LRU.clear()

        utils_module.accumulate_list([1, 2, 3])
        utils_module.accumulate_list([4, 5, 6])

        assert len(utils_module._ACCUMULATE_LIST_CACHE_LRU) == 1

    def test_get_selection_indices_for_tnd_softmax_update_basic(self):
        utils_module = __import__('aura.trainer.train_adapter.mindspeed_rl.cp_opt.utils', fromlist=['get_selection_indices_for_tnd_softmax_update', '_SOFTMAX_INDICES_CACHE_LRU'])

        utils_module.torch = torch
        mock_torch_npu = MagicMock()
        mock_torch_npu.empty_cache = MagicMock()
        utils_module.torch_npu = mock_torch_npu

        mock_npu_module = MagicMock()
        mock_npu_module.current_device.return_value = torch.device('cpu')
        setattr(torch, 'npu', mock_npu_module)

        utils_module._SOFTMAX_INDICES_CACHE_LRU.clear()

        try:
            sub_seq_len = torch.tensor([2, 3])
            result = utils_module.get_selection_indices_for_tnd_softmax_update(10, 2, sub_seq_len)

            assert result is not None
            assert result.dtype == torch.long
            assert len(result) > 0
        finally:
            if hasattr(torch, 'npu') and getattr(torch, 'npu') == mock_npu_module:
                delattr(torch, 'npu')

    def test_get_selection_indices_for_tnd_softmax_update_empty(self):
        utils_module = __import__('aura.trainer.train_adapter.mindspeed_rl.cp_opt.utils', fromlist=['get_selection_indices_for_tnd_softmax_update', '_SOFTMAX_INDICES_CACHE_LRU'])

        utils_module.torch = torch
        mock_torch_npu = MagicMock()
        mock_torch_npu.empty_cache = MagicMock()
        utils_module.torch_npu = mock_torch_npu

        mock_npu_module = MagicMock()
        mock_npu_module.current_device.return_value = torch.device('cpu')
        setattr(torch, 'npu', mock_npu_module)

        utils_module._SOFTMAX_INDICES_CACHE_LRU.clear()

        try:
            sub_seq_len = torch.tensor([])
            result = utils_module.get_selection_indices_for_tnd_softmax_update(10, 2, sub_seq_len)

            assert result is not None
            assert len(result) == 0
        finally:
            if hasattr(torch, 'npu') and getattr(torch, 'npu') == mock_npu_module:
                delattr(torch, 'npu')

    def test_get_selection_indices_for_tnd_softmax_update_cache(self):
        utils_module = __import__('aura.trainer.train_adapter.mindspeed_rl.cp_opt.utils', fromlist=['get_selection_indices_for_tnd_softmax_update', '_SOFTMAX_INDICES_CACHE_LRU'])

        utils_module.torch = torch
        mock_torch_npu = MagicMock()
        mock_torch_npu.empty_cache = MagicMock()
        utils_module.torch_npu = mock_torch_npu

        mock_npu_module = MagicMock()
        mock_npu_module.current_device.return_value = torch.device('cpu')
        setattr(torch, 'npu', mock_npu_module)

        utils_module._SOFTMAX_INDICES_CACHE_LRU.clear()

        try:
            sub_seq_len = torch.tensor([2, 3])
            result1 = utils_module.get_selection_indices_for_tnd_softmax_update(10, 2, sub_seq_len)
            result2 = utils_module.get_selection_indices_for_tnd_softmax_update(10, 2, sub_seq_len)

            assert result1 is result2
            assert len(utils_module._SOFTMAX_INDICES_CACHE_LRU) == 1
        finally:
            if hasattr(torch, 'npu') and getattr(torch, 'npu') == mock_npu_module:
                delattr(torch, 'npu')

    def test_get_selection_indices_for_tnd_softmax_update_cache_cleanup(self):
        utils_module = __import__('aura.trainer.train_adapter.mindspeed_rl.cp_opt.utils', fromlist=['get_selection_indices_for_tnd_softmax_update', '_SOFTMAX_INDICES_CACHE_LRU'])

        utils_module.torch = torch
        mock_torch_npu = MagicMock()
        mock_torch_npu.empty_cache = MagicMock()
        utils_module.torch_npu = mock_torch_npu

        mock_npu_module = MagicMock()
        mock_npu_module.current_device.return_value = torch.device('cpu')
        setattr(torch, 'npu', mock_npu_module)

        utils_module._SOFTMAX_INDICES_CACHE_LRU.clear()

        try:
            sub_seq_len1 = torch.tensor([2, 3])
            sub_seq_len2 = torch.tensor([4, 5])

            utils_module.get_selection_indices_for_tnd_softmax_update(10, 2, sub_seq_len1)
            utils_module.get_selection_indices_for_tnd_softmax_update(10, 2, sub_seq_len2)

            assert len(utils_module._SOFTMAX_INDICES_CACHE_LRU) == 1
            mock_torch_npu.empty_cache.assert_called_once()
        finally:
            if hasattr(torch, 'npu') and getattr(torch, 'npu') == mock_npu_module:
                delattr(torch, 'npu')
