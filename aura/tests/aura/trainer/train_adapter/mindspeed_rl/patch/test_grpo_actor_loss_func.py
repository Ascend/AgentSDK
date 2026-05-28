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
from unittest.mock import patch, MagicMock


class TestGRPOActorLossFunc:

    def test_get_policy_loss_input(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.grpo_actor_loss_func import _get_policy_loss_input
        batch = {
            'responses': torch.randn(4, 10),
            'response_length': torch.tensor([10, 10, 10, 10]),
            'old_log_prob': torch.randn(4, 10),
            'advantages': torch.randn(4, 10),
            'ref_log_prob': torch.randn(4, 10),
        }
        mock_mask = MagicMock()
        mock_mask.npu.return_value = torch.ones(4, 10)
        with patch('aura.trainer.train_adapter.mindspeed_rl.patch.grpo_actor_loss_func.generate_mask') as mock_generate:
            mock_generate.return_value = mock_mask
            response_mask, old_log_prob, advantages, ref_log_prob = _get_policy_loss_input(None, batch)
            assert response_mask is not None
            assert old_log_prob is not None
            assert advantages is not None
            assert ref_log_prob is not None

    def test_get_policy_loss_input_with_response_mask(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.grpo_actor_loss_func import _get_policy_loss_input
        batch = {
            'responses': torch.randn(4, 10),
            'response_length': torch.tensor([10, 10, 10, 10]),
            'response_mask': torch.ones(4, 10),
        }
        mock_mask = MagicMock()
        mock_mask.npu.return_value = torch.ones(4, 10)
        with patch('aura.trainer.train_adapter.mindspeed_rl.patch.grpo_actor_loss_func.generate_mask') as mock_generate:
            mock_generate.return_value = mock_mask
            response_mask, old_log_prob, advantages, ref_log_prob = _get_policy_loss_input(None, batch)
            assert response_mask is not None
            assert old_log_prob is None
            assert advantages is None
            assert ref_log_prob is None

    def test_get_policy_loss_input_missing_responses(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.grpo_actor_loss_func import _get_policy_loss_input
        batch = {
            'response_length': torch.tensor([10, 10, 10, 10]),
        }
        with pytest.raises(ValueError):
            _get_policy_loss_input(None, batch)
