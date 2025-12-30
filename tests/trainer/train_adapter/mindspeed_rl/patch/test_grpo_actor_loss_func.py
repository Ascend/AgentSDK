#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
import sys
from unittest.mock import patch, Mock

import pytest
import torch


class TestPolicyLossInputPatch:

    @pytest.fixture()
    def patch_modules(self):
        with patch.dict(sys.modules, {"mindspeed_rl.utils.utils": Mock()}):
            yield

    def test_get_policy_loss_input_patch_without_responses_and_response_length(self, patch_modules):
        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.patch.grpo_actor_loss_func.generate_mask") as mock:
            mock.return_value = torch.tensor([[1, 1, 1], [1, 1, 1]])

            from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.grpo_actor_loss_func import \
                get_policy_loss_input_patch

            with pytest.raises(ValueError):
                get_policy_loss_input_patch(None, {})

    def test_get_policy_loss_input_patch_success(self, patch_modules):
        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.patch.grpo_actor_loss_func.generate_mask") as mock:
            value = torch.tensor([[1, 1, 0], [1, 1, 1]])

            def npu():
                return value

            value.npu = npu
            mock.return_value = value

            from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.grpo_actor_loss_func import \
                get_policy_loss_input_patch

            responses = torch.tensor([[1, 2, 3], [4, 5, 6]])
            response_length = torch.tensor([[2], [3]])

            get_policy_loss_input_patch(None,
                                        {'responses': responses,
                                         'response_length': response_length})

    def test_get_policy_loss_input_patch_with_mask_failed(self, patch_modules):
        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.patch.grpo_actor_loss_func.generate_mask") as mock:
            value = torch.tensor([[1, 1, 0], [1, 1, 1]])

            def npu():
                return value

            value.npu = npu

            from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.grpo_actor_loss_func import \
                get_policy_loss_input_patch

            # responses has only 1 dim
            responses = torch.tensor([1, 2, 3])
            response_length = torch.tensor([[2], [3]])

            mock.side_effect = AttributeError("test")
            with pytest.raises(AttributeError):
                get_policy_loss_input_patch(None, {'responses': responses,
                                                   'response_length': response_length})

            mock.side_effect = IndexError("list index out of range")
            with pytest.raises(RuntimeError):
                get_policy_loss_input_patch(None, {'responses': responses,
                                                   'response_length': response_length})

    def test_get_policy_loss_input_patch_failed_with_response_mask_dim_0(self, patch_modules):
        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.patch.grpo_actor_loss_func.generate_mask") as mock:
            value = torch.tensor([[1, 1, 0], [1, 1, 1]])

            def npu():
                return value

            value.npu = npu
            mock.return_value = value

            from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.grpo_actor_loss_func import \
                get_policy_loss_input_patch

            # responses has only 1 dim
            responses = torch.tensor([[1, 2, 3], [4, 5, 6]])
            response_length = torch.tensor([[2], [3]])
            # response_mask has dim_0 = 1, which is not same with responses
            response_mask = torch.tensor([[1, 2, 3]])

            with pytest.raises(ValueError):
                get_policy_loss_input_patch(None,
                                            {'responses': responses,
                                             'response_length': response_length,
                                             "response_mask": response_mask})

    def test_get_policy_loss_input_patch_failed_with_response_mask_dim_1(self, patch_modules):
        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.patch.grpo_actor_loss_func.generate_mask") as mock:
            value = torch.tensor([[1, 1, 0], [1, 1, 1]])

            def npu():
                return value

            value.npu = npu
            mock.return_value = value

            from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.grpo_actor_loss_func import \
                get_policy_loss_input_patch

            responses = torch.tensor([[1, 2, 3], [4, 5, 6]])
            response_length = torch.tensor([[2], [3]])
            # response_mask has dim_1 = 2, which is less than responses
            response_mask = torch.tensor([[1, 1], [1, 1]])

            with pytest.raises(ValueError):
                get_policy_loss_input_patch(None,
                                            {'responses': responses,
                                             'response_length': response_length,
                                             "response_mask": response_mask})
