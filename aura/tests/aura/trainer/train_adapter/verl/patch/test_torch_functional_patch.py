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

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

import torch


class TestTorchFunctionalPatch:

    def setup_method(self):
        self.original_torch_npu = sys.modules.get("torch_npu")
        self.mock_torch_npu = ModuleType("torch_npu")
        self.mock_torch_npu.npu_cross_entropy_loss = MagicMock()
        sys.modules["torch_npu"] = self.mock_torch_npu

        import aura.trainer.train_adapter.verl.patch.torch_functional_patch as target_module

        self.target_module = importlib.reload(target_module)

    def teardown_method(self):
        if self.original_torch_npu is None:
            sys.modules.pop("torch_npu", None)
        else:
            sys.modules["torch_npu"] = self.original_torch_npu

    def test_logprobs_without_repeat_labels(self):
        logits = torch.randn(2, 3, 5)
        labels = torch.tensor([1, 2, 3, 0, 1, 2])

        fake_loss = torch.arange(6, dtype=torch.float32)
        self.mock_torch_npu.npu_cross_entropy_loss.return_value = (fake_loss, None, None, None)

        out = self.target_module.logprobs_from_logits_torch_npu_patch(logits, labels)

        called_logits, called_labels = self.mock_torch_npu.npu_cross_entropy_loss.call_args.args[:2]
        assert called_logits.shape == (6, 5)
        assert torch.equal(called_labels, labels.reshape(-1))

        expected = -fake_loss.view(2, 3)
        assert out.shape == (2, 3)
        assert torch.equal(out, expected)

    def test_logprobs_repeat_labels_when_shape_mismatch(self):
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, 2])

        fake_loss = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
        self.mock_torch_npu.npu_cross_entropy_loss.return_value = (fake_loss, None, None, None)

        out = self.target_module.logprobs_from_logits_torch_npu_patch(logits, labels)

        called_logits, called_labels = self.mock_torch_npu.npu_cross_entropy_loss.call_args.args[:2]
        assert called_logits.shape == (4, 3)
        assert torch.equal(called_labels, torch.tensor([0, 2, 0, 2]))

        expected = -fake_loss.view(4)
        assert out.shape == (4,)
        assert torch.equal(out, expected)
