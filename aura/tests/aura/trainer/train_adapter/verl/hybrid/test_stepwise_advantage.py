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
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture()
def stepwise_advantage_module(monkeypatch):
    project_root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "aura" / "aura").exists()
    )
    aura_src = str(project_root / "aura")
    if aura_src not in sys.path:
        sys.path.insert(0, aura_src)

    torch = types.ModuleType("torch")
    torch.Tensor = MagicMock
    torch.no_grad = MagicMock()
    torch.no_grad.return_value.__enter__.return_value = None
    torch.no_grad.return_value.__exit__.return_value = None
    torch.nn = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", torch)

    sys.modules.pop("aura.trainer.train_adapter.verl.hybrid.stepwise_advantage", None)
    return importlib.import_module("aura.trainer.train_adapter.verl.hybrid.stepwise_advantage")


def test_group_keys_prefer_step_indexes(stepwise_advantage_module):
    batch = SimpleNamespace(
        non_tensor_batch={
            "index_in_batch": np.array([0, 0, 1]),
            "index_in_steps": np.array([0, 1, 0]),
            "uid": np.array(["unused-0", "unused-1", "unused-2"]),
        },
        batch={"token_level_rewards": np.zeros((3, 2))},
    )

    assert stepwise_advantage_module._group_keys_from_batch(batch) == ["0_0", "0_1", "1_0"]


def test_group_keys_fall_back_to_uid_or_row_index(stepwise_advantage_module):
    uid_batch = SimpleNamespace(
        non_tensor_batch={"uid": np.array(["sample-a", "sample-b"])},
        batch={"token_level_rewards": np.zeros((2, 2))},
    )
    no_uid_batch = SimpleNamespace(
        non_tensor_batch={},
        batch={"token_level_rewards": np.zeros((3, 2))},
    )

    assert stepwise_advantage_module._group_keys_from_batch(uid_batch) == ["sample-a", "sample-b"]
    assert stepwise_advantage_module._group_keys_from_batch(no_uid_batch) == ["0", "1", "2"]
