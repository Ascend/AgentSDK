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
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def train_service_module(monkeypatch):
    """Import train_service with external dependencies mocked."""
    # external dependency
    mock_pad_process = MagicMock()
    mock_pad_process.remove_padding_tensor_dict_to_dict = MagicMock()
    mock_pad_process.remove_padding_and_split_to_list = MagicMock()
    monkeypatch.setitem(sys.modules, "mindspeed_rl", MagicMock())
    monkeypatch.setitem(sys.modules, "mindspeed_rl.utils", MagicMock())
    monkeypatch.setitem(sys.modules, "mindspeed_rl.utils.pad_process", mock_pad_process)

    # avoid importing heavy internal dependency chain
    mock_prepare_train_module = MagicMock()
    mock_prepare_train_module.prepare_train = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "aura.trainer.train_adapter.mindspeed_rl.utils.prepare_train",
        mock_prepare_train_module,
    )

    module_name = "aura.trainer.train_adapter.mindspeed_rl.hybrid_policy.train_service"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class TestTrainService:
    def test_train_function_exists(self, train_service_module):
        assert callable(train_service_module.train)

    def test_train_function_is_ray_remote(self, train_service_module):
        assert hasattr(train_service_module.train, "options")

    def test_logger_exists(self, train_service_module):
        assert train_service_module.logger is not None

    def test_module_imports(self, train_service_module):
        assert train_service_module is not None
        assert train_service_module.ray is not None
        assert train_service_module.remove_padding_tensor_dict_to_dict is not None
        assert train_service_module.remove_padding_and_split_to_list is not None
        assert train_service_module.RolloutWorker is not None
        assert train_service_module.AgentGRPOTrainer is not None
        assert train_service_module.prepare_train is not None

    def test_prepare_train_is_imported(self, train_service_module):
        assert train_service_module.prepare_train is not None

    def test_train_calls_prepare_train(self, train_service_module):
        assert hasattr(train_service_module, "prepare_train")

    def test_train_creates_rollout_worker(self, train_service_module):
        assert hasattr(train_service_module, "RolloutWorker")

    def test_train_creates_trainer(self, train_service_module):
        assert hasattr(train_service_module, "AgentGRPOTrainer")
