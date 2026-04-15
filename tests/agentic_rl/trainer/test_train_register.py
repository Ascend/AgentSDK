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

import sys
import importlib
import pytest
from unittest.mock import MagicMock, patch


class TestTrainRegistry:
    """
    Tests for the TrainRegistry class.

    Goals:
    - Prevent sys.modules contamination across pytest files
    - Ensure registry initialization executes correctly
    - Ensure coverage metrics are accurate
    """

    def setup_method(self):
        """
        Runs before each test: isolate sys.modules to prevent side effects.
        """
        self.original_modules = sys.modules.copy()
        mock_loggers_module = MagicMock()
        self.mock_logger = MagicMock()
        mock_loggers_module.Loggers.return_value.get_logger.return_value = self.mock_logger
        self.mocked_modules = {
            "agentic_rl.base.log.loggers": mock_loggers_module,
            "agentic_rl.trainer.rollout.rollout_main": MagicMock(),
            "agentic_rl.trainer.train_adapter.mindspeed_rl.hybrid_policy.train_service": MagicMock(),
            "agentic_rl.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train.train_service": MagicMock(),
            "agentic_rl.trainer.train_adapter.verl.full_async.train_main": MagicMock(),
            "agentic_rl.trainer.train_adapter.verl.hybrid.train_main": MagicMock(),
            "agentic_rl.trainer.train_adapter.omni_rl.hybrid.train_main": MagicMock(),
        }
        self.module_patcher = patch.dict(sys.modules, self.mocked_modules)
        self.module_patcher.start()

        import agentic_rl.trainer.train_register as train_register
        importlib.reload(train_register)
        self.train_register = train_register

    def teardown_method(self):
        """
        Runs after each test: restore sys.modules to original state.
        """
        patch.stopall()
        added_keys = set(sys.modules.keys()) - set(self.original_modules.keys())
        for key in added_keys:
            sys.modules.pop(key, None)
        sys.modules.update(self.original_modules)

    def test_initialization(self):
        """
        Test that TrainRegistry initializes correctly.
        """
        TrainRegistry = self.train_register.TrainRegistry
        registry_instance = TrainRegistry()
        assert hasattr(registry_instance, "_registry")
        assert isinstance(registry_instance._registry, dict)
        assert len(registry_instance._registry) == 0

    def test_register_and_get_method(self):
        """
        Test register and get_method functionality.
        """
        TrainRegistry = self.train_register.TrainRegistry
        registry_instance = TrainRegistry()

        rollout_mock = MagicMock()
        train_mock = MagicMock()

        registry_instance.register("engine1", "mode1", rollout_mock, train_mock)

        assert ("engine1", "mode1") in registry_instance._registry
        assert registry_instance.get_method("engine1", "mode1") == (rollout_mock, train_mock)
        assert registry_instance.get_method("non_exist", "mode1") is None

    def test_registry_singleton(self):
        """
        Ensure the registry instance is a singleton.
        """
        first_instance = self.train_register.registry
        second_instance = self.train_register.registry
        assert first_instance is second_instance

    def test_registry_default_entries(self):
        """
        Check that default registry entries exist.
        """
        registry_instance = self.train_register.registry
        default_keys = [
            ("mindspeed_rl", "hybrid"),
            ("mindspeed_rl", "one_step_off"),
            ("mindspeed_rl", "dummy_train"),
            ("verl", "hybrid"),
            ("verl", "one_step_off"),
            ("omni_rl", "hybrid"),
        ]
        for key in default_keys:
            assert registry_instance.get_method(*key) is not None
        assert len(registry_instance._registry) == len(default_keys)

    @pytest.mark.parametrize(
        "failed_modules,expected_messages",
        [
            (["agentic_rl.trainer.rollout.rollout_main"], ["verl/mindspeed_rl hybrid train is not available"]),
            (["agentic_rl.trainer.train_adapter.omni_rl.hybrid.train_main"], ["omni_rl hybrid train is not available"]),
        ],
    )
    def test_import_error_handling(self, failed_modules, expected_messages):
        """
        Test branches triggered by ImportError.
        """
        mocked_modules = {module: None for module in failed_modules}
        self.mock_logger.reset_mock()
        with patch.dict(sys.modules, mocked_modules, clear=False):
            importlib.reload(self.train_register)
            warning_messages = [str(call) for call in self.mock_logger.warning.call_args_list]
            assert any(
                any(msg in warning_text for msg in expected_messages)
                for warning_text in warning_messages
            )

    @pytest.mark.parametrize(
        "initial_entries,expected_count",
        [
            ({}, 0),
            ({("test_engine", "mode"): (None, None)}, 1),
        ],
    )
    def test_registry_clear(self, initial_entries, expected_count):
        """
        Test clearing of the registry.
        """
        TrainRegistry = self.train_register.TrainRegistry
        registry_instance = TrainRegistry()
        registry_instance._registry.update(initial_entries)
        registry_instance._registry.clear()
        assert len(registry_instance._registry) == 0