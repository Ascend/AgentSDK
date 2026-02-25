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

from unittest.mock import MagicMock, patch
import pytest

from agentic_rl.trainer.train_adapter.train_registry import (
    TrainBackendRegistry,
    get_train_fn,
    register_train_fn,
    train_backend_registry,
)


class TestTrainBackendRegistry:
    """Tests for TrainBackendRegistry class."""

    @staticmethod
    def test_register_accepts_remote_function():
        """Registering a callable with .remote attribute succeeds."""
        registry = TrainBackendRegistry()
        mock_fn = MagicMock()
        mock_fn.remote = MagicMock()
        registry.register("test_backend", mock_fn)
        assert registry.get("test_backend") is mock_fn

    @staticmethod
    def test_register_rejects_non_remote_function():
        """Registering an object without .remote raises ValueError."""
        registry = TrainBackendRegistry()
        no_remote = MagicMock(spec=[])  # no .remote
        del no_remote.remote
        with pytest.raises(ValueError, match=r"train_fn test_backend is not a remote function"):
            registry.register("test_backend", no_remote)
        assert registry.get("test_backend") is None

    @staticmethod
    def test_register_overwrites_existing():
        """Registering the same name twice overwrites the first."""
        registry = TrainBackendRegistry()
        fn1 = MagicMock()
        fn1.remote = MagicMock()
        fn2 = MagicMock()
        fn2.remote = MagicMock()
        registry.register("backend", fn1)
        registry.register("backend", fn2)
        assert registry.get("backend") is fn2

    @staticmethod
    def test_get_returns_none_for_missing_name():
        """get() returns None for unregistered name."""
        registry = TrainBackendRegistry()
        assert registry.get("nonexistent") is None

    @staticmethod
    def test_get_returns_registered_function():
        """get() returns the previously registered function."""
        registry = TrainBackendRegistry()
        mock_fn = MagicMock()
        mock_fn.remote = MagicMock()
        registry.register("my_backend", mock_fn)
        assert registry.get("my_backend") is mock_fn


class TestRegisterTrainFn:
    """Tests for register_train_fn()."""

    @staticmethod
    def test_register_train_fn_unsupported_raises():
        """Unsupported backend name raises ValueError and logs error."""
        with patch.object(train_backend_registry, "get", return_value=None):
            with pytest.raises(ValueError, match=r"train_backend unsupported_backend is not supported"):
                register_train_fn("unsupported_backend")

    @staticmethod
    def test_register_train_fn_already_registered_logs_warning_and_returns():
        """If backend is already registered, logs warning and returns without re-registering."""
        mock_registered = MagicMock()
        with patch.object(train_backend_registry, "get", return_value=mock_registered):
            with patch.object(train_backend_registry, "register") as mock_register:
                register_train_fn("mindspeed_rl")
                mock_register.assert_not_called()

    @staticmethod
    def test_register_train_fn_mindspeed_rl_success():
        """register_train_fn('mindspeed_rl') registers the train function when import is patched."""
        import sys

        import agentic_rl.trainer.train_adapter.train_registry as reg_mod

        mock_train = MagicMock()
        mock_train.remote = MagicMock()
        fake_mod = MagicMock()
        fake_mod.train = mock_train
        mod_name = "agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo"
        with patch.object(train_backend_registry, "get", return_value=None):
            with patch.object(train_backend_registry, "register") as mock_register:
                with patch("agentic_rl.trainer.train_adapter.train_registry.logger"):
                    sys.modules[mod_name] = fake_mod
                    try:
                        reg_mod.register_train_fn("mindspeed_rl")
                        mock_register.assert_called_once_with("mindspeed_rl", mock_train)
                    finally:
                        sys.modules.pop(mod_name, None)

    @staticmethod
    def test_register_train_fn_verl_success():
        """register_train_fn('verl') registers the train function when import is patched."""
        import sys

        import agentic_rl.trainer.train_adapter.train_registry as reg_mod

        mock_train = MagicMock()
        mock_train.remote = MagicMock()
        fake_mod = MagicMock()
        fake_mod.train = mock_train
        mod_name = "agentic_rl.trainer.train_adapter.verl.train_agent_grpo"
        with patch.object(train_backend_registry, "get", return_value=None):
            with patch.object(train_backend_registry, "register") as mock_register:
                with patch("agentic_rl.trainer.train_adapter.train_registry.logger"):
                    sys.modules[mod_name] = fake_mod
                    try:
                        reg_mod.register_train_fn("verl")
                        mock_register.assert_called_once_with("verl", mock_train)
                    finally:
                        sys.modules.pop(mod_name, None)


class TestGetTrainFn:
    """Tests for get_train_fn()."""

    @staticmethod
    def test_get_train_fn_unsupported_raises():
        """get_train_fn() with unsupported name raises ValueError."""
        with patch.object(train_backend_registry, "get", return_value=None):
            with pytest.raises(ValueError, match=r"train_backend bad_backend is not supported"):
                get_train_fn("bad_backend")

    @staticmethod
    def test_get_train_fn_calls_register_and_returns_registered():
        """get_train_fn() calls register_train_fn and returns the backend from registry."""
        mock_fn = MagicMock()
        mock_fn.remote = MagicMock()
        with patch(
            "agentic_rl.trainer.train_adapter.train_registry.register_train_fn",
        ) as mock_register_train_fn:
            with patch.object(
                train_backend_registry, "get", return_value=mock_fn
            ):
                with patch(
                    "agentic_rl.trainer.train_adapter.train_registry.logger",
                ):
                    result = get_train_fn("mindspeed_rl")
                    mock_register_train_fn.assert_called_once_with("mindspeed_rl")
                    assert result is mock_fn

    @staticmethod
    def test_get_train_fn_logs_info():
        """get_train_fn() logs that the train_backend is used."""
        mock_fn = MagicMock()
        mock_fn.remote = MagicMock()
        with patch(
            "agentic_rl.trainer.train_adapter.train_registry.register_train_fn",
        ):
            with patch.object(train_backend_registry, "get", return_value=mock_fn):
                with patch(
                    "agentic_rl.trainer.train_adapter.train_registry.logger",
                ) as mock_logger:
                    get_train_fn("verl")
                    mock_logger.info.assert_called_once()
                    call_args = mock_logger.info.call_args[0][0]
                    assert "verl" in call_args and "used" in call_args
