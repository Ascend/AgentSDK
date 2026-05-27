#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

import sys
import types
import os
import time
from unittest.mock import MagicMock, patch
import pytest

# ---------------------------------------------------------------------------
# Fixture: fake module tree for data_analysis
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_env():
    """Build an isolated fake module tree and return the module under test."""

    # ---- fake torch ----
    fake_torch = types.ModuleType("torch")
    class FakeTensor:
        def tolist(self):
            return [1, 2, 3]  # stub return
    fake_torch.Tensor = FakeTensor
    fake_torch.save = MagicMock()

    # ---- fake loggers module (aura.base.log.loggers) ----
    fake_loggers_mod = types.ModuleType("aura.base.log.loggers")
    fake_logger_instance = MagicMock()
    fake_loggers_mod.Loggers = MagicMock()
    fake_loggers_mod.Loggers.return_value.get_logger.return_value = fake_logger_instance

    # ---- aura package structure to locate the real data_analysis module ----
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_log = types.ModuleType("aura.base.log")
    fake_aura_base_log.__path__ = [os.path.join(base, "base/log")]
    fake_aura_base_log_loggers = fake_loggers_mod
    fake_aura_base_analysis = types.ModuleType("aura.base.analysis")
    fake_aura_base_analysis.__path__ = [os.path.join(base, "base/analysis")]

    fakes = {
        "torch": fake_torch,
        "aura": fake_aura,
        "aura.base": fake_aura_base,
        "aura.base.log": fake_aura_base_log,
        "aura.base.log.loggers": fake_aura_base_log_loggers,
        "aura.base.analysis": fake_aura_base_analysis,
    }

    target = "aura.base.analysis.data_analysis"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.base.analysis.data_analysis as mod
        yield {
            "mod": mod,
            "fake_torch": fake_torch,
            "fake_logger": fake_logger_instance,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestEnableDataDebug:
    def test_true_lowercase(self, fake_env):
        mod = fake_env["mod"]
        with patch.dict(os.environ, {"ENABLE_DATA_DEBUG": "true"}):
            assert mod.enable_data_debug() is True

    def test_false_when_unset(self, fake_env):
        mod = fake_env["mod"]
        with patch.dict(os.environ, {}, clear=True):
            assert mod.enable_data_debug() is False

    def test_false_when_other_value(self, fake_env):
        mod = fake_env["mod"]
        with patch.dict(os.environ, {"ENABLE_DATA_DEBUG": "false"}):
            assert mod.enable_data_debug() is False

    def test_true_case_insensitive(self, fake_env):
        mod = fake_env["mod"]
        with patch.dict(os.environ, {"ENABLE_DATA_DEBUG": "True"}):
            assert mod.enable_data_debug() is True


class TestGetPath:
    def test_creates_directory_if_missing(self, fake_env):
        mod = fake_env["mod"]
        with patch("os.path.exists", return_value=False) as mock_exists, \
             patch("os.makedirs") as mock_makedirs:
            result = mod.get_path()
            mock_exists.assert_called_once_with("./data_analysis/")
            mock_makedirs.assert_called_once_with("./data_analysis/")
            assert result == "./data_analysis/"

    def test_no_create_when_directory_exists(self, fake_env):
        mod = fake_env["mod"]
        with patch("os.path.exists", return_value=True) as mock_exists, \
             patch("os.makedirs") as mock_makedirs:
            result = mod.get_path()
            mock_exists.assert_called_once_with("./data_analysis/")
            mock_makedirs.assert_not_called()
            assert result == "./data_analysis/"


class TestGetCurrentTimestamp:
    def test_format(self, fake_env):
        mod = fake_env["mod"]
        fake_time_struct = time.struct_time((2026, 5, 23, 10, 30, 0, 0, 0, 0))
        with patch("time.localtime", return_value=fake_time_struct):
            ts = mod.get_current_timestamp()
            assert ts == "20260523_103000"


class TestTorchSaveData:
    def test_debug_disabled_returns_immediately(self, fake_env):
        mod = fake_env["mod"]
        fake_torch = fake_env["fake_torch"]
        fake_logger = fake_env["fake_logger"]
        with patch.object(mod, "enable_data_debug", return_value=False):
            mod.torch_save_data("data", "prefix")
        fake_torch.save.assert_not_called()
        fake_logger.info.assert_not_called()

    def test_debug_enabled_saves_and_logs(self, fake_env):
        mod = fake_env["mod"]
        fake_torch = fake_env["fake_torch"]
        fake_logger = fake_env["fake_logger"]
        with patch.object(mod, "enable_data_debug", return_value=True), \
             patch.object(mod, "get_path", return_value="/fake/path/"), \
             patch.object(mod, "get_current_timestamp", return_value="20260523_103000"), \
             patch("os.path.join", return_value="/fake/path/prefix_0_20260523_103000.pt") as mock_join:
            mod.torch_save_data("data", "prefix")
            mock_join.assert_called_once_with("/fake/path/", "prefix_0_20260523_103000.pt")
            fake_torch.save.assert_called_once_with("data", "/fake/path/prefix_0_20260523_103000.pt")
            fake_logger.info.assert_called_once_with(
                "torch save data to /fake/path/prefix_0_20260523_103000.pt done"
            )


class TestConvertToString:
    def test_tensor(self, fake_env):
        mod = fake_env["mod"]
        from aura.base.analysis.data_analysis import torch as fake_torch
        tensor = fake_torch.Tensor()
        result = mod.convert_to_string(tensor)
        assert result == str(tensor.tolist())

    def test_list(self, fake_env):
        mod = fake_env["mod"]
        from aura.base.analysis.data_analysis import torch as fake_torch
        tensor = fake_torch.Tensor()
        value = [tensor, 42, "hello"]
        result = mod.convert_to_string(value)
        assert isinstance(result, list)
        assert result == [str(tensor.tolist()), "42", "hello"]

    def test_dict(self, fake_env):
        mod = fake_env["mod"]
        from aura.base.analysis.data_analysis import torch as fake_torch
        tensor = fake_torch.Tensor()
        value = {"a": tensor, "b": 3.14}
        result = mod.convert_to_string(value)
        assert isinstance(result, dict)
        assert result == {"a": str(tensor.tolist()), "b": "3.14"}

    def test_nested_structure(self, fake_env):
        mod = fake_env["mod"]
        from aura.base.analysis.data_analysis import torch as fake_torch
        tensor = fake_torch.Tensor()
        value = {"outer": [tensor, {"inner": tensor}]}
        result = mod.convert_to_string(value)
        expected = {"outer": [str(tensor.tolist()), {"inner": str(tensor.tolist())}]}
        assert result == expected

    def test_plain_values(self, fake_env):
        mod = fake_env["mod"]
        assert mod.convert_to_string(123) == "123"
        assert mod.convert_to_string(True) == "True"
        assert mod.convert_to_string("text") == "text"


class TestJsonSaveData:
    def test_debug_disabled_returns_immediately(self, fake_env):
        mod = fake_env["mod"]
        fake_logger = fake_env["fake_logger"]
        with patch.object(mod, "enable_data_debug", return_value=False):
            mod.json_save_data("data", "prefix")
        fake_logger.info.assert_not_called()

    def test_debug_enabled_writes_json_and_logs(self, fake_env):
        mod = fake_env["mod"]
        fake_logger = fake_env["fake_logger"]
        with patch.object(mod, "enable_data_debug", return_value=True), \
             patch.object(mod, "get_path", return_value="/fake/path/"), \
             patch.object(mod, "get_current_timestamp", return_value="20260523_103000"), \
             patch("builtins.open", create=True) as mock_open, \
             patch("json.dump") as mock_json_dump, \
             patch("os.path.join", return_value="/fake/path/rollout_prefix_20260523_103000.json") as mock_join:
            mod.json_save_data("data", "prefix")
            mock_join.assert_called_once_with("/fake/path/", "rollout_prefix_20260523_103000.json")
            mock_open.assert_called_once_with("/fake/path/rollout_prefix_20260523_103000.json", "a")
            # json.dump should be called with the converted string data and the file handle
            assert mock_json_dump.called
            fake_logger.info.assert_called_once_with(
                "dump data to /fake/path/rollout_prefix_20260523_103000.json done"
            )
