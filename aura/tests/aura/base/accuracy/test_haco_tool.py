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
from unittest.mock import MagicMock, patch, mock_open
import pytest

# ---------------------------------------------------------------------------
# Fixture: fake module tree for haco_tool
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_env():
    """Build an isolated fake module tree and return the module under test."""

    # ---- fake haco ----
    fake_haco = types.ModuleType("haco")
    fake_haco.Sentinel = MagicMock(name="Sentinel")
    fake_haco.parse_config = MagicMock(name="parse_config")

    # ---- fake portalocker ----
    fake_portalocker = types.ModuleType("portalocker")
    fake_portalocker.lock = MagicMock(name="portalocker.lock")
    fake_portalocker.LOCK_EX = "LOCK_EX"

    # ---- aura packages to locate the real haco_tool module ----
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_accuracy = types.ModuleType("aura.base.accuracy")
    fake_aura_base_accuracy.__path__ = [os.path.join(base, "base/accuracy")]

    fakes = {
        "haco": fake_haco,
        "portalocker": fake_portalocker,
        "aura": fake_aura,
        "aura.base": fake_aura_base,
        "aura.base.accuracy": fake_aura_base_accuracy,
    }

    target = "aura.base.accuracy.haco_tool"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.base.accuracy.haco_tool as mod
        yield {
            "mod": mod,
            "fake_haco": fake_haco,
            "fake_portalocker": fake_portalocker,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestEnableHaco:
    def test_haco_available(self, fake_env):
        """When HAS_HACO is True, return True and do not log."""
        mod = fake_env["mod"]
        logger = MagicMock()
        mod.HAS_HACO = True
        assert mod.enable_haco(logger) is True
        logger.info.assert_not_called()

    def test_haco_unavailable_logs_and_returns_false(self, fake_env):
        """When HAS_HACO is False, return False and log a message."""
        mod = fake_env["mod"]
        logger = MagicMock()
        mod.HAS_HACO = False
        assert mod.enable_haco(logger) is False
        logger.info.assert_called_once()
        assert "No module haco" in logger.info.call_args[0][0]


class TestUpdateHacoRolloutMasterIp:
    def test_uses_default_path_when_env_not_set(self, fake_env):
        """If ROLLOUT_HACO_JSON is not set, fall back to default path."""
        mod = fake_env["mod"]
        fake_portalocker = fake_env["fake_portalocker"]
        new_ip = "10.0.0.1"

        file_content = '{"sampler": {"host_name": "old"}}'
        m_open = mock_open(read_data=file_content)

        with patch.dict(os.environ, {}, clear=True), \
             patch("builtins.open", m_open), \
             patch("json.dump") as mock_dump:
            mod.update_haco_rollout_master_ip(new_ip)

        # Should have opened the default file
        m_open.assert_called_once_with("./configs/rollout_raw_data_sampler.json", "r+", encoding="utf8")
        fake_portalocker.lock.assert_called_once()
        # Verify that host_name was updated in the dumped data
        args, _ = mock_dump.call_args
        dumped_data = args[0]
        assert dumped_data["sampler"]["host_name"] == new_ip

    def test_uses_env_path(self, fake_env):
        """If ROLLOUT_HACO_JSON is set, use that path."""
        mod = fake_env["mod"]
        fake_portalocker = fake_env["fake_portalocker"]
        new_ip = "10.0.0.2"
        env_path = "/custom/path/config.json"

        # The JSON must contain the "sampler" key because the code accesses it directly
        file_content = '{"sampler": {"host_name": "old"}}'
        m_open = mock_open(read_data=file_content)

        with patch.dict(os.environ, {"ROLLOUT_HACO_JSON": env_path}), \
             patch("builtins.open", m_open), \
             patch("json.dump") as mock_dump:
            mod.update_haco_rollout_master_ip(new_ip)

        m_open.assert_called_once_with(env_path, "r+", encoding="utf8")
        fake_portalocker.lock.assert_called_once_with(m_open(), "LOCK_EX")
        args, _ = mock_dump.call_args
        assert args[0]["sampler"]["host_name"] == new_ip

    def test_updates_existing_sampler_key(self, fake_env):
        """When the JSON already contains other keys besides sampler, it still updates host_name."""
        mod = fake_env["mod"]
        new_ip = "10.0.0.4"
        file_content = '{"sampler": {"host_name": "old", "port": 8080}, "other": "data"}'
        m_open = mock_open(read_data=file_content)

        with patch.dict(os.environ, {}, clear=True), \
             patch("builtins.open", m_open), \
             patch("json.dump") as mock_dump:
            mod.update_haco_rollout_master_ip(new_ip)

        args, _ = mock_dump.call_args
        assert args[0]["sampler"]["host_name"] == new_ip
        assert args[0]["sampler"]["port"] == 8080
        assert args[0]["other"] == "data"


class TestActorWorkerUpdateHaco:
    def test_actor_worker_update_haco(self, fake_env):
        mod = fake_env["mod"]
        fake_haco = fake_env["fake_haco"]

        addr = "worker_addr"
        model = MagicMock()
        optimizer = MagicMock()

        sentinel_config = {"sampler": {"host_name": "old"}}
        fake_haco.parse_config.return_value = sentinel_config
        fake_haco.Sentinel.return_value = MagicMock(name="sentinel_instance")

        with patch.dict(os.environ, {"ACTOR_HACO_JSON": "/fake/actor_config.json"}), \
             patch.object(mod, "update_haco_rollout_master_ip") as mock_update_ip:
            result = mod.actor_worker_update_haco(addr, model, optimizer)

        fake_haco.parse_config.assert_called_once_with("/fake/actor_config.json")
        assert sentinel_config["sampler"]["host_name"] == addr
        mock_update_ip.assert_called_once_with(addr)
        fake_haco.Sentinel.assert_called_once_with(sentinel_config, model=model, optimizer=optimizer)
        assert result == fake_haco.Sentinel.return_value

    def test_actor_worker_uses_default_path(self, fake_env):
        mod = fake_env["mod"]
        fake_haco = fake_env["fake_haco"]

        sentinel_config = {"sampler": {}}
        fake_haco.parse_config.return_value = sentinel_config

        with patch.dict(os.environ, {}, clear=True), \
             patch.object(mod, "update_haco_rollout_master_ip"):
            mod.actor_worker_update_haco("addr", MagicMock(), MagicMock())

        fake_haco.parse_config.assert_called_once_with("./configs/accuracy/rollout_raw_data_sampler.json")


class TestVllmModelRunnerUpdateHaco:
    def test_vllm_model_runner_update_haco(self, fake_env):
        mod = fake_env["mod"]
        fake_haco = fake_env["fake_haco"]

        model = MagicMock()
        sentinel_configs = {"key": "value"}
        fake_haco.parse_config.return_value = sentinel_configs
        fake_haco.Sentinel.return_value = MagicMock(name="sentinel_instance")

        with patch.dict(os.environ, {"ROLLOUT_HACO_JSON": "/fake/vllm_config.json"}):
            result = mod.vllm_model_runner_update_haco(model)

        fake_haco.parse_config.assert_called_once_with("/fake/vllm_config.json")
        fake_haco.Sentinel.assert_called_once_with(sentinel_configs, model)
        assert result == fake_haco.Sentinel.return_value

    def test_vllm_model_runner_default_path(self, fake_env):
        mod = fake_env["mod"]
        fake_haco = fake_env["fake_haco"]

        with patch.dict(os.environ, {}, clear=True):
            mod.vllm_model_runner_update_haco(MagicMock())

        fake_haco.parse_config.assert_called_once_with("../../configs/accuracy/rollout_raw_data_sampler.json")
