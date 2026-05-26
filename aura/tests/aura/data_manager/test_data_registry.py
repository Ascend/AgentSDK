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
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Fixture: fake module tree for data_registry
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_registry_env():
    """Build an isolated fake module tree for data_registry."""

    # ---- fake threading ----
    fake_threading = types.ModuleType("threading")

    class FakeLock:
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    fake_threading.Lock = FakeLock

    # ---- fake ray ----
    fake_ray = types.ModuleType("ray")

    def ray_remote(cls):
        """Decorator that adds options classmethod and keeps the class intact."""
        cls.options = lambda **kw: cls
        cls.remote = lambda *args, **kwargs: cls(*args, **kwargs)
        return cls

    fake_ray.remote = ray_remote
    fake_ray.get_actor = MagicMock()
    fake_ray.get = MagicMock()

    # ---- fake aura.data_manager submodules ----
    fake_infer_data = types.ModuleType("aura.data_manager.infer_data")
    fake_infer_data.InferDataManager = MagicMock()

    fake_msrl_data = types.ModuleType("aura.data_manager.mindspeed_rl_data")
    fake_msrl_data.MindSpeedRLDataManager = MagicMock()

    fake_verl_data = types.ModuleType("aura.data_manager.verl_data")
    fake_verl_data.VerlDataManager = MagicMock()

    # ---- aura packages to locate the real file ----
    import os
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_data_manager = types.ModuleType("aura.data_manager")
    fake_aura_data_manager.__path__ = [os.path.join(base, "data_manager")]

    fakes = {
        "threading": fake_threading,
        "ray": fake_ray,
        "aura.data_manager.infer_data": fake_infer_data,
        "aura.data_manager.mindspeed_rl_data": fake_msrl_data,
        "aura.data_manager.verl_data": fake_verl_data,
        "aura": fake_aura,
        "aura.data_manager": fake_aura_data_manager,
    }

    target = "aura.data_manager.data_registry"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.data_manager.data_registry as mod
        yield {
            "mod": mod,
            "fake_ray": fake_ray,
            "fake_infer_data": fake_infer_data,
            "fake_msrl_data": fake_msrl_data,
            "fake_verl_data": fake_verl_data,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_actor():
    """Instantiate a DataManagerRegistryActor for testing."""
    from aura.data_manager.data_registry import DataManagerRegistryActor
    return DataManagerRegistryActor()


# ---------------------------------------------------------------------------
# Tests for DataManagerRegistry
# ---------------------------------------------------------------------------
class TestDataManagerRegistry:
    def test_register_and_get(self, fake_registry_env):
        """Register an instance and retrieve it by (backend, mode) key."""
        mod = fake_registry_env["mod"]
        reg = mod.DataManagerRegistry()
        instance = MagicMock()
        reg.register("verl", "train", instance)
        assert reg.get_instance("verl", "train") is instance
        assert reg.get_instance("verl", "infer") is None


# ---------------------------------------------------------------------------
# Tests for DataManagerRegistryActor
# ---------------------------------------------------------------------------
class TestDataManagerRegistryActor:
    def test_initialization(self, fake_registry_env):
        """Actor attributes are properly initialised."""
        actor = make_actor()
        assert isinstance(actor.registry, fake_registry_env["mod"].DataManagerRegistry)
        assert actor.registry_done is False
        assert actor.msrl_data_instance is None
        assert actor.verl_data_instance is None
        assert actor.infer_data_instance is None

    def test_register_msrl_first_time(self, fake_registry_env):
        """First call registers mindspore RL and infer instances, sets done flag."""
        actor = make_actor()
        actor.registry_msrl_data_manager()

        assert actor.registry_done is True
        # Check instances were created
        fake_registry_env["fake_msrl_data"].MindSpeedRLDataManager.assert_called_once()
        fake_registry_env["fake_infer_data"].InferDataManager.assert_called_once()
        # Check registry entries
        assert actor.registry.get_instance("mindspeed_rl", "train") is actor.msrl_data_instance
        assert actor.registry.get_instance("mindspeed_rl", "infer") is actor.infer_data_instance

    def test_register_msrl_second_call_skips(self, fake_registry_env):
        """Second call to register_msrl is a no-op."""
        actor = make_actor()
        actor.registry_done = True
        # Reset mock counts to verify they are not called again
        fake_registry_env["fake_msrl_data"].MindSpeedRLDataManager.reset_mock()
        fake_registry_env["fake_infer_data"].InferDataManager.reset_mock()

        actor.registry_msrl_data_manager()
        fake_registry_env["fake_msrl_data"].MindSpeedRLDataManager.assert_not_called()
        fake_registry_env["fake_infer_data"].InferDataManager.assert_not_called()

    def test_register_verl_first_time(self, fake_registry_env):
        """First call registers verl and infer instances, sets done flag."""
        actor = make_actor()
        actor.registry_verl_data_manager()

        assert actor.registry_done is True
        fake_registry_env["fake_verl_data"].VerlDataManager.assert_called_once()
        fake_registry_env["fake_infer_data"].InferDataManager.assert_called_once()
        assert actor.registry.get_instance("verl", "train") is actor.verl_data_instance
        assert actor.registry.get_instance("verl", "infer") is actor.infer_data_instance

    def test_register_verl_second_call_skips(self, fake_registry_env):
        """Second call to register_verl is a no-op."""
        actor = make_actor()
        actor.registry_done = True
        fake_registry_env["fake_verl_data"].VerlDataManager.reset_mock()
        fake_registry_env["fake_infer_data"].InferDataManager.reset_mock()

        actor.registry_verl_data_manager()
        fake_registry_env["fake_verl_data"].VerlDataManager.assert_not_called()
        fake_registry_env["fake_infer_data"].InferDataManager.assert_not_called()

    def test_get_instance_delegates(self, fake_registry_env):
        """get_instance calls through to the internal registry."""
        actor = make_actor()
        mock_reg = MagicMock()
        actor.registry = mock_reg
        mock_reg.get_instance.return_value = "result"
        result = actor.get_instance("verl", "infer")
        mock_reg.get_instance.assert_called_once_with("verl", "infer")
        assert result == "result"


# ---------------------------------------------------------------------------
# Tests for global helper functions
# ---------------------------------------------------------------------------
class TestGlobalFunctions:
    def test_get_data_manager_actor(self, fake_registry_env):
        """get_data_manager_actor returns the result of .options().remote()."""
        mod = fake_registry_env["mod"]
        mock_actor = MagicMock()
        # Patch options to return a mock that has .remote returning mock_actor
        mock_options = MagicMock()
        mock_options.remote.return_value = mock_actor
        with patch.object(mod.DataManagerRegistryActor, 'options', return_value=mock_options):
            result = mod.get_data_manager_actor()
            assert result is mock_actor
            mod.DataManagerRegistryActor.options.assert_called_once_with(
                name="data_register_actor",
                namespace="register_raygroup",
                lifetime="detached",
            )

    def test_get_data_manager_instance(self, fake_registry_env):
        """get_data_manager_instance obtains actor and calls get_instance.remote."""
        mod = fake_registry_env["mod"]
        fake_ray = fake_registry_env["fake_ray"]

        # Setup a mock actor with get_instance.remote returning an object ref
        mock_actor = MagicMock()
        mock_actor.get_instance.remote.return_value = "object_ref"

        fake_ray.get_actor.return_value = mock_actor
        fake_ray.get.return_value = "final_instance"

        result = mod.get_data_manager_instance("verl", "train")
        fake_ray.get_actor.assert_called_once_with(
            "data_register_actor", namespace="register_raygroup"
        )
        mock_actor.get_instance.remote.assert_called_once_with("verl", "train")
        fake_ray.get.assert_called_once_with("object_ref")
        assert result == "final_instance"
