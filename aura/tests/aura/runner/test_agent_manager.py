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
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Fixture: fake module tree for agent_manager
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_agent_manager_env():
    """Build fully isolated fake modules so agent_manager can be imported and tested."""

    # ---- Fake traceback ----
    fake_traceback = types.ModuleType("traceback")
    fake_traceback.print_exc = MagicMock()

    # ---- Fake ray ----
    fake_ray = types.ModuleType("ray")
    fake_ray.get_actor = MagicMock()
    fake_ray.kill = MagicMock()
    actor_mock = MagicMock()
    # make setup.remote awaitable
    actor_mock.setup = AsyncMock()
    class FakeRemoteActor:
        def options(self, **kwargs):
            return self
        def remote(self, *args, **kwargs):
            return actor_mock
    fake_ray.remote = MagicMock(return_value=FakeRemoteActor())

    # ---- Fake omegaconf ----
    fake_omegaconf = types.ModuleType("omegaconf")
    fake_omegaconf.OmegaConf = MagicMock()
    fake_omegaconf.OmegaConf.to_container = MagicMock(side_effect=lambda obj: obj)

    # ---- Fake ExecutorManager base ----
    fake_executor_manager = types.ModuleType("aura.base.execution.executor_manager")
    class FakeExecutorManager:
        def __init__(self):
            self.instance_dict = {}
        async def create_instance(self, **kwargs):
            pass
    fake_executor_manager.ExecutorManager = FakeExecutorManager

    # ---- Fake loggers ----
    fake_loggers = types.ModuleType("aura.base.log.loggers")
    mock_logger = MagicMock()
    fake_loggers.Loggers = MagicMock(return_value=MagicMock(get_logger=MagicMock(return_value=mock_logger)))

    # ---- Fake AgentExecutor ----
    fake_agent_executor = types.ModuleType("aura.runner.agent_service.agent_executor")
    fake_agent_executor.AgentExecutor = MagicMock()

    # ---- Fake AgenticRLConf (imported inside setup) ----
    fake_conf = types.ModuleType("aura.base.conf.conf")
    fake_conf.AgenticRLConf = MagicMock()
    fake_conf.AgenticRLConf.load_config = MagicMock()

    # ---- Aura packages to locate the real file ----
    import os
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = [os.path.join(base, "runner")]
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_execution = types.ModuleType("aura.base.execution")
    fake_aura_base_execution.__path__ = []
    fake_aura_base_log = types.ModuleType("aura.base.log")
    fake_aura_base_log.__path__ = []
    fake_aura_runner_agent_service = types.ModuleType("aura.runner.agent_service")
    fake_aura_runner_agent_service.__path__ = []

    fakes = {
        "traceback": fake_traceback,
        "ray": fake_ray,
        "omegaconf": fake_omegaconf,
        "aura.base.execution.executor_manager": fake_executor_manager,
        "aura.base.log.loggers": fake_loggers,
        "aura.runner.agent_service.agent_executor": fake_agent_executor,
        "aura.base.conf.conf": fake_conf,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.base": fake_aura_base,
        "aura.base.execution": fake_aura_base_execution,
        "aura.base.log": fake_aura_base_log,
        "aura.runner.agent_service": fake_aura_runner_agent_service,
    }

    target = "aura.runner.agent_manager"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.runner.agent_manager as mod
        yield {
            "mod": mod,
            "AgentManager": mod.AgentManager,
            "get_or_create": mod.get_or_create_agent_manager,
            "destroy": mod.destroy_agent_manager,
            "fake_ray": fake_ray,
            "actor_mock": actor_mock,
            "mock_logger": mock_logger,
            "fake_conf": fake_conf,
            "fake_executor_manager": fake_executor_manager,
            "fake_agent_executor": fake_agent_executor,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_config_with_instances(instances):
    """Return a mock config object with agent_instances list."""
    config = MagicMock()
    config.agent_instances = instances
    return config


def make_instance_conf(name="agent1", executor_num=1, executor_kwargs={"key": "val"}, resource_info={"cpu": 1}):
    conf = MagicMock()
    conf.name = name
    conf.executor_num = executor_num
    conf.executor_kwargs = executor_kwargs
    conf.resource_info = resource_info
    return conf


# ---------------------------------------------------------------------------
# Tests for AgentManager.setup
# ---------------------------------------------------------------------------
class TestAgentManagerSetup:
    @pytest.mark.asyncio
    async def test_setup_creates_instances(self, fake_agent_manager_env):
        """Setup reads config and calls create_instance for each agent instance."""
        AgentManager = fake_agent_manager_env["AgentManager"]
        fake_conf = fake_agent_manager_env["fake_conf"]

        inst_conf = make_instance_conf("agent1", 2, {"model": "test"}, {"gpu": 1})
        config = MagicMock()
        config.agent_instances = [inst_conf]
        fake_conf.AgenticRLConf.load_config.return_value = config

        manager = AgentManager()
        manager.create_instance = AsyncMock()
        await manager.setup()

        manager.create_instance.assert_called_once_with(
            name="agent1",
            executor_class=fake_agent_manager_env["fake_agent_executor"].AgentExecutor,
            executor_num=2,
            executor_kwargs={"model": "test"},
            resource_info={"gpu": 1},
        )

    @pytest.mark.asyncio
    async def test_setup_multiple_instances(self, fake_agent_manager_env):
        """Setup handles multiple agent instances."""
        AgentManager = fake_agent_manager_env["AgentManager"]
        fake_conf = fake_agent_manager_env["fake_conf"]

        inst1 = make_instance_conf("a1", 1, {}, {})
        inst2 = make_instance_conf("a2", 3, {"x": 1}, {"mem": 4})
        config = MagicMock()
        config.agent_instances = [inst1, inst2]
        fake_conf.AgenticRLConf.load_config.return_value = config

        manager = AgentManager()
        manager.create_instance = AsyncMock()
        await manager.setup()

        assert manager.create_instance.call_count == 2
        args2 = manager.create_instance.call_args_list[1].kwargs
        assert args2["name"] == "a2"
        assert args2["executor_num"] == 3

    @pytest.mark.asyncio
    async def test_setup_exception_raises_and_logs(self, fake_agent_manager_env):
        """If an exception occurs, it is re-raised and traceback is printed."""
        AgentManager = fake_agent_manager_env["AgentManager"]
        fake_conf = fake_agent_manager_env["fake_conf"]

        inst = make_instance_conf("agent_fail", 1)
        config = MagicMock()
        config.agent_instances = [inst]
        fake_conf.AgenticRLConf.load_config.return_value = config

        manager = AgentManager()
        manager.create_instance = AsyncMock(side_effect=RuntimeError("setup error"))
        with pytest.raises(RuntimeError, match="setup error"):
            await manager.setup()

        fake_agent_manager_env["mod"].traceback.print_exc.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for get_or_create_agent_manager
# ---------------------------------------------------------------------------
class TestGetOrCreateAgentManager:
    @pytest.mark.asyncio
    async def test_existing_actor(self, fake_agent_manager_env):
        """If actor exists, return it directly."""
        fake_ray = fake_agent_manager_env["fake_ray"]
        existing = MagicMock()
        fake_ray.get_actor.return_value = existing

        result = await fake_agent_manager_env["get_or_create"]()
        assert result is existing
        fake_ray.get_actor.assert_called_once_with("AgentManager")
        fake_ray.remote.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_new_actor(self, fake_agent_manager_env):
        """If actor does not exist, create a new one and call setup."""
        fake_ray = fake_agent_manager_env["fake_ray"]
        fake_ray.get_actor.side_effect = ValueError("not found")

        actor_mock = fake_agent_manager_env["actor_mock"]
        result = await fake_agent_manager_env["get_or_create"]()
        assert result is actor_mock
        fake_ray.remote.assert_called_once()
        actor_mock.setup.remote.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests for destroy_agent_manager
# ---------------------------------------------------------------------------
class TestDestroyAgentManager:
    def test_existing_actor(self, fake_agent_manager_env):
        """If actor exists, kill it."""
        fake_ray = fake_agent_manager_env["fake_ray"]
        actor = MagicMock()
        fake_ray.get_actor.return_value = actor

        fake_agent_manager_env["destroy"]()
        fake_ray.kill.assert_called_once_with(actor)

    def test_missing_actor(self, fake_agent_manager_env):
        """If actor not found, log info and do not kill."""
        fake_ray = fake_agent_manager_env["fake_ray"]
        fake_ray.get_actor.side_effect = ValueError("missing")

        fake_agent_manager_env["destroy"]()
        fake_ray.kill.assert_not_called()
        fake_agent_manager_env["mock_logger"].info.assert_called()
