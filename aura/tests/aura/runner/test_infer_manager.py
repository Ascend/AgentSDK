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
from types import SimpleNamespace

import pytest

@pytest.fixture
def fake_infer_env():
    import os
    import aura
    real_aura_path = aura.__path__
    real_runner_path = [os.path.join(real_aura_path[0], "runner")] if real_aura_path else []

    # Packages
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = real_aura_path
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = real_runner_path

    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_execution = types.ModuleType("aura.base.execution")
    fake_aura_base_execution.__path__ = []
    fake_aura_base_log = types.ModuleType("aura.base.log")
    fake_aura_base_log.__path__ = []
    fake_aura_base_conf = types.ModuleType("aura.base.conf")
    fake_aura_base_conf.__path__ = []

    fake_aura_runner_infer_service = types.ModuleType("aura.runner.infer_service")
    fake_aura_runner_infer_service.__path__ = []

    # Leaf modules
    fake_executor_manager = types.ModuleType("aura.base.execution.executor_manager")
    class FakeExecutorManager:
        def __init__(self):
            self.instance_dict = {}
            self.create_instance = AsyncMock()
    fake_executor_manager.ExecutorManager = FakeExecutorManager

    fake_infer_executor = types.ModuleType("aura.runner.infer_service.infer_executor")
    fake_infer_executor.InferExecutor = MagicMock

    fake_infer_pd_executor = types.ModuleType("aura.runner.infer_service.infer_pd_executor")
    fake_infer_pd_executor.InferPDSepExecutor = MagicMock

    fake_loggers = types.ModuleType("aura.base.log.loggers")
    fake_loggers.Loggers = MagicMock(return_value=MagicMock())

    fake_conf = types.ModuleType("aura.base.conf.conf")
    # load_config returns a SimpleNamespace to support attribute access
    from types import SimpleNamespace
    fake_conf.AgenticRLConf = MagicMock()
    fake_conf.AgenticRLConf.load_config = MagicMock(return_value=SimpleNamespace())

    # omegaconf
    fake_omegaconf = types.ModuleType("omegaconf")
    def to_container(obj, resolve=None):
        if isinstance(obj, dict):
            return obj
        # Convert SimpleNamespace or similar object to dict
        return vars(obj)
    fake_omegaconf.OmegaConf = MagicMock()
    fake_omegaconf.OmegaConf.to_container = to_container

    # ray
    fake_ray = types.ModuleType("ray")
    fake_ray.__path__ = []
    actor_mock = MagicMock()
    actor_mock.setup.remote = AsyncMock()
    fake_remote_class = MagicMock()
    fake_remote_class.options.return_value = fake_remote_class
    fake_remote_class.remote.return_value = actor_mock
    fake_ray.remote = MagicMock(return_value=fake_remote_class)
    fake_ray.get_actor = MagicMock()
    fake_ray.kill = MagicMock()

    all_fakes = {
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.base": fake_aura_base,
        "aura.base.execution": fake_aura_base_execution,
        "aura.base.log": fake_aura_base_log,
        "aura.base.conf": fake_aura_base_conf,
        "aura.runner.infer_service": fake_aura_runner_infer_service,
        "aura.base.execution.executor_manager": fake_executor_manager,
        "aura.runner.infer_service.infer_executor": fake_infer_executor,
        "aura.runner.infer_service.infer_pd_executor": fake_infer_pd_executor,
        "aura.base.log.loggers": fake_loggers,
        "aura.base.conf.conf": fake_conf,
        "omegaconf": fake_omegaconf,
        "ray": fake_ray,
    }

    with patch.dict(sys.modules, all_fakes):
        import aura.runner.infer_manager as infer_manager
        yield {
            "module": infer_manager,
            "InferManager": infer_manager.InferManager,
            "get_or_create": infer_manager.get_or_create_infer_manager,
            "destroy": infer_manager.destroy_infer_manager,
            "fake_ray": fake_ray,
            "actor_mock": actor_mock,
            "fake_conf": fake_conf,
            "fake_infer_executor": fake_infer_executor,
            "fake_infer_pd_executor": fake_infer_pd_executor,
        }


class TestInferManagerSetup:
    @pytest.mark.asyncio
    async def test_setup_pd_mode(self, fake_infer_env):
        InferManager = fake_infer_env["InferManager"]
        fake_conf = fake_infer_env["fake_conf"]
        fake_infer_pd = fake_infer_env["fake_infer_pd_executor"]

        config = SimpleNamespace(
            pd_mode=True,
            infer_pd_instances=[
                SimpleNamespace(name="pd1", executor_num=2,
                                executor_kwargs={"key": "val"},
                                resource_info={"gpu": 1})
            ]
        )
        fake_conf.AgenticRLConf.load_config.return_value = config

        manager = InferManager()
        await manager.setup()

        call_args = manager.create_instance.call_args_list
        assert len(call_args) == 1
        kwargs = call_args[0].kwargs
        assert kwargs["executor_class"] == fake_infer_pd.InferPDSepExecutor
        assert kwargs["name"] == "pd1"
        assert kwargs["executor_num"] == 2
        assert kwargs["executor_kwargs"] == {"key": "val"}
        assert kwargs["resource_info"] == {"gpu": 1}

    @pytest.mark.asyncio
    async def test_setup_non_pd_mode(self, fake_infer_env):
        InferManager = fake_infer_env["InferManager"]
        fake_conf = fake_infer_env["fake_conf"]
        fake_infer = fake_infer_env["fake_infer_executor"]

        config = SimpleNamespace(
            pd_mode=False,
            infer_instances=[
                SimpleNamespace(name="inf1", executor_num=1,
                                executor_kwargs={}, resource_info={})
            ]
        )
        fake_conf.AgenticRLConf.load_config.return_value = config

        manager = InferManager()
        await manager.setup()

        kwargs = manager.create_instance.call_args_list[0].kwargs
        assert kwargs["executor_class"] == fake_infer.InferExecutor

    @pytest.mark.asyncio
    async def test_setup_missing_pd_mode_defaults_to_non_pd(self, fake_infer_env):
        InferManager = fake_infer_env["InferManager"]
        fake_conf = fake_infer_env["fake_conf"]
        fake_infer = fake_infer_env["fake_infer_executor"]

        config = SimpleNamespace(
            infer_instances=[
                SimpleNamespace(name="inf1", executor_num=1,
                                executor_kwargs={}, resource_info={})
            ]
        )
        fake_conf.AgenticRLConf.load_config.return_value = config

        manager = InferManager()
        await manager.setup()

        kwargs = manager.create_instance.call_args_list[0].kwargs
        assert kwargs["executor_class"] == fake_infer.InferExecutor

    @pytest.mark.asyncio
    async def test_setup_exception(self, fake_infer_env):
        InferManager = fake_infer_env["InferManager"]
        fake_conf = fake_infer_env["fake_conf"]

        config = SimpleNamespace(
            pd_mode=False,
            infer_instances=[
                SimpleNamespace(name="i", executor_num=1,
                                executor_kwargs={}, resource_info={})
            ]
        )
        fake_conf.AgenticRLConf.load_config.return_value = config

        manager = InferManager()
        manager.create_instance.side_effect = Exception("setup error")
        with pytest.raises(Exception, match="setup error"):
            await manager.setup()
        manager.create_instance.assert_called_once()


class TestGetOrCreateInferManager:
    @pytest.mark.asyncio
    async def test_existing_actor(self, fake_infer_env):
        get_or_create = fake_infer_env["get_or_create"]
        fake_ray = fake_infer_env["fake_ray"]
        existing_actor = MagicMock()
        fake_ray.get_actor.return_value = existing_actor

        result = await get_or_create()

        assert result is existing_actor
        fake_ray.get_actor.assert_called_once_with("InferManager")
        fake_ray.remote.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_actor(self, fake_infer_env):
        get_or_create = fake_infer_env["get_or_create"]
        fake_ray = fake_infer_env["fake_ray"]
        actor_mock = fake_infer_env["actor_mock"]

        fake_ray.get_actor.side_effect = ValueError("not found")
        result = await get_or_create()

        assert result is actor_mock
        fake_ray.remote.assert_called_once()
        actor_mock.setup.remote.assert_awaited_once()


class TestDestroyInferManager:
    def test_existing_actor(self, fake_infer_env):
        destroy = fake_infer_env["destroy"]
        fake_ray = fake_infer_env["fake_ray"]
        actor = MagicMock()
        fake_ray.get_actor.return_value = actor

        destroy()

        fake_ray.kill.assert_called_once_with(actor)

    def test_missing_actor(self, fake_infer_env):
        destroy = fake_infer_env["destroy"]
        fake_ray = fake_infer_env["fake_ray"]
        fake_ray.get_actor.side_effect = ValueError("not found")

        destroy()

        fake_ray.kill.assert_not_called()
