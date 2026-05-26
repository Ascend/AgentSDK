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

import torch
import pytest
from omegaconf import OmegaConf


@pytest.fixture
def base_conf():
    return OmegaConf.create(
        {
            "name": "test_conf",
            "agentic_ai": {"mode": "direct"},
            "serve_conf": {"host": "0.0.0.0", "port": 8000},
            "train_instances": [{"executor_kwargs": {"work_mode": "one_step_off"}}],
            "direct_conf": {"entrypoints": []},
        }
    )


def build_actor():
    actor = MagicMock()
    actor.registry_msrl_train.remote.return_value = "ok"
    actor.registry_verl_train.remote.return_value = "ok"
    actor.registry_omnirl_train.remote.return_value = "ok"
    actor.registry_msrl_data_manager.remote.return_value = "ok"
    actor.registry_verl_data_manager.remote.return_value = "ok"
    actor.registry_omnirl_data_manager.remote.return_value = "ok"
    return actor


class DummyRouter:
    @classmethod
    async def create(cls):
        obj = cls()
        obj.train = AsyncMock()
        return obj


class FailRouter:
    @classmethod
    async def create(cls):
        obj = cls()
        async def fail_train(*args, **kwargs):
            raise RuntimeError("train failed")
        obj.train = fail_train
        return obj


@pytest.fixture
def fake_env():
    train_actor = build_actor()
    data_actor = build_actor()

    fake_ray = types.ModuleType("ray")
    fake_ray.__path__ = []
    fake_ray.get = MagicMock()                # ray.get
    fake_ray.init = MagicMock()               # ray.init
    fake_ray.get_actor = MagicMock()          # ray.get_actor

    fake_serve = types.ModuleType("ray.serve")
    fake_serve.HTTPOptions = MagicMock
    fake_serve.start = MagicMock()
    fake_serve.run = MagicMock()
    fake_serve.get_app_handle = MagicMock()

    fake_actor = types.ModuleType("ray.actor")
    fake_actor.ActorHandle = MagicMock

    fake_ray.serve = fake_serve
    fake_ray.actor = fake_actor

    with patch.dict(sys.modules, {
        "ray": fake_ray,
        "ray.serve": fake_serve,
        "ray.actor": fake_actor,
    }):
        import aura as _real_aura_pkg
        aura_path = _real_aura_pkg.__path__

    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = aura_path

    fake_runner = types.ModuleType("aura.runner")
    fake_runner.__path__ = []

    fake_infer = types.ModuleType("aura.runner.infer_manager")
    fake_infer.get_or_create_infer_manager = AsyncMock(return_value="mock_infer_actor")
    fake_infer.destroy_infer_manager = MagicMock()

    fake_agent = types.ModuleType("aura.runner.agent_manager")
    fake_agent.get_or_create_agent_manager = AsyncMock(return_value="mock_agent_actor")
    fake_agent.destroy_agent_manager = MagicMock()

    fake_utils = types.ModuleType("aura.controllers.utils.utils")
    fake_utils.kill_actor = MagicMock()
    fake_utils.DEFAULT_SLEEP_TIME = 1

    fake_conf = types.ModuleType("aura.base.conf.conf")
    class FakeConf:
        CONF_ENV = "FAKE_ENV"
    fake_conf.AgenticRLConf = FakeConf

    fake_loggers = types.ModuleType("aura.base.log.loggers")
    fake_loggers.Loggers = MagicMock(return_value=MagicMock())

    fake_train_register_mod = types.ModuleType("aura.trainer.trainer_register.train_register")
    fake_train_register_mod.get_train_actor = MagicMock(return_value=train_actor)

    fake_data_registry_mod = types.ModuleType("aura.data_manager.data_registry")
    fake_data_registry_mod.get_data_manager_actor = MagicMock(return_value=data_actor)

    fake_train_router_mod = types.ModuleType("aura.trainer.train_router")
    fake_train_router_mod.TrainRouter = DummyRouter

    fake_serve_pkg = types.ModuleType("aura.serve")
    fake_serve_pkg.__path__ = []
    fake_serve_mod = types.ModuleType("aura.serve.serve")
    fake_router_mod = types.ModuleType("aura.serve.router")
    fake_serve_mod.deployment = MagicMock()

    all_fakes = {
        "ray": fake_ray,
        "ray.serve": fake_serve,
        "ray.actor": fake_actor,
        "aura": fake_aura,
        "aura.runner": fake_runner,
        "aura.runner.infer_manager": fake_infer,
        "aura.runner.agent_manager": fake_agent,
        "aura.controllers.utils.utils": fake_utils,
        "aura.base.conf.conf": fake_conf,
        "aura.base.log.loggers": fake_loggers,
        "aura.trainer.trainer_register.train_register": fake_train_register_mod,
        "aura.data_manager.data_registry": fake_data_registry_mod,
        "aura.trainer.train_router": fake_train_router_mod,
        "aura.serve": fake_serve_pkg,
        "aura.serve.serve": fake_serve_mod,
        "aura.serve.router": fake_router_mod,
    }

    with patch.dict(sys.modules, all_fakes):
        import aura.start as real_start
        with patch.dict(sys.modules, {"aura.start": real_start}):
            yield {
                "infer_mock": fake_infer.get_or_create_infer_manager,
                "fake_train_router": fake_train_router_mod,
                "fake_serve_module": fake_serve_mod,
                "train_actor": train_actor,
                "data_actor": data_actor,
            }


def test_start_direct_mode(base_conf, fake_env):
    import aura.start as start

    base_conf.direct_conf.entrypoints = [
        {"job_type": "msrl_train", "job_name": "job1", "job_kwargs": {}},
        {"job_type": "verl_train", "job_name": "job2", "job_kwargs": {}},
    ]
    start.start_direct_mode(base_conf)

    train_actor = fake_env["train_actor"]
    data_actor = fake_env["data_actor"]
    assert train_actor.registry_msrl_train.remote.called
    assert train_actor.registry_verl_train.remote.called
    assert data_actor.registry_msrl_data_manager.remote.called
    assert data_actor.registry_verl_data_manager.remote.called


def test_start_direct_mode_unknown_job_type(base_conf, fake_env):
    import aura.start as start

    base_conf.direct_conf.entrypoints = [
        {"job_type": "unknown_train", "job_name": "bad_job", "job_kwargs": {}}
    ]
    start.start_direct_mode(base_conf)
    assert not fake_env["train_actor"].registry_msrl_train.remote.called


def test_start_direct_mode_hybrid_skip_infer(base_conf, fake_env):
    import aura.start as start

    base_conf.train_instances[0].executor_kwargs.work_mode = "hybrid"
    base_conf.direct_conf.entrypoints = [
        {"job_type": "msrl_train", "job_name": "job1", "job_kwargs": {}}
    ]
    start.start_direct_mode(base_conf)
    fake_env["infer_mock"].assert_not_called()


def test_start_direct_mode_submit_job_exception(base_conf, fake_env):
    import aura.start as start

    base_conf.direct_conf.entrypoints = [
        {"job_type": "msrl_train", "job_name": "job1", "job_kwargs": {}}
    ]
    fake_env["fake_train_router"].TrainRouter = FailRouter
    with pytest.raises(RuntimeError):
        start.start_direct_mode(base_conf)


def test_start_serve_mode_existing_app(base_conf, fake_env):
    import aura.start as start

    with patch.object(start, "serve", MagicMock()) as mock_serve, \
         patch("time.sleep", side_effect=KeyboardInterrupt):
        mock_serve.get_app_handle.return_value = "app"
        with pytest.raises(KeyboardInterrupt):
            start.start_serve_mode(base_conf)


def test_start_serve_mode_create_app(base_conf, fake_env):
    import aura.start as start

    fake_env["fake_serve_module"].deployment = object()

    with patch.object(start, "serve", MagicMock()) as mock_serve, \
         patch("time.sleep", side_effect=KeyboardInterrupt):
        mock_serve.get_app_handle.side_effect = Exception("not found")
        with pytest.raises(KeyboardInterrupt):
            start.start_serve_mode(base_conf)
        mock_serve.run.assert_called_once()


def test_start_direct_mode_with_serve(base_conf, fake_env):
    import aura.start as start

    base_conf.direct_conf.entrypoints = [
        {"job_type": "omni_rl", "job_name": "omni_job", "job_kwargs": {}}
    ]
    with patch.object(start, "serve", MagicMock()) as mock_serve:
        mock_serve.get_app_handle.return_value = "app"
        start.start_direct_mode_with_serve(base_conf)

    assert fake_env["train_actor"].registry_omnirl_train.remote.called


def test_main_serve(base_conf, fake_env):
    import aura.start as start
    base_conf.agentic_ai.mode = "serve"
    with patch("ray.init"), patch("aura.start.start_serve_mode") as serve_mock:
        start.main(base_conf)
        serve_mock.assert_called_once()


def test_main_direct(base_conf, fake_env):
    import aura.start as start
    base_conf.agentic_ai.mode = "direct"
    with patch("ray.init"), patch("aura.start.start_direct_mode") as direct_mock:
        start.main(base_conf)
        direct_mock.assert_called_once()


def test_main_direct_with_serve(base_conf, fake_env):
    import aura.start as start
    base_conf.agentic_ai.mode = "direct_with_serve"
    with patch("ray.init"), patch("aura.start.start_direct_mode_with_serve") as mode_mock:
        start.main(base_conf)
        mode_mock.assert_called_once()


def test_main_invalid_mode(base_conf, fake_env):
    import aura.start as start
    base_conf.agentic_ai.mode = "invalid"
    with patch("ray.init"):
        with pytest.raises(ValueError):
            start.main(base_conf)
