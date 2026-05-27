#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import sys
import types
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Fixture: fake module tree
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_infer_router_env():
    import os as _os
    import aura as _aura
    real_aura_path = _aura.__path__
    real_runner_path = [_os.path.join(real_aura_path[0], "runner")] if real_aura_path else []

    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = real_aura_path
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = real_runner_path
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_log = types.ModuleType("aura.base.log")
    fake_aura_base_log.__path__ = []
    fake_aura_base_conf = types.ModuleType("aura.base.conf")
    fake_aura_base_conf.__path__ = []
    fake_aura_runner_scheduler = types.ModuleType("aura.runner.scheduler")
    fake_aura_runner_scheduler.__path__ = []

    # ---- leaf modules ----
    fake_loggers = types.ModuleType("aura.base.log.loggers")
    fake_loggers.Loggers = MagicMock(return_value=MagicMock(name="logger"))

    fake_conf = types.ModuleType("aura.base.conf.conf")
    fake_conf.AgenticRLConf = MagicMock()
    fake_conf.AgenticRLConf.load_config = MagicMock()

    fake_req_scheduler = types.ModuleType("aura.runner.scheduler.req_scheduler")
    mock_scheduler = MagicMock(name="Scheduler")
    mock_scheduler.schedule = AsyncMock()
    mock_scheduler.release = MagicMock()
    mock_scheduler.running_reqs = {}
    fake_req_scheduler.SchedulerFactory = MagicMock()
    fake_req_scheduler.SchedulerFactory.get_scheduler = MagicMock(return_value=mock_scheduler)

    fake_infer_manager_mod = types.ModuleType("aura.runner.infer_manager")
    fake_infer_manager_mod.get_or_create_infer_manager = AsyncMock()

    fake_ray = types.ModuleType("ray")
    fake_ray.__path__ = []
    fake_ray.get = MagicMock(return_value=[])

    mock_os = MagicMock()
    mock_os.getenv = MagicMock(return_value="1")

    # ---- random: always return first element ----
    mock_random = MagicMock()
    mock_random.choice = MagicMock(side_effect=lambda lst: lst[0])

    mock_lock = MagicMock()
    mock_lock.__aenter__ = AsyncMock()
    mock_lock.__aexit__ = AsyncMock()

    all_fakes = {
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.base": fake_aura_base,
        "aura.base.log": fake_aura_base_log,
        "aura.base.conf": fake_aura_base_conf,
        "aura.runner.scheduler": fake_aura_runner_scheduler,
        "aura.base.log.loggers": fake_loggers,
        "aura.base.conf.conf": fake_conf,
        "aura.runner.scheduler.req_scheduler": fake_req_scheduler,
        "aura.runner.infer_manager": fake_infer_manager_mod,
        "ray": fake_ray,
        "os": mock_os,
        "random": mock_random,
    }

    with patch.dict(sys.modules, all_fakes):
        import aura.runner.infer_router as infer_router_mod
        infer_router_mod.InferRouter._router = None
        yield {
            "module": infer_router_mod,
            "InferRouter": infer_router_mod.InferRouter,
            "fake_conf": fake_conf,
            "fake_req_scheduler": fake_req_scheduler,
            "mock_scheduler": mock_scheduler,
            "fake_infer_manager_mod": fake_infer_manager_mod,
            "fake_ray": fake_ray,
            "mock_os": mock_os,
            "mock_random": mock_random,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_infer_instance(model_name="model", executor_num=2, executor_list=None):
    """Create a mock infer instance with exactly `executor_num` executors."""
    instance = MagicMock(name=f"instance_{model_name}")
    instance.executor_num = executor_num
    if executor_list is None:
        executor_list = []
        for i in range(executor_num):
            e = MagicMock(name=f"executor_{i}")
            e.stream_execute_method.remote = MagicMock()
            e.execute_method.remote = AsyncMock()
            e.ref = MagicMock()
            e.ref.update_weights.remote = MagicMock()
            e.ref.vllm_statistics.remote = MagicMock()
            e.ref.reset_prefix_cache.remote = MagicMock()
            executor_list.append(e)
    instance.executor_list = executor_list
    return instance


def make_config(chat_server_mode=False):
    engine_kwargs = {}
    if chat_server_mode:
        engine_kwargs["chat_server"] = True
    instance_conf = SimpleNamespace(
        executor_kwargs=SimpleNamespace(engine_kwargs=engine_kwargs)
    )
    return SimpleNamespace(infer_instances=[instance_conf])


async def async_stream(items):
    """Yield awaitable futures that resolve to given string values."""
    for item in items:
        future = asyncio.Future()
        future.set_result(item)
        yield future


# ===========================================================================
# Tests
# ===========================================================================
class TestInferRouterCreation:
    @pytest.mark.asyncio
    async def test_create_singleton(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        fake_infer_manager_mod = fake_infer_router_env["fake_infer_manager_mod"]
        fake_conf = fake_infer_router_env["fake_conf"]
        fake_conf.AgenticRLConf.load_config.return_value = make_config(False)

        router1 = await InferRouter.create()
        assert router1 is InferRouter._router
        fake_infer_manager_mod.get_or_create_infer_manager.assert_awaited_once()

        fake_infer_manager_mod.get_or_create_infer_manager.reset_mock()
        router2 = await InferRouter.create()
        assert router1 is router2
        fake_infer_manager_mod.get_or_create_infer_manager.assert_not_called()


class TestInferRouterInit:
    @pytest.mark.asyncio
    async def test_chat_server_mode_true(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        fake_conf = fake_infer_router_env["fake_conf"]
        fake_conf.AgenticRLConf.load_config.return_value = make_config(True)

        router = InferRouter(MagicMock())
        await router.init("any_model")
        assert router.inited is True

    @pytest.mark.asyncio
    async def test_chat_server_mode_false(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        mock_scheduler = fake_infer_router_env["mock_scheduler"]
        fake_req_scheduler = fake_infer_router_env["fake_req_scheduler"]
        fake_conf = fake_infer_router_env["fake_conf"]
        fake_conf.AgenticRLConf.load_config.return_value = make_config(False)

        infer_manager = MagicMock()
        instance = make_infer_instance("model", 2)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)

        router = InferRouter(infer_manager)
        await router.init("test_model")
        fake_req_scheduler.SchedulerFactory.get_scheduler.assert_called_once()
        assert router.default_model_name == "test_model"
        assert router.inited is True

        infer_manager.get_instance.remote.reset_mock()
        await router.init("test_model")
        infer_manager.get_instance.remote.assert_not_called()


class TestGetApplicationId:
    def test_get_application_id(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        router = InferRouter(MagicMock())
        assert router.get_application_id("app123--req456") == "app123"
        assert router.get_application_id("no--sep") == "no"


class TestStreamMethods:
    @pytest.mark.asyncio
    async def test_stream_chat_completions_success(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        infer_manager = MagicMock()
        instance = make_infer_instance("model_a", 1)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)
        instance.executor_list[0].stream_execute_method.remote.return_value = async_stream(["chunk1", "chunk2"])

        router = InferRouter(infer_manager)
        request = {"model": "model_a"}
        results = []
        async for chunk in router.stream_chat_completions(request):
            results.append(chunk)

        assert results == ["chunk1", "chunk2"]
        infer_manager.get_instance.remote.assert_called_with("model_a")
        instance.executor_list[0].stream_execute_method.remote.assert_called_with(
            "stream_chat_completions", request_data=request
        )

    @pytest.mark.asyncio
    async def test_stream_chat_completions_exception(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        infer_manager = MagicMock()
        instance = make_infer_instance("model", 1)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)
        instance.executor_list[0].stream_execute_method.remote.side_effect = Exception("stream error")

        router = InferRouter(infer_manager)
        with pytest.raises(Exception, match="stream error"):
            async for _ in router.stream_chat_completions({"model": "model"}):
                pass

    @pytest.mark.asyncio
    async def test_stream_completions(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        infer_manager = MagicMock()
        instance = make_infer_instance("model", 1)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)
        instance.executor_list[0].stream_execute_method.remote.return_value = async_stream(["data"])

        router = InferRouter(infer_manager)
        results = []
        async for chunk in router.stream_completions({"model": "model"}):
            results.append(chunk)
        assert results == ["data"]


class TestCompletions:
    @pytest.mark.asyncio
    async def test_completions(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        fake_conf = fake_infer_router_env["fake_conf"]
        fake_conf.AgenticRLConf.load_config.return_value = make_config(False)

        infer_manager = MagicMock()
        instance = make_infer_instance("model", 2)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)
        instance.executor_list[0].execute_method.remote = AsyncMock(return_value="result")

        router = InferRouter(infer_manager)
        router.chat_server_mode = False
        router.init = AsyncMock()

        result = await router.completions({"model": "model"})
        assert result == "result"


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_chat_server_mode_true(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        fake_conf = fake_infer_router_env["fake_conf"]
        fake_conf.AgenticRLConf.load_config.return_value = make_config(True)

        infer_manager = MagicMock()
        instance = make_infer_instance("model", 1)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)
        instance.executor_list[0].execute_method.remote = AsyncMock(return_value="chat_result")

        router = InferRouter(infer_manager)
        router.chat_server_mode = True
        router.init = AsyncMock()

        request = {"model": "model", "extra_headers": {}}
        result = await router.chat_completions(request)
        assert result == "chat_result"

    @pytest.mark.asyncio
    async def test_chat_server_mode_false_success(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        mock_scheduler = fake_infer_router_env["mock_scheduler"]
        fake_conf = fake_infer_router_env["fake_conf"]
        fake_conf.AgenticRLConf.load_config.return_value = make_config(False)

        infer_manager = MagicMock()
        instance = make_infer_instance("model", 2)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)
        instance.executor_list[0].execute_method.remote = AsyncMock(return_value="response")
        mock_scheduler.schedule = AsyncMock(return_value="0-1")

        router = InferRouter(infer_manager)
        router.chat_server_mode = False
        router.init = AsyncMock()
        router.scheduler = mock_scheduler

        request = {"model": "model", "extra_headers": {"X-Request-Id": "app--req123"}}
        result = await router.chat_completions(request)
        assert result == "response"
        mock_scheduler.schedule.assert_awaited_once_with("app", "app--req123")
        mock_scheduler.release.assert_called_once_with("0-1", "app", "app--req123")

    @pytest.mark.asyncio
    async def test_chat_server_mode_false_schedule_fails(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        mock_scheduler = fake_infer_router_env["mock_scheduler"]
        fake_conf = fake_infer_router_env["fake_conf"]
        fake_conf.AgenticRLConf.load_config.return_value = make_config(False)

        infer_manager = MagicMock()
        instance = make_infer_instance("model", 2)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)
        mock_scheduler.schedule = AsyncMock(return_value=None)

        router = InferRouter(infer_manager)
        router.chat_server_mode = False
        router.init = AsyncMock()
        router.scheduler = mock_scheduler

        request = {"model": "model", "extra_headers": {"X-Request-Id": "app--req"}}
        result = await router.chat_completions(request)
        assert result is None


class TestBatchMethods:
    @pytest.mark.asyncio
    async def test_launch_server(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        infer_manager = MagicMock()
        instance = make_infer_instance("model", 2)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)
        for e in instance.executor_list:
            e.execute_method.remote = AsyncMock(return_value="ok")

        router = InferRouter(infer_manager)
        results = await router.launch_server("model")
        assert results == ["ok", "ok"]

    @pytest.mark.asyncio
    async def test_wake_up_default_kwargs(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        infer_manager = MagicMock()
        instance = make_infer_instance("model", 1)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)
        instance.executor_list[0].execute_method.remote = AsyncMock(return_value="awake")

        router = InferRouter(infer_manager)
        results = await router.wake_up("model")
        assert results == ["awake"]

    @pytest.mark.asyncio
    async def test_sleep(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        infer_manager = MagicMock()
        instance = make_infer_instance("model", 1)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)
        instance.executor_list[0].execute_method.remote = AsyncMock(return_value="zZz")

        router = InferRouter(infer_manager)
        results = await router.sleep("model")
        assert results == ["zZz"]

    @pytest.mark.asyncio
    async def test_update_weights(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        fake_ray = fake_infer_router_env["fake_ray"]
        infer_manager = MagicMock()
        instance = make_infer_instance("model", 2)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)

        router = InferRouter(infer_manager)
        result = await router.update_weights("model")
        fake_ray.get.assert_called()
        assert isinstance(result, list)


class TestVllmStats:
    @pytest.mark.asyncio
    async def test_vllm_statistics(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        fake_ray = fake_infer_router_env["fake_ray"]
        infer_manager = MagicMock()
        instance = make_infer_instance("model", 2)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)

        router = InferRouter(infer_manager)
        await router.vllm_statistics("model")
        fake_ray.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_prefix_cache(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        fake_ray = fake_infer_router_env["fake_ray"]
        infer_manager = MagicMock()
        instance = make_infer_instance("model", 2)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)

        router = InferRouter(infer_manager)
        await router.reset_prefix_cache("model")
        fake_ray.get.assert_called_once()


class TestWorkloadAndCancel:
    @pytest.mark.asyncio
    async def test_get_workload_chat_server_mode_true(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        router = InferRouter(MagicMock())
        router.chat_server_mode = True
        result = await router.get_workload()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_workload_normal(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        fake_conf = fake_infer_router_env["fake_conf"]
        fake_conf.AgenticRLConf.load_config.return_value = make_config(False)

        infer_manager = MagicMock()
        instance = make_infer_instance("default", 2)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)
        for i, e in enumerate(instance.executor_list):
            e.execute_method.remote = AsyncMock(return_value=f"workload_{i}")

        router = InferRouter(infer_manager)
        router.default_model_name = "default"
        router.chat_server_mode = False

        result = await router.get_workload()
        assert result == {"0": "workload_0", "1": "workload_1"}

    @pytest.mark.asyncio
    async def test_cancel_requests_chat_server_mode_true(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        router = InferRouter(MagicMock())
        router.chat_server_mode = True
        await router.cancel_requests()  # should not raise

    @pytest.mark.asyncio
    async def test_cancel_requests_with_running(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        mock_scheduler = fake_infer_router_env["mock_scheduler"]
        fake_conf = fake_infer_router_env["fake_conf"]
        fake_conf.AgenticRLConf.load_config.return_value = make_config(False)

        infer_manager = MagicMock()
        instance = make_infer_instance("default", 2)
        infer_manager.get_instance.remote = AsyncMock(return_value=instance)
        # simulate requests: only executor 0 has running requests
        mock_scheduler.running_reqs = {"0": ["req1"], "1": []}

        router = InferRouter(infer_manager)
        router.default_model_name = "default"
        router.chat_server_mode = False
        router.scheduler = mock_scheduler

        await router.cancel_requests()

        # Executor 0 gets empty kwargs, executor 1 gets the cancel payload
        instance.executor_list[0].execute_method.remote.assert_awaited()
        instance.executor_list[1].execute_method.remote.assert_awaited_with(
            method_name="cancel_requests", requests=["req1"]
        )
        assert mock_scheduler.running_reqs["0"] == []

    def test_stop(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        router = InferRouter(MagicMock())
        router.cancel_requests = AsyncMock()
        asyncio.run(router.stop())
        router.cancel_requests.assert_awaited_once()

    def test_reset(self, fake_infer_router_env):
        InferRouter = fake_infer_router_env["InferRouter"]
        router = InferRouter(MagicMock())
        router.reset()  # no-op
