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
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Fixture: fake module tree for agent_router
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_agent_router_env():
    """Construct fully isolated environment for agent_router module."""

    # ---- Fake random ----
    fake_random = types.ModuleType("random")
    choice_mock = MagicMock()
    fake_random.choice = choice_mock

    # ---- Fake aura.base.log.loggers ----
    fake_loggers = types.ModuleType("aura.base.log.loggers")
    mock_logger = MagicMock()
    fake_logger_instance = MagicMock()
    fake_logger_instance.get_logger.return_value = mock_logger
    fake_loggers.Loggers = MagicMock(return_value=fake_logger_instance)

    # ---- Fake aura.runner.agent_engine_wrapper.base_engine_wrapper ----
    fake_engine_wrapper = types.ModuleType(
        "aura.runner.agent_engine_wrapper.base_engine_wrapper"
    )
    class FakeAgentTask:
        def __init__(self, agent_name="test_agent", **kwargs):
            self.agent_name = agent_name
    class FakeTrajectory:
        pass
    fake_engine_wrapper.AgentTask = FakeAgentTask
    fake_engine_wrapper.Trajectory = FakeTrajectory

    # ---- Fake aura.runner.agent_manager ----
    fake_agent_manager_mod = types.ModuleType("aura.runner.agent_manager")
    agent_manager_mock = MagicMock()
    fake_agent_manager_mod.get_or_create_agent_manager = AsyncMock(
        return_value=agent_manager_mock
    )

    # ---- Fake executor and instance mocks ----
    executor_mock = MagicMock()
    executor_mock.stream_execute_method = MagicMock()
    executor_mock.execute_method = MagicMock()
    # executor_mock.execute_method.remote will be set per test
    instance_mock = MagicMock()
    instance_mock.executor_list = [executor_mock]
    agent_manager_mock.get_instance.remote = AsyncMock(return_value=instance_mock)

    # Configure random.choice to return the executor by default
    choice_mock.side_effect = lambda lst: lst[0]

    # ---- Aura packages (to locate the real agent_router.py) ----
    import os
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = [os.path.join(base, "runner")]
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_log = types.ModuleType("aura.base.log")
    fake_aura_base_log.__path__ = []
    fake_aura_runner_agent_engine_wrapper = types.ModuleType(
        "aura.runner.agent_engine_wrapper"
    )
    fake_aura_runner_agent_engine_wrapper.__path__ = []

    fakes = {
        "random": fake_random,
        "aura.base.log.loggers": fake_loggers,
        "aura.runner.agent_engine_wrapper.base_engine_wrapper": fake_engine_wrapper,
        "aura.runner.agent_manager": fake_agent_manager_mod,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.base": fake_aura_base,
        "aura.base.log": fake_aura_base_log,
        "aura.runner.agent_engine_wrapper": fake_aura_runner_agent_engine_wrapper,
    }

    target = "aura.runner.agent_router"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.runner.agent_router as agent_router_mod
        # reset _router for isolation
        agent_router_mod.AgentRouter._router = None
        yield {
            "mod": agent_router_mod,
            "AgentRouter": agent_router_mod.AgentRouter,
            "logger_mock": mock_logger,
            "agent_manager_mock": agent_manager_mock,
            "agent_manager_mod": fake_agent_manager_mod,
            "executor_mock": executor_mock,
            "instance_mock": instance_mock,
            "choice_mock": choice_mock,
            "AgentTask": FakeAgentTask,
            "Trajectory": FakeTrajectory,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Helpers for awaitable streaming
# ---------------------------------------------------------------------------
async def async_gen_awaitable(items):
    """Yield futures that resolve to given strings, simulating remote stream."""
    for item in items:
        future = asyncio.Future()
        future.set_result(item)
        yield future


async def async_gen_empty_responses():
    """Stream that yields empty strings (to be skipped)."""
    futures = [asyncio.Future() for _ in range(2)]
    futures[0].set_result(None)
    futures[1].set_result("valid_chunk")
    for f in futures:
        yield f


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestAgentRouterCreate:
    @pytest.mark.asyncio
    async def test_create_first_time(self, fake_agent_router_env):
        """First create calls get_or_create_agent_manager and returns a router."""
        AgentRouter = fake_agent_router_env["AgentRouter"]
        agent_manager_mock = fake_agent_router_env["agent_manager_mock"]
        AgentRouter._router = None
        router = await AgentRouter.create()
        assert isinstance(router, AgentRouter)
        assert router.agent_manager is agent_manager_mock
        fake_agent_router_env["agent_manager_mod"].get_or_create_agent_manager.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_singleton(self, fake_agent_router_env):
        """Second create returns the same router without calling get_or_create_agent_manager again."""
        AgentRouter = fake_agent_router_env["AgentRouter"]
        AgentRouter._router = None
        router1 = await AgentRouter.create()
        fake_agent_router_env["agent_manager_mod"].get_or_create_agent_manager.reset_mock()
        router2 = await AgentRouter.create()
        assert router1 is router2
        fake_agent_router_env["agent_manager_mod"].get_or_create_agent_manager.assert_not_called()


class TestStreamGenerateTrajectory:
    @pytest.mark.asyncio
    async def test_stream_success(self, fake_agent_router_env):
        """Streaming returns yielded chunks and logs each of them."""
        AgentRouter = fake_agent_router_env["AgentRouter"]
        executor_mock = fake_agent_router_env["executor_mock"]
        logger_mock = fake_agent_router_env["logger_mock"]

        executor_mock.stream_execute_method.remote.return_value = async_gen_awaitable(
            ["chunk1", "chunk2"]
        )
        router = AgentRouter(fake_agent_router_env["agent_manager_mock"])
        task = fake_agent_router_env["AgentTask"](agent_name="agent1")

        results = []
        async for chunk in router.stream_generate_trajectory(task):
            results.append(chunk)

        assert results == ["chunk1", "chunk2"]
        assert logger_mock.error.call_count == 2
        executor_mock.stream_execute_method.remote.assert_called_once_with(
            "stream_generate_trajectory", task
        )

    @pytest.mark.asyncio
    async def test_stream_skips_empty_responses(self, fake_agent_router_env):
        """Empty (falsy) responses are skipped and not yielded."""
        AgentRouter = fake_agent_router_env["AgentRouter"]
        executor_mock = fake_agent_router_env["executor_mock"]

        executor_mock.stream_execute_method.remote.return_value = async_gen_empty_responses()
        router = AgentRouter(fake_agent_router_env["agent_manager_mock"])
        task = fake_agent_router_env["AgentTask"]()

        results = []
        async for chunk in router.stream_generate_trajectory(task):
            results.append(chunk)

        assert results == ["valid_chunk"]

    @pytest.mark.asyncio
    async def test_stream_exception_raised(self, fake_agent_router_env):
        """If the remote stream raises, the exception is re-raised."""
        AgentRouter = fake_agent_router_env["AgentRouter"]
        executor_mock = fake_agent_router_env["executor_mock"]

        async def failing_gen():
            raise RuntimeError("remote failure")
            yield

        executor_mock.stream_execute_method.remote.return_value = failing_gen()
        router = AgentRouter(fake_agent_router_env["agent_manager_mock"])
        task = fake_agent_router_env["AgentTask"]()

        with pytest.raises(RuntimeError, match="remote failure"):
            async for _ in router.stream_generate_trajectory(task):
                pass


class TestGenerateTrajectory:
    @pytest.mark.asyncio
    async def test_generate_trajectory(self, fake_agent_router_env):
        """generate_trajectory calls executor and returns the result."""
        AgentRouter = fake_agent_router_env["AgentRouter"]
        executor_mock = fake_agent_router_env["executor_mock"]
        executor_mock.execute_method.remote = AsyncMock(return_value="trajectory_result")

        router = AgentRouter(fake_agent_router_env["agent_manager_mock"])
        task = fake_agent_router_env["AgentTask"](agent_name="agent1")
        result = await router.generate_trajectory(task, mode="Text", addresses=["addr1"])

        assert result == "trajectory_result"
        executor_mock.execute_method.remote.assert_awaited_once_with(
            "generate_trajectory", task=task, mode="Text", addresses=["addr1"], server_handles=None
        )


class TestGenerateTrajectories:
    @pytest.mark.asyncio
    async def test_multiple_tasks(self, fake_agent_router_env):
        """generate_trajectories runs multiple tasks concurrently and returns all results."""
        AgentRouter = fake_agent_router_env["AgentRouter"]
        executor_mock = fake_agent_router_env["executor_mock"]
        executor_mock.execute_method.remote = AsyncMock(side_effect=["r1", "r2"])

        router = AgentRouter(fake_agent_router_env["agent_manager_mock"])
        tasks = [
            fake_agent_router_env["AgentTask"](agent_name="a1"),
            fake_agent_router_env["AgentTask"](agent_name="a2"),
        ]
        results = await router.generate_trajectories(tasks, mode="Text")
        assert results == ["r1", "r2"]

    @pytest.mark.asyncio
    async def test_empty_tasks(self, fake_agent_router_env):
        """Empty task list returns an empty list."""
        AgentRouter = fake_agent_router_env["AgentRouter"]
        router = AgentRouter(fake_agent_router_env["agent_manager_mock"])
        results = await router.generate_trajectories([])
        assert results == []


class TestCancelRequest:
    @pytest.mark.asyncio
    async def test_cancel_request(self, fake_agent_router_env):
        """cancel_request calls executor's cancel_request method."""
        AgentRouter = fake_agent_router_env["AgentRouter"]
        executor_mock = fake_agent_router_env["executor_mock"]
        executor_mock.execute_method.remote = AsyncMock(return_value=None)

        router = AgentRouter(fake_agent_router_env["agent_manager_mock"])
        task = fake_agent_router_env["AgentTask"](agent_name="agent1")
        await router.cancel_request(task)

        executor_mock.execute_method.remote.assert_awaited_once_with("cancel_request", task)


class TestClearCache:
    @pytest.mark.asyncio
    async def test_clear_cache(self, fake_agent_router_env):
        """clear_cache calls executor's clear_cache method."""
        AgentRouter = fake_agent_router_env["AgentRouter"]
        executor_mock = fake_agent_router_env["executor_mock"]
        executor_mock.execute_method.remote = AsyncMock(return_value=None)

        router = AgentRouter(fake_agent_router_env["agent_manager_mock"])
        await router.clear_cache("agent_name")

        executor_mock.execute_method.remote.assert_awaited_once_with("clear_cache")
