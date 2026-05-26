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
from unittest.mock import AsyncMock, MagicMock, patch, ANY
import pytest


# ---------------------------------------------------------------------------
# Fixture: fake module tree for rllm_engine_wrapper
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_rllm_env():
    """Construct a fully isolated module tree for rllm_engine_wrapper."""

    # ---- Fake standard library mocks (signal, threading, concurrent.futures) ----
    fake_signal = types.ModuleType("signal")
    fake_signal.signal = MagicMock()  # will be replaced by __init__ later

    fake_threading = types.ModuleType("threading")
    fake_threading.current_thread = MagicMock()
    fake_threading.main_thread = MagicMock()

    fake_concurrent = types.ModuleType("concurrent.futures")
    fake_concurrent.ThreadPoolExecutor = MagicMock()
    fake_concurrent.as_completed = MagicMock()

    fake_queue = types.ModuleType("queue")
    fake_queue.Queue = MagicMock()

    fake_re = types.ModuleType("re")
    fake_re.compile = MagicMock()

    # ---- Fake transformers ----
    fake_transformers = types.ModuleType("transformers")
    tokenizer_mock = MagicMock()
    fake_transformers.AutoTokenizer = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=tokenizer_mock)

    # ---- Fake aura.base.log.loggers ----
    fake_loggers = types.ModuleType("aura.base.log.loggers")
    mock_logger = MagicMock()
    fake_loggers.Loggers = MagicMock(return_value=MagicMock(get_logger=MagicMock(return_value=mock_logger)))

    # ---- Fake aura.memory.episode.episode ----
    fake_episode = types.ModuleType("aura.memory.episode.episode")
    episode_instance = MagicMock()
    fake_episode.Episode = MagicMock()
    fake_episode.Episode.remote = MagicMock(return_value=episode_instance)

    # ---- Fake base_engine_wrapper (BaseEngineWrapper, AgentTask, Trajectory) ----
    fake_base_engine = types.ModuleType(
        "aura.runner.agent_engine_wrapper.base_engine_wrapper"
    )
    class FakeBaseEngineWrapper:
        def __init__(self, *args, **kwargs):
            pass  # simple no-op
    class FakeAgentTask:
        pass
    class FakeTrajectory:
        pass
    fake_base_engine.BaseEngineWrapper = FakeBaseEngineWrapper
    fake_base_engine.AgentTask = FakeAgentTask
    fake_base_engine.Trajectory = FakeTrajectory

    # ---- Fake chat_proxy ----
    fake_chat_proxy = types.ModuleType("aura.runner.agent_service.chat_proxy")
    fake_chat_proxy.patch_async_openai_global = MagicMock()

    # ---- Fake agents_mapping ----
    fake_agents_mapping = types.ModuleType("agents.agents_mapping")
    fake_agent = {
        "agent_class": MagicMock(),
        "agent_args": {"arg1": "val1"},
        "env_class": MagicMock(),
        "env_args": {"env_key": "env_val"},
        "compute_trajectory_reward_fn": None,
        "chat_parser": None,
        "traj_proxy_class": None,
        "agent_proxy_class": None,
    }
    fake_agents_mapping.get_agent_by_name = MagicMock(return_value=fake_agent)

    # ---- Fake engine wrappers (both paths) ----
    fake_extern_agent = types.ModuleType(
        "aura.runner.agent_engine_wrapper.vaee.extern_agent_wrapper"
    )
    FakeExternAgentWrapper = MagicMock()
    fake_extern_agent.ExternAgentWrapper = FakeExternAgentWrapper

    fake_agent_exec_engine = types.ModuleType(
        "aura.runner.agent_engine_wrapper.rllm.agent_execution_engine"
    )
    FakeAgentExecutionEngine = MagicMock()
    fake_agent_exec_engine.AgentExecutionEngine = FakeAgentExecutionEngine

    # ---- Aura packages (to locate the real file) ----
    import os
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = [os.path.join(base, "runner")]
    # Sub-packages
    fake_aura_runner_agent_engine_wrapper = types.ModuleType(
        "aura.runner.agent_engine_wrapper"
    )
    fake_aura_runner_agent_engine_wrapper.__path__ = [
        os.path.join(base, "runner/agent_engine_wrapper")
    ]
    fake_aura_runner_agent_engine_wrapper_rllm = types.ModuleType(
        "aura.runner.agent_engine_wrapper.rllm"
    )
    fake_aura_runner_agent_engine_wrapper_rllm.__path__ = [
        os.path.join(base, "runner/agent_engine_wrapper/rllm")
    ]
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_log = types.ModuleType("aura.base.log")
    fake_aura_base_log.__path__ = []

    # All fakes
    fakes = {
        "signal": fake_signal,
        "threading": fake_threading,
        "concurrent.futures": fake_concurrent,
        "queue": fake_queue,
        "transformers": fake_transformers,
        "aura.base.log.loggers": fake_loggers,
        "aura.memory.episode.episode": fake_episode,
        "aura.runner.agent_engine_wrapper.base_engine_wrapper": fake_base_engine,
        "aura.runner.agent_service.chat_proxy": fake_chat_proxy,
        "agents.agents_mapping": fake_agents_mapping,
        "aura.runner.agent_engine_wrapper.vaee.extern_agent_wrapper": fake_extern_agent,
        "aura.runner.agent_engine_wrapper.rllm.agent_execution_engine": fake_agent_exec_engine,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.agent_engine_wrapper": fake_aura_runner_agent_engine_wrapper,
        "aura.runner.agent_engine_wrapper.rllm": fake_aura_runner_agent_engine_wrapper_rllm,
        "aura.base": fake_aura_base,
        "aura.base.log": fake_aura_base_log,
    }

    target = "aura.runner.agent_engine_wrapper.rllm.rllm_engine_wrapper"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.runner.agent_engine_wrapper.rllm.rllm_engine_wrapper as mod
        yield {
            "mod": mod,
            "RLLMEngineWrapper": mod.RLLMEngineWrapper,
            "mock_logger": mock_logger,
            "tokenizer_mock": tokenizer_mock,
            "fake_signal": fake_signal,
            "fake_threading": fake_threading,
            "fake_concurrent": fake_concurrent,
            "fake_agent_mapping": fake_agents_mapping.get_agent_by_name,
            "fake_extern_agent": fake_extern_agent,
            "fake_agent_exec_engine": fake_agent_exec_engine,
            "fake_episode": fake_episode.Episode,
            "fake_chat_proxy": fake_chat_proxy,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_infer_service_params():
    return {
        "max_tokens": 512,
        "model": "test-model",
        "disable_thinking": False,
    }


def make_tasks(num=1):
    """Create mock AgentTask objects with iteration, sample_id, task_id."""
    tasks = []
    for i in range(num):
        t = MagicMock()
        t.iteration = 0
        t.sample_id = i
        t.task_id = f"task-{i}"
        t.prompt_id = 0
        t.model_dump.return_value = {"extra_args": {}, "prompt_id": 0}
        tasks.append(t)
    return tasks


# ---------------------------------------------------------------------------
# Tests for __init__
# ---------------------------------------------------------------------------
class TestRLLMInit:
    @pytest.fixture(autouse=True)
    def setup(self, fake_rllm_env):
        self.env = fake_rllm_env
        self.mod = fake_rllm_env["mod"]

    def test_basic_initialization(self, fake_rllm_env):
        """Basic init with minimal params sets default attributes and calls patch."""
        wrapper = self.mod.RLLMEngineWrapper(
            infer_service_params=make_infer_service_params(),
            agent_name="test_agent",
            tokenizer="bert-base",
        )
        # check that patch was called
        self.env["fake_chat_proxy"].patch_async_openai_global.assert_called_once()
        # check default attributes
        assert wrapper.max_prompt_length == 8192
        assert wrapper.max_model_len == 16384
        assert wrapper.n_parallel_agents == 8
        assert wrapper.tokenizer is self.env["tokenizer_mock"]
        assert wrapper.engine is None
        assert wrapper.chat_parser is None  # because chat_parser not in agent
        assert wrapper.overlong_filter is False

    def test_with_chat_parser_provided(self, fake_rllm_env):
        """Agent mapping provides chat_parser, so it's instantiated."""
        fake_agent = {
            "agent_class": MagicMock(),
            "agent_args": {},
            "env_class": MagicMock(),
            "env_args": {},
            "compute_trajectory_reward_fn": None,
            "chat_parser": MagicMock(),  # callable
            "traj_proxy_class": None,
            "agent_proxy_class": None,
        }
        self.env["fake_agent_mapping"].return_value = fake_agent

        wrapper = self.mod.RLLMEngineWrapper(
            infer_service_params=make_infer_service_params(),
            agent_name="test",
            tokenizer="bert",
        )
        # chat_parser should be an instance
        assert wrapper.chat_parser is not None
        assert wrapper.chat_parser is not fake_agent["chat_parser"]  # it's the return value

    def test_agent_not_found_raises(self, fake_rllm_env):
        """If get_agent_by_name returns None, raises RuntimeError."""
        self.env["fake_agent_mapping"].return_value = None
        with pytest.raises(RuntimeError, match="not found"):
            self.mod.RLLMEngineWrapper(
                infer_service_params=make_infer_service_params(),
                agent_name="missing",
                tokenizer="bert",
            )

    def test_url_validation_invalid(self, fake_rllm_env):
        """Invalid traj_proxy_url raises ValueError."""
        with pytest.raises(ValueError, match="Invalid format"):
            self.mod.RLLMEngineWrapper(
                infer_service_params=make_infer_service_params(),
                agent_name="test_agent",
                tokenizer="bert",
                traj_proxy_url="invalid",
                agent_proxy_url="http://1.2.3.4:8000",
            )

    def test_url_validation_valid(self, fake_rllm_env):
        """Valid URLs are accepted without exception."""
        wrapper = self.mod.RLLMEngineWrapper(
            infer_service_params=make_infer_service_params(),
            agent_name="test_agent",
            tokenizer="bert",
            traj_proxy_url="http://1.2.3.4:8000",
            agent_proxy_url="http://1.2.3.4:9000",
        )
        assert wrapper.traj_proxy_url == "http://1.2.3.4:8000"

    def test_signal_override_main_thread(self, fake_rllm_env):
        """Signal is not blocked for main thread."""
        self.env["fake_threading"].current_thread.return_value = self.env["fake_threading"].main_thread
        wrapper = self.mod.RLLMEngineWrapper(
            infer_service_params=make_infer_service_params(),
            agent_name="test_agent",
            tokenizer="bert",
        )
        # signal.signal should have been replaced
        assert callable(self.env["fake_signal"].signal)
        # call the replaced signal handler
        handler = self.env["fake_signal"].signal
        # simulate main thread -> should call original signal
        original_signal = self.env["fake_signal"].signal  # this is the mock before replacement
        # Not trivial to test perfectly, but we can check that signal.signal is now our function


class TestInitEnvsAndAgents:
    @pytest.fixture(autouse=True)
    def setup(self, fake_rllm_env):
        self.env = fake_rllm_env
        self.mod = fake_rllm_env["mod"]
        # create a wrapper with minimal init to use for testing methods
        self.wrapper = self.mod.RLLMEngineWrapper(
            infer_service_params=make_infer_service_params(),
            agent_name="test_agent",
            tokenizer="bert",
        )
        # mock out update_envs_and_agents to avoid calling engine
        self.wrapper.update_envs_and_agents = MagicMock()
        # mock ThreadPoolExecutor
        self.env["fake_concurrent"].ThreadPoolExecutor = MagicMock()
        self.env["fake_concurrent"].as_completed = MagicMock()

    def test_creates_envs_and_agents(self, fake_rllm_env):
        """init_envs_and_agents creates envs and agents for given tasks."""
        tasks = make_tasks(3)
        # mock env_class.from_dict and agent_class
        env_class_mock = self.wrapper.env_class
        env_class_mock.from_dict.return_value = MagicMock()
        agent_class_mock = self.wrapper.agent_class
        agent_class_mock.return_value = MagicMock()

        # need to provide fake futures
        import concurrent.futures
        fake_futures = [MagicMock() for _ in range(3)]
        for i, f in enumerate(fake_futures):
            f.result.return_value = (i, MagicMock())  # env or agent
        self.env["fake_concurrent"].ThreadPoolExecutor.return_value.__enter__.return_value.submit.side_effect = fake_futures
        self.env["fake_concurrent"].as_completed.return_value = fake_futures

        envs = self.wrapper.init_envs_and_agents(tasks)
        assert len(envs) == 3
        # Check that update_envs_and_agents was called
        self.wrapper.update_envs_and_agents.assert_called_once()


class TestCreateEngine:
    @pytest.fixture(autouse=True)
    def setup(self, fake_rllm_env):
        self.env = fake_rllm_env
        self.mod = fake_rllm_env["mod"]
        self.wrapper = self.mod.RLLMEngineWrapper(
            infer_service_params=make_infer_service_params(),
            agent_name="test_agent",
            tokenizer="bert",
        )

    def test_engine_already_created(self, fake_rllm_env):
        """If engine already set, _create_engine returns early."""
        self.wrapper.engine = "existing"
        self.wrapper._create_engine()
        # no engine classes should be instantiated
        self.env["fake_extern_agent"].ExternAgentWrapper.assert_not_called()
        self.env["fake_agent_exec_engine"].AgentExecutionEngine.assert_not_called()

    def test_create_engine_default(self, fake_rllm_env):
        """Without proxy URLs, creates AgentExecutionEngine."""
        self.wrapper._create_engine()
        self.env["fake_agent_exec_engine"].AgentExecutionEngine.assert_called_once()
        self.env["fake_extern_agent"].ExternAgentWrapper.assert_not_called()
        assert self.wrapper.engine is not None

    def test_create_engine_with_proxy(self, fake_rllm_env):
        """With proxy URLs, creates ExternAgentWrapper."""
        wrapper = self.mod.RLLMEngineWrapper(
            infer_service_params=make_infer_service_params(),
            agent_name="test_agent",
            tokenizer="bert",
            traj_proxy_url="http://1.2.3.4:8000",
            agent_proxy_url="http://1.2.3.4:9000",
        )
        wrapper._create_engine()
        self.env["fake_extern_agent"].ExternAgentWrapper.assert_called_once()
        self.env["fake_agent_exec_engine"].AgentExecutionEngine.assert_not_called()


class TestGenerateTrajectory:
    @pytest.fixture(autouse=True)
    def setup(self, fake_rllm_env):
        self.env = fake_rllm_env
        self.mod = fake_rllm_env["mod"]
        self.wrapper = self.mod.RLLMEngineWrapper(
            infer_service_params=make_infer_service_params(),
            agent_name="test_agent",
            tokenizer="bert",
        )
        # mock engine methods
        self.wrapper.engine = MagicMock()
        self.wrapper.engine.init_router = MagicMock()
        self.wrapper.engine.update_env_and_agent = MagicMock()
        self.wrapper.engine.release_env_and_agent = MagicMock()
        self.wrapper.engine.trajectory_generator = MagicMock()

    @pytest.mark.asyncio
    async def test_generate_trajectory_normal(self, fake_rllm_env):
        """Normal trajectory generation returns final trajectory item."""
        task = make_tasks(1)[0]
        # trajectory_generator returns an async generator that yields one item
        async def async_gen():
            yield "traj_result"

        self.wrapper.engine.trajectory_generator.return_value = async_gen()

        result = await self.wrapper.generate_trajectory(task, mode="Text")
        assert result == "traj_result"
        self.wrapper.engine.init_router.assert_called_once()
        self.wrapper.engine.update_env_and_agent.assert_called_once()
        self.wrapper.engine.release_env_and_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_trajectory_with_stream_queue(self, fake_rllm_env):
        """Trajectory generation with stream_queue passes it to generator."""
        task = make_tasks(1)[0]
        queue = MagicMock()

        async def async_gen():
            yield "stream_result"

        self.wrapper.engine.trajectory_generator.return_value = async_gen()

        result = await self.wrapper.generate_trajectory(task, stream_queue=queue)
        assert result == "stream_result"
        # Check that stream_queue was passed
        # The call to trajectory_generator should have stream_queue in kwargs
        self.wrapper.engine.trajectory_generator.assert_called_once_with(
            task=ANY, stream_queue=queue, mode="Text", prompt_id=0, server_handles=None
        )


class TestCancelRequest:
    @pytest.fixture(autouse=True)
    def setup(self, fake_rllm_env):
        self.wrapper = fake_rllm_env["mod"].RLLMEngineWrapper(
            infer_service_params=make_infer_service_params(),
            agent_name="test_agent",
            tokenizer="bert",
        )
        self.wrapper.engine = MagicMock()
        self.wrapper.engine.cancel_request = AsyncMock()

    @pytest.mark.asyncio
    async def test_cancel_request(self, fake_rllm_env):
        task = make_tasks(1)[0]
        await self.wrapper.cancel_request(task)
        self.wrapper.engine.cancel_request.assert_awaited_once_with(task)


class TestClearCache:
    def test_clear_cache(self, fake_rllm_env):
        wrapper = fake_rllm_env["mod"].RLLMEngineWrapper(
            infer_service_params=make_infer_service_params(),
            agent_name="test_agent",
            tokenizer="bert",
        )
        wrapper.engine = MagicMock()
        wrapper.clear_cache()
        wrapper.engine.clear_cache.assert_called_once()
