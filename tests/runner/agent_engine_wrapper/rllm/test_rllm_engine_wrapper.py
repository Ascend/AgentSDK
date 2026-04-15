# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for rllm/rllm_engine_wrapper module: RLLMEngineWrapper."""

import json
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from agentic_rl.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask, Trajectory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dependencies():
    """Patch all external dependencies for RLLMEngineWrapper."""
    patches = {}

    mock_tokenizer = MagicMock()
    mock_tokenizer.name_or_path = "test-model"
    mock_tokenizer.__class__.__name__ = "QwenTokenizer"
    mock_tokenizer.bos_token = "<s>"
    mock_tokenizer.eos_token = "</s>"
    mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3])

    mock_agent_class = MagicMock()
    mock_env_class = MagicMock()
    mock_env_class.is_multithread_safe = MagicMock(return_value=True)
    mock_env_class.from_dict = MagicMock(return_value=MagicMock())

    mock_agent_info = {
        "agent_class": mock_agent_class,
        "agent_args": {},
        "env_class": mock_env_class,
        "env_args": {"max_steps": 3},
    }

    with (
        patch("agentic_rl.runner.agent_engine_wrapper.rllm.rllm_engine_wrapper.AutoTokenizer") as mock_auto_tok,
        patch("agentic_rl.runner.agent_engine_wrapper.rllm.rllm_engine_wrapper.Episode") as mock_episode,
        patch(
            "agentic_rl.runner.agent_engine_wrapper.rllm.rllm_engine_wrapper.load_object_by_path"
        ) as mock_load,
        patch(
            "agentic_rl.runner.agent_engine_wrapper.rllm.rllm_engine_wrapper.RLLMEngineWrapper.__init__",
            return_value=None,
        ) as mock_init,
    ):
        mock_auto_tok.from_pretrained = MagicMock(return_value=mock_tokenizer)
        mock_episode.remote = MagicMock(return_value=MagicMock())

        patches["mock_auto_tok"] = mock_auto_tok
        patches["mock_episode"] = mock_episode
        patches["mock_tokenizer"] = mock_tokenizer
        patches["mock_agent_class"] = mock_agent_class
        patches["mock_env_class"] = mock_env_class
        patches["mock_agent_info"] = mock_agent_info
        patches["mock_init"] = mock_init

        yield patches


def _make_wrapper(mock_deps):
    """Create a RLLMEngineWrapper with pre-set attributes (bypassing __init__)."""
    from agentic_rl.runner.agent_engine_wrapper.rllm.rllm_engine_wrapper import RLLMEngineWrapper

    wrapper = RLLMEngineWrapper.__new__(RLLMEngineWrapper)
    wrapper.server_addresses = ["0.0.0.0:8000"]
    wrapper.simplify_think_content = False
    wrapper.max_prompt_length = 8192
    wrapper.max_model_len = 16384
    wrapper.n_parallel_agents = 2
    wrapper.tokenizer_name_or_path = "test-model"
    wrapper.tokenizer = mock_deps["mock_tokenizer"]
    wrapper.sampling_params = {}
    wrapper.agent_class = mock_deps["mock_agent_class"]
    wrapper.agent_args = {}
    wrapper.env_class = mock_deps["mock_env_class"]
    wrapper.env_args = {"max_steps": 3, "tokenizer": mock_deps["mock_tokenizer"]}
    wrapper.compute_trajectory_reward_fn = None
    wrapper.max_steps = 3
    wrapper.overlong_filter = False
    wrapper.episode = MagicMock()
    wrapper.engine = None
    return wrapper


# ---------------------------------------------------------------------------
# Delegation method tests
# ---------------------------------------------------------------------------

class TestRLLMEngineWrapperDelegation:
    def test_update_envs_and_agents_delegates(self, mock_dependencies):
        wrapper = _make_wrapper(mock_dependencies)
        wrapper.engine = MagicMock()
        envs = [MagicMock()]
        agents = [MagicMock()]
        wrapper.update_envs_and_agents(envs, agents, 1, 0)
        wrapper.engine.update_envs_and_agents.assert_called_once_with(envs, agents, 1, 0)

    def test_update_env_and_agent_delegates(self, mock_dependencies):
        wrapper = _make_wrapper(mock_dependencies)
        wrapper.engine = MagicMock()
        env = MagicMock()
        agent = MagicMock()
        wrapper.update_env_and_agent("task_1", env, agent, 1, 0)
        wrapper.engine.update_env_and_agent.assert_called_once_with("task_1", env, agent, 1, 0)

    def test_release_env_and_agent_delegates(self, mock_dependencies):
        wrapper = _make_wrapper(mock_dependencies)
        wrapper.engine = MagicMock()
        wrapper.release_env_and_agent("task_1")
        wrapper.engine.release_env_and_agent.assert_called_once_with("task_1")

    def test_clear_cache_delegates(self, mock_dependencies):
        wrapper = _make_wrapper(mock_dependencies)
        wrapper.engine = MagicMock()
        wrapper.clear_cache()
        wrapper.engine.clear_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_request_delegates(self, mock_dependencies):
        wrapper = _make_wrapper(mock_dependencies)
        wrapper.engine = MagicMock()
        wrapper.engine.cancel_request = AsyncMock()
        task = AgentTask(task_id="t1", sample_id=0, iteration=0, agent_name="a", problem="p")
        await wrapper.cancel_request(task)
        wrapper.engine.cancel_request.assert_called_once_with(task)


# ---------------------------------------------------------------------------
# _create_engine tests
# ---------------------------------------------------------------------------

class TestCreateEngine:
    @patch("agentic_rl.runner.agent_engine_wrapper.rllm.agent_execution_engine.AgentExecutionEngine")
    def test_creates_engine_if_none(self, MockAEE, mock_dependencies):
        wrapper = _make_wrapper(mock_dependencies)
        wrapper.engine = None
        wrapper._create_engine()
        MockAEE.assert_called_once()
        assert wrapper.engine is not None

    @patch("agentic_rl.runner.agent_engine_wrapper.rllm.agent_execution_engine.AgentExecutionEngine")
    def test_does_not_recreate_engine(self, MockAEE, mock_dependencies):
        wrapper = _make_wrapper(mock_dependencies)
        existing_engine = MagicMock()
        wrapper.engine = existing_engine
        wrapper._create_engine()
        MockAEE.assert_not_called()
        assert wrapper.engine is existing_engine


# ---------------------------------------------------------------------------
# init_envs_and_agents tests
# ---------------------------------------------------------------------------

class TestInitEnvsAndAgents:
    def test_creates_envs_and_agents(self, mock_dependencies):
        wrapper = _make_wrapper(mock_dependencies)
        wrapper.engine = MagicMock()

        mock_task = MagicMock()
        mock_task.iteration = 1
        mock_task.sample_id = 0

        tasks = [mock_task]

        mock_env = MagicMock()
        mock_dependencies["mock_env_class"].from_dict = MagicMock(return_value=mock_env)
        mock_agent = MagicMock()
        mock_dependencies["mock_agent_class"].return_value = mock_agent

        envs = wrapper.init_envs_and_agents(tasks)
        assert len(envs) == 1
        wrapper.engine.update_envs_and_agents.assert_called_once()

    def test_json_string_task_converted_to_dict(self, mock_dependencies):
        wrapper = _make_wrapper(mock_dependencies)
        wrapper.engine = MagicMock()

        mock_task_0 = MagicMock()
        mock_task_0.iteration = 1
        mock_task_0.sample_id = 0
        mock_task_0.__str__ = lambda self: '{"key": "val"}'

        tasks = [mock_task_0]

        mock_env = MagicMock()
        mock_dependencies["mock_env_class"].from_dict = MagicMock(return_value=mock_env)

        envs = wrapper.init_envs_and_agents(tasks)
        assert len(envs) == 1
        wrapper.engine.update_envs_and_agents.assert_called_once()


# ---------------------------------------------------------------------------
# generate_trajectory tests
# ---------------------------------------------------------------------------

class TestGenerateTrajectory:

    @pytest.mark.asyncio
    async def test_generate_trajectory_basic(self, mock_dependencies):
        wrapper = _make_wrapper(mock_dependencies)

        mock_engine = MagicMock()
        mock_engine.init_router = MagicMock()
        mock_engine.update_env_and_agent = MagicMock()
        mock_engine.release_env_and_agent = MagicMock()

        mock_trajectory = MagicMock()

        async def mock_generator(*args, **kwargs):
            yield mock_trajectory

        mock_engine.trajectory_generator = mock_generator
        wrapper.engine = mock_engine
        wrapper._create_engine = MagicMock()

        task = AgentTask(
            task_id="t1", sample_id=0, iteration=1,
            agent_name="agent", problem="do something"
        )

        result = await wrapper.generate_trajectory(task, mode="Text")

        assert result is mock_trajectory
        mock_engine.release_env_and_agent.assert_called_once_with("t1")

    @pytest.mark.asyncio
    async def test_generate_trajectory_creates_engine(self, mock_dependencies):
        wrapper = _make_wrapper(mock_dependencies)

        mock_engine = MagicMock()
        mock_engine.init_router = MagicMock()
        mock_engine.update_env_and_agent = MagicMock()
        mock_engine.release_env_and_agent = MagicMock()

        async def mock_generator(*args, **kwargs):
            yield MagicMock()

        mock_engine.trajectory_generator = mock_generator

        create_called = False

        def track_create():
            nonlocal create_called
            create_called = True
            wrapper.engine = mock_engine

        wrapper._create_engine = track_create

        task = AgentTask(
            task_id="t1", sample_id=0, iteration=1,
            agent_name="agent", problem="test"
        )

        await wrapper.generate_trajectory(task, mode="Text")
        assert create_called

    @pytest.mark.asyncio
    async def test_generate_trajectory_passes_addresses(self, mock_dependencies):
        wrapper = _make_wrapper(mock_dependencies)

        mock_engine = MagicMock()
        mock_engine.init_router = MagicMock()
        mock_engine.update_env_and_agent = MagicMock()
        mock_engine.release_env_and_agent = MagicMock()

        async def mock_generator(*args, **kwargs):
            yield MagicMock()

        mock_engine.trajectory_generator = mock_generator
        wrapper.engine = mock_engine
        wrapper._create_engine = MagicMock()

        task = AgentTask(
            task_id="t1", sample_id=0, iteration=1,
            agent_name="agent", problem="test"
        )
        addresses = ["http://host1:8000", "http://host2:8000"]

        await wrapper.generate_trajectory(task, addresses=addresses)
        mock_engine.init_router.assert_called_once_with(addresses)

    @pytest.mark.asyncio
    async def test_generate_trajectory_uses_extra_args(self, mock_dependencies):
        wrapper = _make_wrapper(mock_dependencies)

        mock_engine = MagicMock()
        mock_engine.init_router = MagicMock()
        mock_engine.update_env_and_agent = MagicMock()
        mock_engine.release_env_and_agent = MagicMock()

        async def mock_generator(*args, **kwargs):
            yield MagicMock()

        mock_engine.trajectory_generator = mock_generator
        wrapper.engine = mock_engine
        wrapper._create_engine = MagicMock()

        task = AgentTask(
            task_id="t1", sample_id=0, iteration=1,
            agent_name="agent", problem="test",
            extra_args={"timeout": 30}
        )

        await wrapper.generate_trajectory(task)

        call_args = mock_dependencies["mock_env_class"].from_dict.call_args
        assert "timeout" in call_args[0][0]
