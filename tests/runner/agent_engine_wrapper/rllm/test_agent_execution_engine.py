# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for rllm/agent_execution_engine module."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import torch

from agentic_rl.runner.agent_engine_wrapper.base.agent.base_agent import (
    Action,
    BaseAgent,
    Step,
    Trajectory,
)
from agentic_rl.runner.agent_engine_wrapper.base_engine_wrapper import AgentTask
from agentic_rl.runner.agent_engine_wrapper.rllm.agent_execution_engine import (
    AgentExecutionEngine,
    AsyncAgentExecutionEngine,
    _generate_key,
    create_application_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.name_or_path = "test-model"
    tokenizer.__class__.__name__ = "QwenTokenizer"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
    tokenizer.apply_chat_template = MagicMock(return_value="formatted")
    return tokenizer


@pytest.fixture
def mock_chat_parser():
    parser = MagicMock()
    parser.assistant_token = "<assistant>"
    parser.parse = MagicMock(return_value="parsed_prompt")
    return parser


@pytest.fixture
def mock_env_class():
    env_cls = MagicMock()
    env_cls.is_multithread_safe = MagicMock(return_value=True)
    return env_cls


@pytest.fixture
def mock_agent_class():
    return MagicMock()


@pytest.fixture
def engine(mock_tokenizer, mock_chat_parser, mock_env_class, mock_agent_class):
    with patch("agentic_rl.runner.agent_engine_wrapper.rllm.agent_execution_engine.ChatTemplateParser") as mock_ctp:
        mock_ctp.get_parser = MagicMock(return_value=mock_chat_parser)
        eng = AgentExecutionEngine(
            tokenizer=mock_tokenizer,
            server_addresses=None,
            chat_parser=mock_chat_parser,
            n_parallel_agents=2,
            max_steps=5,
            max_prompt_length=1024,
            max_model_len=4096,
            agent_class=mock_agent_class,
            env_class=mock_env_class,
            env_args={},
            agent_args={},
        )
        return eng


# ---------------------------------------------------------------------------
# create_application_id tests
# ---------------------------------------------------------------------------

class TestCreateApplicationId:
    def test_returns_string(self):
        result = create_application_id(0)
        assert isinstance(result, str)

    def test_starts_with_prompt_id(self):
        result = create_application_id(42)
        assert result.startswith("42-")

    def test_unique_ids(self):
        ids = {create_application_id(0) for _ in range(100)}
        assert len(ids) == 100

    def test_contains_pid(self):
        result = create_application_id(0)
        assert str(os.getpid()) in result


# ---------------------------------------------------------------------------
# _generate_key tests
# ---------------------------------------------------------------------------

class TestGenerateKey:
    def test_dict_task(self):
        task = {"task_id": "abc", "prompt_id": 1}
        result = _generate_key(task)
        assert result == "abc_1"

    def test_agent_task(self):
        task = AgentTask(
            task_id="xyz", sample_id=0, iteration=0,
            agent_name="a", problem="p", prompt_id=5
        )
        result = _generate_key(task)
        assert result == "xyz_5"

    def test_other_type_returns_none(self):
        result = _generate_key("string_task")
        assert result is None

    def test_none_returns_none(self):
        result = _generate_key(None)
        assert result is None


# ---------------------------------------------------------------------------
# AgentExecutionEngine __init__ tests
# ---------------------------------------------------------------------------

class TestAgentExecutionEngineInit:
    def test_basic_attributes(self, engine, mock_tokenizer, mock_chat_parser):
        assert engine.tokenizer is mock_tokenizer
        assert engine.chat_parser is mock_chat_parser
        assert engine.n_parallel_agents == 2
        assert engine.max_steps == 5
        assert engine.max_prompt_length == 1024
        assert engine.max_model_len == 4096

    def test_default_gamma(self, engine):
        assert engine.gamma == 0.2

    def test_default_retry_limit(self, engine):
        assert engine.retry_limit == 3

    def test_agents_initialized(self, engine):
        assert len(engine.agents) == 2
        assert all(a is None for a in engine.agents)

    def test_envs_initialized(self, engine):
        assert len(engine.envs) == 2
        assert all(e is None for e in engine.envs)

    def test_env_not_multithread_safe_raises(self, mock_tokenizer, mock_chat_parser):
        env_cls = MagicMock()
        env_cls.is_multithread_safe = MagicMock(return_value=False)
        with pytest.raises(TypeError, match="multi-thread safe"):
            AgentExecutionEngine(
                tokenizer=mock_tokenizer,
                chat_parser=mock_chat_parser,
                env_class=env_cls,
            )

    def test_none_env_class_allowed(self, mock_tokenizer, mock_chat_parser):
        eng = AgentExecutionEngine(
            tokenizer=mock_tokenizer,
            chat_parser=mock_chat_parser,
            env_class=None,
        )
        assert eng.env_class is None


# ---------------------------------------------------------------------------
# update_envs_and_agents tests
# ---------------------------------------------------------------------------

class TestUpdateEnvsAndAgents:
    def test_updates_envs_and_agents(self, engine):
        envs = [MagicMock(), MagicMock()]
        agents = [MagicMock(), MagicMock()]
        engine.update_envs_and_agents(envs, agents, iteration=1, sample_id=0)
        assert engine.envs is envs
        assert engine.agents is agents
        assert engine.n_parallel_agents == 2
        assert engine.iteration == 1
        assert engine.sample_id == 0

    def test_mismatched_lengths_raises(self, engine):
        envs = [MagicMock()]
        agents = [MagicMock(), MagicMock()]
        with pytest.raises(ValueError, match="Number of agents must equal"):
            engine.update_envs_and_agents(envs, agents, iteration=1, sample_id=0)

    def test_sets_env_idx(self, engine):
        env0 = MagicMock()
        env1 = MagicMock()
        engine.update_envs_and_agents([env0, env1], [MagicMock(), MagicMock()], 1, 0)
        env0.__setattr__("idx", 0)
        env1.__setattr__("idx", 1)


# ---------------------------------------------------------------------------
# update_env_and_agent / release tests
# ---------------------------------------------------------------------------

class TestUpdateAndReleaseEnvAgent:
    def test_update_env_and_agent(self, engine):
        env = MagicMock()
        agent = MagicMock()
        engine.update_env_and_agent("1", env, agent, iteration=2, sample_id=3)
        assert engine.env_dict["1"] is env
        assert engine.agent_dict["1"] is agent
        assert engine.iteration == 2

    def test_release_env_and_agent(self, engine):
        engine.env_dict["1"] = MagicMock()
        engine.agent_dict["1"] = MagicMock()
        engine.release_env_and_agent("1")
        assert "1" not in engine.env_dict
        assert "1" not in engine.agent_dict

    def test_release_nonexistent_key_no_error(self, engine):
        engine.release_env_and_agent("999")


# ---------------------------------------------------------------------------
# store/pop application_id tests
# ---------------------------------------------------------------------------

class TestApplicationIdManagement:
    def test_store_and_pop(self, engine):
        task = {"task_id": "t1", "prompt_id": 0}
        engine.store_application_id(task, "app_123")
        result = engine.pop_application_id(task)
        assert result == "app_123"

    def test_pop_removes_key(self, engine):
        task = {"task_id": "t1", "prompt_id": 0}
        engine.store_application_id(task, "app_123")
        engine.pop_application_id(task)
        result = engine.pop_application_id(task)
        assert result is None

    def test_pop_nonexistent_returns_none(self, engine):
        task = {"task_id": "missing", "prompt_id": 0}
        assert engine.pop_application_id(task) is None

    def test_clear_cache(self, engine):
        task = {"task_id": "t1", "prompt_id": 0}
        engine.store_application_id(task, "app_1")
        engine.clear_cache()
        assert engine.pop_application_id(task) is None

    def test_store_with_none_key(self, engine):
        engine.store_application_id("string_task", "app_1")
        assert len(engine.application_ids) == 0


# ---------------------------------------------------------------------------
# init_router tests
# ---------------------------------------------------------------------------

class TestInitRouter:
    def test_none_addresses_no_router(self, engine):
        engine.router = None
        engine.init_router(None)
        assert engine.router is None

    def test_none_list_no_router(self, engine):
        engine.router = None
        engine.init_router([None])
        assert engine.router is None

    def test_existing_router_not_replaced(self, engine):
        existing_router = MagicMock()
        engine.router = existing_router
        engine.init_router(["http://localhost:8000"])
        assert engine.router is existing_router

    def test_creates_router(self, engine):
        mock_router_instance = MagicMock()
        mock_router_class = MagicMock()
        mock_router_class.create = MagicMock(return_value=mock_router_instance)

        mock_router_module = MagicMock()
        mock_router_module.Router = mock_router_class

        with patch.dict('sys.modules', {'agentic_rl.runner.scheduler.router': mock_router_module}):
            engine.router = None
            engine.init_router(["http://localhost:8000"])
            assert engine.router is not None
            assert engine.router == mock_router_instance
            mock_router_class.create.assert_called_once()


# ---------------------------------------------------------------------------
# cancel_trajectories / reset tests
# ---------------------------------------------------------------------------

class TestCancelAndReset:

    @pytest.mark.asyncio
    async def test_cancel_sets_stop(self, engine):
        engine.router = MagicMock()
        engine.router.stop = AsyncMock()
        await engine.cancel_trajectories()
        assert engine.stop is True

    def test_reset_clears_stop(self, engine):
        engine.router = MagicMock()
        engine.stop = True
        engine.reset()
        assert engine.stop is False
        engine.router.reset.assert_called_once()


# ---------------------------------------------------------------------------
# assemble_steps tests
# ---------------------------------------------------------------------------

class TestAssembleSteps:
    def test_single_step(self, engine):
        steps = [{
            "prompt": "prompt",
            "response": "response",
            "prompt_ids": [1, 2, 3],
            "completion_ids": [4, 5],
            "logprobs": [0.1, 0.2],
        }]
        prompt_tokens, response_tokens, response_masks, is_valid = engine.assemble_steps(steps, masked_out=False)
        assert torch.equal(prompt_tokens, torch.tensor([1, 2, 3], dtype=torch.long))
        assert torch.equal(response_tokens, torch.tensor([4, 5], dtype=torch.long))
        assert torch.equal(response_masks, torch.tensor([1, 1], dtype=torch.long))
        assert is_valid is True

    def test_two_cumulative_steps(self, engine):
        steps = [
            {
                "prompt": "p1",
                "response": "r1",
                "prompt_ids": [1, 2, 3],
                "completion_ids": [4, 5],
                "logprobs": [0.1, 0.2],
            },
            {
                "prompt": "p2",
                "response": "r2",
                "prompt_ids": [1, 2, 3, 4, 5, 6, 7],
                "completion_ids": [8, 9],
                "logprobs": [0.3, 0.4],
            },
        ]
        prompt_tokens, response_tokens, response_masks, is_valid = engine.assemble_steps(steps, masked_out=False)
        assert torch.equal(prompt_tokens, torch.tensor([1, 2, 3], dtype=torch.long))
        # response_tokens: [4, 5] (step0 completion) + [6, 7] (env tokens) + [8, 9] (step1 completion)
        assert torch.equal(response_tokens, torch.tensor([4, 5, 6, 7, 8, 9], dtype=torch.long))
        # masks: [1, 1] (step0 completion) + [0, 0] (env) + [1, 1] (step1 completion)
        assert torch.equal(response_masks, torch.tensor([1, 1, 0, 0, 1, 1], dtype=torch.long))

    def test_masked_out_zeroes_all_masks(self, engine):
        steps = [{
            "prompt": "p",
            "response": "r",
            "prompt_ids": [1, 2],
            "completion_ids": [3, 4],
            "logprobs": [0.1, 0.2],
        }]
        _, _, response_masks, _ = engine.assemble_steps(steps, masked_out=True)
        assert all(m == 0 for m in response_masks)

    def test_non_cumulative_steps_raise(self, engine):
        steps = [
            {
                "prompt": "p1",
                "response": "r1",
                "prompt_ids": [1, 2, 3],
                "completion_ids": [4, 5],
                "logprobs": [0.1, 0.2],
            },
            {
                "prompt": "p2",
                "response": "r2",
                "prompt_ids": [9, 9, 9, 9, 9, 9, 9],
                "completion_ids": [8, 9],
                "logprobs": [0.3, 0.4],
            },
        ]
        with pytest.raises(Exception, match="Detected invalid trajectory"):
            engine.assemble_steps(steps, masked_out=False)


# ---------------------------------------------------------------------------
# AsyncAgentExecutionEngine tests
# ---------------------------------------------------------------------------

class TestAsyncAgentExecutionEngine:
    def test_inherits_from_agent_execution_engine(self):
        assert issubclass(AsyncAgentExecutionEngine, AgentExecutionEngine)

    def test_can_instantiate(self, mock_tokenizer, mock_chat_parser):
        eng = AsyncAgentExecutionEngine(
            tokenizer=mock_tokenizer,
            chat_parser=mock_chat_parser,
        )
        assert isinstance(eng, AgentExecutionEngine)
