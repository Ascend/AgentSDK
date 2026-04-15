# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for base_engine_wrapper module: AgentTask, Trajectory, BaseEngineWrapper."""

import uuid
from asyncio import Queue
from unittest.mock import AsyncMock

import pytest

from agentic_rl.runner.agent_engine_wrapper.base_engine_wrapper import (
    AgentTask,
    BaseEngineWrapper,
    Trajectory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_task():
    return AgentTask(sample_id=0, iteration=1, agent_name="test_agent", problem="solve this")


@pytest.fixture
def full_task():
    return AgentTask(
        task_id="fixed-id",
        sample_id=5,
        iteration=3,
        agent_name="my_agent",
        problem="complex problem",
        ground_truth="42",
        prompt_id=10,
        content="some content",
        extra_args={"key": "value"},
    )


class ConcreteEngineWrapper(BaseEngineWrapper):
    """Minimal concrete subclass to test BaseEngineWrapper."""

    async def generate_trajectory(self, task, stream_queue=None, *args, **kwargs):
        return Trajectory()


@pytest.fixture
def engine_wrapper():
    return ConcreteEngineWrapper()


# ---------------------------------------------------------------------------
# AgentTask tests
# ---------------------------------------------------------------------------

class TestAgentTask:
    def test_default_task_id_is_uuid(self, default_task):
        """task_id should be a valid UUID string when not explicitly provided."""
        parsed = uuid.UUID(default_task.task_id)
        assert str(parsed) == default_task.task_id

    def test_required_fields(self, default_task):
        assert default_task.sample_id == 0
        assert default_task.iteration == 1
        assert default_task.agent_name == "test_agent"
        assert default_task.problem == "solve this"

    def test_default_optional_fields(self, default_task):
        assert default_task.ground_truth == ""
        assert default_task.prompt_id == 0
        assert default_task.content == ""
        assert default_task.extra_args is None

    def test_full_task_values(self, full_task):
        assert full_task.task_id == "fixed-id"
        assert full_task.sample_id == 5
        assert full_task.iteration == 3
        assert full_task.agent_name == "my_agent"
        assert full_task.problem == "complex problem"
        assert full_task.ground_truth == "42"
        assert full_task.prompt_id == 10
        assert full_task.content == "some content"
        assert full_task.extra_args == {"key": "value"}

    def test_each_instance_gets_unique_task_id(self):
        """Two tasks created without explicit task_id should have different IDs."""
        t1 = AgentTask(sample_id=0, iteration=0, agent_name="a", problem="p")
        t2 = AgentTask(sample_id=0, iteration=0, agent_name="a", problem="p")
        assert t1.task_id != t2.task_id

    def test_model_dump_contains_all_fields(self, full_task):
        dumped = full_task.model_dump()
        expected_keys = {
            "task_id", "sample_id", "iteration", "agent_name",
            "problem", "ground_truth", "prompt_id", "content", "extra_args",
        }
        assert set(dumped.keys()) == expected_keys

    def test_extra_args_none_by_default(self):
        task = AgentTask(sample_id=0, iteration=0, agent_name="a", problem="p")
        assert task.extra_args is None

    def test_extra_args_dict(self):
        task = AgentTask(
            sample_id=0, iteration=0, agent_name="a", problem="p",
            extra_args={"timeout": 30, "retries": 3},
        )
        assert task.extra_args["timeout"] == 30
        assert task.extra_args["retries"] == 3

    @pytest.mark.parametrize("sample_id,iteration", [(0, 0), (100, 50), (-1, -1)])
    def test_numeric_fields_accept_various_values(self, sample_id, iteration):
        task = AgentTask(sample_id=sample_id, iteration=iteration, agent_name="a", problem="p")
        assert task.sample_id == sample_id
        assert task.iteration == iteration


# ---------------------------------------------------------------------------
# Trajectory (pydantic BaseModel) tests
# ---------------------------------------------------------------------------

class TestTrajectoryBaseModel:
    def test_trajectory_is_empty_by_default(self):
        t = Trajectory()
        assert isinstance(t, Trajectory)


# ---------------------------------------------------------------------------
# BaseEngineWrapper tests
# ---------------------------------------------------------------------------

class TestBaseEngineWrapper:
    def test_cannot_instantiate_abstract_class(self):
        """BaseEngineWrapper is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseEngineWrapper()

    def test_concrete_subclass_instantiation(self, engine_wrapper):
        assert isinstance(engine_wrapper, BaseEngineWrapper)

    @pytest.mark.asyncio
    async def test_generate_trajectory_returns_trajectory(self, engine_wrapper):
        task = AgentTask(sample_id=0, iteration=0, agent_name="a", problem="p")
        result = await engine_wrapper.generate_trajectory(task)
        assert isinstance(result, Trajectory)

    @pytest.mark.asyncio
    async def test_generate_trajectory_with_stream_queue(self, engine_wrapper):
        task = AgentTask(sample_id=0, iteration=0, agent_name="a", problem="p")
        queue = Queue()
        result = await engine_wrapper.generate_trajectory(task, stream_queue=queue)
        assert isinstance(result, Trajectory)

    def test_missing_abstract_method_raises_type_error(self):
        """Omitting generate_trajectory should prevent instantiation."""

        class IncompleteWrapper(BaseEngineWrapper):
            pass

        with pytest.raises(TypeError):
            IncompleteWrapper()
