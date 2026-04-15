# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for base/environment/env_utils module."""

from unittest.mock import MagicMock, patch

import pytest

from agentic_rl.runner.agent_engine_wrapper.base.agent.base_agent import Step, Trajectory
from agentic_rl.runner.agent_engine_wrapper.base.environment.env_utils import (
    compute_mc_return,
    compute_trajectory_reward,
    compute_trajectory_reward_raw,
    parallel_task_manager,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trajectory_with_steps():
    steps = [
        Step(reward=1.0, done=False, step_id=0),
        Step(reward=2.0, done=False, step_id=1),
        Step(reward=3.0, done=True, step_id=2),
    ]
    return Trajectory(steps=steps)


@pytest.fixture
def empty_trajectory():
    return Trajectory()


@pytest.fixture
def trajectory_no_done():
    steps = [
        Step(reward=0.5, done=False, step_id=0),
        Step(reward=0.8, done=False, step_id=1),
    ]
    return Trajectory(steps=steps)


@pytest.fixture
def trajectory_all_done():
    steps = [
        Step(reward=1.0, done=True, step_id=0),
        Step(reward=2.0, done=True, step_id=1),
    ]
    return Trajectory(steps=steps)


# ---------------------------------------------------------------------------
# compute_trajectory_reward_raw tests
# ---------------------------------------------------------------------------

class TestComputeTrajectoryRewardRaw:
    def test_sums_all_step_rewards(self, trajectory_with_steps):
        result = compute_trajectory_reward_raw(trajectory_with_steps)
        assert result.reward == pytest.approx(6.0)

    def test_returns_same_trajectory(self, trajectory_with_steps):
        result = compute_trajectory_reward_raw(trajectory_with_steps)
        assert result is trajectory_with_steps

    def test_empty_trajectory_returns_as_is(self):
        result = compute_trajectory_reward_raw(None)
        assert result is None

    def test_single_step(self):
        t = Trajectory(steps=[Step(reward=5.0)])
        result = compute_trajectory_reward_raw(t)
        assert result.reward == pytest.approx(5.0)

    def test_zero_rewards(self):
        steps = [Step(reward=0.0) for _ in range(3)]
        t = Trajectory(steps=steps)
        result = compute_trajectory_reward_raw(t)
        assert result.reward == pytest.approx(0.0)

    def test_negative_rewards(self):
        steps = [Step(reward=-1.0), Step(reward=-2.0)]
        t = Trajectory(steps=steps)
        result = compute_trajectory_reward_raw(t)
        assert result.reward == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# compute_trajectory_reward tests
# ---------------------------------------------------------------------------

class TestComputeTrajectoryReward:
    def test_splits_toolcall_and_res_rewards(self, trajectory_with_steps):
        result = compute_trajectory_reward(trajectory_with_steps)
        assert result.toolcall_reward == pytest.approx(1.5)
        assert result.res_reward == pytest.approx(3.0)
        assert result.reward == pytest.approx(1.5 + 3.0)

    def test_returns_same_trajectory(self, trajectory_with_steps):
        result = compute_trajectory_reward(trajectory_with_steps)
        assert result is trajectory_with_steps

    def test_empty_trajectory_returns_as_is(self):
        result = compute_trajectory_reward(None)
        assert result is None

    def test_no_done_steps(self, trajectory_no_done):
        result = compute_trajectory_reward(trajectory_no_done)
        assert result.toolcall_reward == pytest.approx(0.65)
        assert result.res_reward == -2
        assert result.reward == pytest.approx(0.65 + (-2))

    def test_all_done_steps(self, trajectory_all_done):
        result = compute_trajectory_reward(trajectory_all_done)
        assert result.toolcall_reward == 0
        assert result.res_reward == pytest.approx(2.0)

    def test_single_non_done_step(self):
        t = Trajectory(steps=[Step(reward=0.5, done=False)])
        result = compute_trajectory_reward(t)
        assert result.toolcall_reward == pytest.approx(0.5)
        assert result.res_reward == -2
        assert result.reward == pytest.approx(0.5 + (-2))

    def test_single_done_step(self):
        t = Trajectory(steps=[Step(reward=3.0, done=True)])
        result = compute_trajectory_reward(t)
        assert result.toolcall_reward == 0
        assert result.res_reward == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# compute_mc_return tests
# ---------------------------------------------------------------------------

class TestComputeMcReturn:
    def test_mc_return_single_step(self):
        t = Trajectory(steps=[Step(reward=1.0)])
        result = compute_mc_return(t)
        assert result.steps[0].mc_return == pytest.approx(1.0)

    def test_mc_return_multiple_steps_default_gamma(self):
        steps = [Step(reward=1.0), Step(reward=2.0), Step(reward=3.0)]
        t = Trajectory(steps=steps)
        result = compute_mc_return(t, gamma=0.95)
        # G_2 = 3.0
        # G_1 = 2.0 + 0.95 * 3.0 = 4.85
        # G_0 = 1.0 + 0.95 * 4.85 = 5.6075
        assert result.steps[2].mc_return == pytest.approx(3.0)
        assert result.steps[1].mc_return == pytest.approx(4.85)
        assert result.steps[0].mc_return == pytest.approx(5.6075)

    def test_mc_return_custom_gamma(self):
        steps = [Step(reward=1.0), Step(reward=1.0)]
        t = Trajectory(steps=steps)
        result = compute_mc_return(t, gamma=0.5)
        # G_1 = 1.0
        # G_0 = 1.0 + 0.5 * 1.0 = 1.5
        assert result.steps[1].mc_return == pytest.approx(1.0)
        assert result.steps[0].mc_return == pytest.approx(1.5)

    def test_mc_return_gamma_zero(self):
        steps = [Step(reward=1.0), Step(reward=2.0), Step(reward=3.0)]
        t = Trajectory(steps=steps)
        result = compute_mc_return(t, gamma=0.0)
        for i, step in enumerate(result.steps):
            assert step.mc_return == pytest.approx(step.reward)

    def test_mc_return_gamma_one(self):
        steps = [Step(reward=1.0), Step(reward=2.0), Step(reward=3.0)]
        t = Trajectory(steps=steps)
        result = compute_mc_return(t, gamma=1.0)
        assert result.steps[2].mc_return == pytest.approx(3.0)
        assert result.steps[1].mc_return == pytest.approx(5.0)
        assert result.steps[0].mc_return == pytest.approx(6.0)

    def test_mc_return_empty_trajectory(self, empty_trajectory):
        result = compute_mc_return(empty_trajectory)
        assert result.steps == []

    def test_mc_return_returns_same_trajectory(self):
        t = Trajectory(steps=[Step(reward=1.0)])
        result = compute_mc_return(t)
        assert result is t


# ---------------------------------------------------------------------------
# parallel_task_manager tests
# ---------------------------------------------------------------------------

class TestParallelTaskManager:
    def test_basic_parallel_execution(self):
        def add(a, b):
            return a + b

        items = [(1, 2), (3, 4), (5, 6)]
        with parallel_task_manager(add, items) as results:
            results_dict = dict(results)
            assert results_dict[0] == 3
            assert results_dict[1] == 7
            assert results_dict[2] == 11

    def test_preserves_all_indices(self):
        def identity(x):
            return x

        items = [(i,) for i in range(10)]
        with parallel_task_manager(identity, items) as results:
            indices = sorted([idx for idx, _ in results])
            assert indices == list(range(10))

    def test_empty_items(self):
        def noop():
            return None

        with parallel_task_manager(noop, []) as results:
            assert results == []

    def test_single_item(self):
        def double(x):
            return x * 2

        with parallel_task_manager(double, [(5,)]) as results:
            assert len(results) == 1
            assert results[0][1] == 10

    def test_custom_max_workers(self):
        def identity(x):
            return x

        items = [(i,) for i in range(5)]
        with parallel_task_manager(identity, items, max_workers=2) as results:
            assert len(results) == 5

    def test_exception_propagation(self):
        def fail(x):
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            with parallel_task_manager(fail, [(1,)]) as results:
                pass
