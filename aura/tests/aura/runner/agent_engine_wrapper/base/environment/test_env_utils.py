# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for base/environment/env_utils module."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aura.runner.agent_engine_wrapper.base.agent.base_agent import Step, Trajectory
from aura.runner.agent_engine_wrapper.base.environment import env_utils
from aura.runner.agent_engine_wrapper.base.environment.env_utils import (
    _compute_webwalker_chain_reward,
    _reached_source_url,
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


# ---------------------------------------------------------------------------
# WebWalker chain-mode reward (PR2)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_webwalker_reward_config(monkeypatch):
    """Inject a fake ``agents.webwalker_agent.reward.reward_config`` module so
    the lazy import inside ``_compute_webwalker_chain_reward`` resolves without
    the real webwalker package being installed."""
    fake_cfg = SimpleNamespace(
        chain_success_reward=1.0,
        chain_failure_reward=-0.5,
    )

    fake_reward_config = types.ModuleType("agents.webwalker_agent.reward.reward_config")
    fake_reward_config.get_webwalker_reward_config = MagicMock(return_value=fake_cfg)

    fake_reward = types.ModuleType("agents.webwalker_agent.reward")
    fake_webwalker_agent = types.ModuleType("agents.webwalker_agent")
    fake_agents = types.ModuleType("agents")

    monkeypatch.setitem(sys.modules, "agents", fake_agents)
    monkeypatch.setitem(sys.modules, "agents.webwalker_agent", fake_webwalker_agent)
    monkeypatch.setitem(sys.modules, "agents.webwalker_agent.reward", fake_reward)
    monkeypatch.setitem(
        sys.modules, "agents.webwalker_agent.reward.reward_config", fake_reward_config
    )
    return fake_reward_config, fake_cfg


def _make_step(source_url_hit=None):
    """Build a Step with the given source_url_hit metadata flag."""
    info = {}
    if source_url_hit is not None:
        info["metadata"] = {"source_url_hit": source_url_hit}
    return Step(reward=0.0, done=False, info=info, step_id=0)


class TestReachedSourceUrl:
    def test_returns_true_when_any_step_hit(self):
        traj = Trajectory(
            steps=[
                _make_step(source_url_hit=False),
                _make_step(source_url_hit=True),
            ]
        )
        assert _reached_source_url(traj) is True

    def test_returns_false_when_no_hit(self):
        traj = Trajectory(
            steps=[_make_step(source_url_hit=False), _make_step(source_url_hit=False)]
        )
        assert _reached_source_url(traj) is False

    def test_returns_false_when_no_metadata(self):
        traj = Trajectory(steps=[Step(reward=0.0, done=False, info={})])
        assert _reached_source_url(traj) is False

    def test_returns_false_when_info_empty(self):
        traj = Trajectory(steps=[Step(reward=0.0, done=False)])
        assert _reached_source_url(traj) is False

    def test_returns_false_for_empty_trajectory(self):
        traj = Trajectory(steps=[])
        assert _reached_source_url(traj) is False

    def test_treats_falsy_source_url_hit_as_false(self):
        traj = Trajectory(steps=[_make_step(source_url_hit=0)])
        assert _reached_source_url(traj) is False

    def test_ignores_non_dict_info(self):
        traj = Trajectory(steps=[Step(reward=0.0, done=False, info=False)])
        assert _reached_source_url(traj) is False


class TestComputeWebwalkerChainReward:
    def test_success_broadcasts_reward_to_all_steps(self, fake_webwalker_reward_config):
        _, cfg = fake_webwalker_reward_config
        steps = [_make_step(source_url_hit=False), _make_step(source_url_hit=True)]
        traj = Trajectory(steps=steps)
        result = _compute_webwalker_chain_reward(traj)
        assert result.reward == cfg.chain_success_reward
        assert result.res_reward == cfg.chain_success_reward
        assert result.toolcall_reward == 0.0
        for step in result.steps:
            assert step.reward == cfg.chain_success_reward

    def test_failure_uses_failure_reward(self, fake_webwalker_reward_config):
        _, cfg = fake_webwalker_reward_config
        traj = Trajectory(steps=[_make_step(source_url_hit=False)])
        result = _compute_webwalker_chain_reward(traj)
        assert result.reward == cfg.chain_failure_reward
        assert result.res_reward == cfg.chain_failure_reward
        assert result.toolcall_reward == 0.0
        assert result.steps[0].reward == cfg.chain_failure_reward

    def test_returns_same_trajectory(self, fake_webwalker_reward_config):
        traj = Trajectory(steps=[_make_step(source_url_hit=True)])
        result = _compute_webwalker_chain_reward(traj)
        assert result is traj

    def test_uses_registered_reward_config_fn(self, fake_webwalker_reward_config):
        fake_module, _ = fake_webwalker_reward_config
        traj = Trajectory(steps=[_make_step(source_url_hit=True)])
        _compute_webwalker_chain_reward(traj)
        fake_module.get_webwalker_reward_config.assert_called_once_with()


class TestComputeTrajectoryRewardChainBranch:
    def test_chain_mode_routes_to_webwalker_chain_reward(self, fake_webwalker_reward_config):
        _, cfg = fake_webwalker_reward_config
        steps = [_make_step(source_url_hit=True), _make_step(source_url_hit=False)]
        traj = Trajectory(steps=steps)
        traj.trajectory_generation_method = "chain"
        result = compute_trajectory_reward(traj)
        assert result.reward == cfg.chain_success_reward
        assert result.toolcall_reward == 0.0

    def test_chain_mode_failure(self, fake_webwalker_reward_config):
        _, cfg = fake_webwalker_reward_config
        traj = Trajectory(steps=[_make_step(source_url_hit=False)])
        traj.trajectory_generation_method = "chain"
        result = compute_trajectory_reward(traj)
        assert result.reward == cfg.chain_failure_reward

    def test_chain_mode_missing_webwalker_config_falls_back(self, monkeypatch):
        monkeypatch.setitem(
            compute_trajectory_reward.__globals__,
            "_compute_webwalker_chain_reward",
            MagicMock(side_effect=ImportError("missing webwalker reward config")),
        )
        steps = [Step(reward=1.0, done=False, step_id=0), Step(reward=3.0, done=True, step_id=1)]
        traj = Trajectory(steps=steps)
        traj.trajectory_generation_method = "chain"

        result = compute_trajectory_reward(traj)

        assert result.toolcall_reward == pytest.approx(1.0)
        assert result.res_reward == pytest.approx(3.0)
        assert result.reward == pytest.approx(4.0)

    @pytest.mark.parametrize("error", [AttributeError("bad attr"), KeyError("missing key")])
    def test_chain_mode_config_errors_fall_back(self, monkeypatch, error):
        monkeypatch.setitem(
            compute_trajectory_reward.__globals__,
            "_compute_webwalker_chain_reward",
            MagicMock(side_effect=error),
        )
        steps = [Step(reward=1.0, done=False, step_id=0), Step(reward=3.0, done=True, step_id=1)]
        traj = Trajectory(steps=steps)
        traj.trajectory_generation_method = "chain"

        result = compute_trajectory_reward(traj)

        assert result.toolcall_reward == pytest.approx(1.0)
        assert result.res_reward == pytest.approx(3.0)
        assert result.reward == pytest.approx(4.0)

    def test_non_chain_mode_skips_chain_branch(self, fake_webwalker_reward_config):
        steps = [Step(reward=1.0, done=False, step_id=0), Step(reward=3.0, done=True, step_id=1)]
        traj = Trajectory(steps=steps)
        traj.trajectory_generation_method = "tree"
        result = compute_trajectory_reward(traj)
        # Falls back to the default toolcall + res reward logic.
        assert result.toolcall_reward == pytest.approx(1.0)
        assert result.res_reward == pytest.approx(3.0)
        assert result.reward == pytest.approx(4.0)

    def test_no_method_attr_uses_default_branch(self, fake_webwalker_reward_config):
        steps = [Step(reward=2.0, done=True, step_id=0)]
        traj = Trajectory(steps=steps)
        result = compute_trajectory_reward(traj)
        assert result.toolcall_reward == 0
        assert result.res_reward == pytest.approx(2.0)
