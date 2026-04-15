# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for base_agent module: Step, Action, Trajectory, and BaseAgent."""

from typing import Any

import pytest

from agentic_rl.runner.agent_engine_wrapper.base.agent.base_agent import (
    Action,
    BaseAgent,
    Step,
    Trajectory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_step():
    return Step()


@pytest.fixture
def custom_step():
    return Step(
        chat_completions=[{"role": "user", "content": "hello"}],
        thought="thinking",
        action="do_something",
        observation="result",
        model_response="response text",
        info={"key": "value"},
        reward=1.5,
        done=True,
        mc_return=2.0,
        step_id=3,
    )


@pytest.fixture
def default_trajectory():
    return Trajectory()


@pytest.fixture
def trajectory_with_steps():
    steps = [
        Step(
            chat_completions=[{"role": "user", "content": "q1"}],
            reward=0.5,
            done=False,
            step_id=0,
        ),
        Step(
            chat_completions=[{"role": "user", "content": "q2"}],
            reward=1.0,
            done=True,
            step_id=1,
        ),
    ]
    return Trajectory(
        task="test_task",
        steps=steps,
        reward=1.5,
        toolcall_reward=0.3,
        res_reward=0.7,
        prompt_id=42,
        termination_reason="completed",
    )


class ConcreteAgent(BaseAgent):
    """Minimal concrete subclass for testing BaseAgent."""

    def __init__(self):
        self._trajectory = Trajectory()

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def update_from_env(self, observation, reward, done, info, **kwargs):
        pass

    def update_from_model(self, response, **kwargs) -> Action:
        return Action(action=response)

    def reset(self):
        self._trajectory = Trajectory()


@pytest.fixture
def concrete_agent():
    return ConcreteAgent()


# ---------------------------------------------------------------------------
# Step dataclass tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_default_values(self, default_step):
        """Default Step should have empty/zero values for all fields."""
        assert default_step.chat_completions == []
        assert default_step.thought == ""
        assert default_step.action is None
        assert default_step.observation is None
        assert default_step.model_response == ""
        assert default_step.info == {}
        assert default_step.reward == 0.0
        assert default_step.done is False
        assert default_step.mc_return == 0.0
        assert default_step.step_id == 0

    def test_custom_values(self, custom_step):
        """Step should store all provided custom values."""
        assert custom_step.chat_completions == [{"role": "user", "content": "hello"}]
        assert custom_step.thought == "thinking"
        assert custom_step.action == "do_something"
        assert custom_step.observation == "result"
        assert custom_step.model_response == "response text"
        assert custom_step.info == {"key": "value"}
        assert custom_step.reward == 1.5
        assert custom_step.done is True
        assert custom_step.mc_return == 2.0
        assert custom_step.step_id == 3

    def test_to_dict_default(self, default_step):
        """to_dict on default Step returns correct keys with default values."""
        result = default_step.to_dict()
        assert result == {
            "chat_completions": [],
            "reward": 0.0,
            "mc_return": 0.0,
            "done": False,
            "step_id": 0,
        }

    def test_to_dict_custom(self, custom_step):
        """to_dict on custom Step returns only the serialized subset of fields."""
        result = custom_step.to_dict()
        assert result == {
            "chat_completions": [{"role": "user", "content": "hello"}],
            "reward": 1.5,
            "mc_return": 2.0,
            "done": True,
            "step_id": 3,
        }

    def test_to_dict_excludes_non_serialized_fields(self, custom_step):
        """to_dict should not include thought, action, observation, model_response, or info."""
        result = custom_step.to_dict()
        for key in ("thought", "action", "observation", "model_response", "info"):
            assert key not in result

    def test_to_dict_mc_return_is_float(self):
        """mc_return in to_dict should always be a Python float."""
        step = Step(mc_return=3)
        result = step.to_dict()
        assert isinstance(result["mc_return"], float)
        assert result["mc_return"] == 3.0

    def test_default_factory_isolation(self):
        """Each Step instance should have independent list/dict defaults."""
        step_a = Step()
        step_b = Step()
        step_a.chat_completions.append({"role": "system", "content": "x"})
        step_a.info["k"] = "v"
        assert step_b.chat_completions == []
        assert step_b.info == {}

    @pytest.mark.parametrize(
        "reward, mc_return, done, step_id",
        [
            (0.0, 0.0, False, 0),
            (-1.0, -0.5, True, 99),
            (100.0, 50.5, False, 1),
        ],
    )
    def test_to_dict_parametrized(self, reward, mc_return, done, step_id):
        """to_dict should correctly serialize various reward/done/step_id combos."""
        step = Step(reward=reward, mc_return=mc_return, done=done, step_id=step_id)
        result = step.to_dict()
        assert result["reward"] == reward
        assert result["mc_return"] == float(mc_return)
        assert result["done"] == done
        assert result["step_id"] == step_id


# ---------------------------------------------------------------------------
# Action dataclass tests
# ---------------------------------------------------------------------------

class TestAction:
    def test_default_action_is_none(self):
        """Default Action should have action=None."""
        action = Action()
        assert action.action is None

    def test_custom_action_string(self):
        action = Action(action="run")
        assert action.action == "run"

    def test_custom_action_dict(self):
        payload = {"tool": "search", "query": "test"}
        action = Action(action=payload)
        assert action.action == payload

    def test_custom_action_complex_object(self):
        action = Action(action=[1, 2, {"nested": True}])
        assert action.action == [1, 2, {"nested": True}]


# ---------------------------------------------------------------------------
# Trajectory dataclass tests
# ---------------------------------------------------------------------------

class TestTrajectory:
    def test_default_values(self, default_trajectory):
        """Default Trajectory should have sensible zero/empty defaults."""
        assert default_trajectory.task is None
        assert default_trajectory.steps == []
        assert default_trajectory.reward == 0.0
        assert default_trajectory.toolcall_reward == 0.0
        assert default_trajectory.res_reward == 0.0
        assert default_trajectory.prompt_id == 0
        assert default_trajectory.termination_reason == "unknown"

    def test_class_level_attributes_defaults(self, default_trajectory):
        """epoch_id, iteration_id, sample_id, trajectory_id, application_id
        are class-level attributes (no type annotation) and should have defaults."""
        assert default_trajectory.epoch_id == 0
        assert default_trajectory.iteration_id == 0
        assert default_trajectory.sample_id == 0
        assert default_trajectory.trajectory_id == 0
        assert default_trajectory.application_id == ""

    def test_steps_default_factory_isolation(self):
        """Each Trajectory should have its own independent steps list."""
        t1 = Trajectory()
        t2 = Trajectory()
        t1.steps.append(Step())
        assert len(t2.steps) == 0

    def test_to_dict_empty(self, default_trajectory):
        """to_dict on empty Trajectory produces correct structure."""
        result = default_trajectory.to_dict()
        assert result["task"] is None
        assert result["steps"] == []
        assert result["reward"] == 0.0
        assert result["prompt_id"] == 0
        assert result["epoch_id"] == 0
        assert result["iteration_id"] == 0
        assert result["sample_id"] == 0
        assert result["trajectory_id"] == 0
        assert result["application_id"] == ""
        assert result["termination_reason"] == "unknown"

    def test_to_dict_with_steps(self, trajectory_with_steps):
        """to_dict should serialize each step via Step.to_dict()."""
        result = trajectory_with_steps.to_dict()
        assert result["task"] == "test_task"
        assert len(result["steps"]) == 2
        assert result["steps"][0]["step_id"] == 0
        assert result["steps"][1]["step_id"] == 1
        assert result["reward"] == 1.5
        assert result["termination_reason"] == "completed"

    def test_to_dict_reward_is_float(self):
        """reward in to_dict should always be a Python float."""
        t = Trajectory(reward=5)
        result = t.to_dict()
        assert isinstance(result["reward"], float)

    def test_to_dict_keys(self, default_trajectory):
        """to_dict should return exactly the expected set of keys."""
        expected_keys = {
            "task", "steps", "reward", "prompt_id", "data_id", "training_id",
            "epoch_id", "iteration_id", "sample_id", "trajectory_id",
            "application_id", "termination_reason",
        }
        assert set(default_trajectory.to_dict().keys()) == expected_keys

    def test_to_info_dict_empty(self, default_trajectory):
        """to_info_dict on empty Trajectory produces correct structure."""
        result = default_trajectory.to_info_dict()
        assert result["task"] is None
        assert result["total_steps"] == 0
        assert result["termination_reason"] == "unknown"

    def test_to_info_dict_with_steps(self, trajectory_with_steps):
        """to_info_dict should report total_steps count."""
        result = trajectory_with_steps.to_info_dict()
        assert result["task"] == "test_task"
        assert result["total_steps"] == 2
        assert result["termination_reason"] == "completed"

    def test_to_info_dict_keys(self, default_trajectory):
        """to_info_dict should return exactly the expected set of keys."""
        expected_keys = {
            "task", "data_id", "training_id", "epoch_id", "iteration_id",
            "sample_id", "trajectory_id", "application_id", "total_steps",
            "termination_reason",
        }
        assert set(default_trajectory.to_info_dict().keys()) == expected_keys

    def test_to_info_dict_does_not_include_steps_list(self, trajectory_with_steps):
        """to_info_dict should not contain the raw steps list."""
        result = trajectory_with_steps.to_info_dict()
        assert "steps" not in result

    def test_instance_attribute_mutation(self):
        """Mutating class-level attributes on an instance should not affect other instances."""
        t1 = Trajectory()
        t1.epoch_id = 10
        t1.application_id = "app_1"
        t2 = Trajectory()
        assert t2.epoch_id == 0
        assert t2.application_id == ""

    @pytest.mark.parametrize(
        "termination_reason",
        ["completed", "timeout", "error", "max_steps", "unknown"],
    )
    def test_termination_reason_values(self, termination_reason):
        """Trajectory should accept various termination_reason strings."""
        t = Trajectory(termination_reason=termination_reason)
        assert t.termination_reason == termination_reason
        assert t.to_dict()["termination_reason"] == termination_reason
        assert t.to_info_dict()["termination_reason"] == termination_reason


# ---------------------------------------------------------------------------
# BaseAgent abstract class tests
# ---------------------------------------------------------------------------

class TestBaseAgent:
    def test_cannot_instantiate_directly(self):
        """BaseAgent is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseAgent()

    def test_concrete_subclass_instantiation(self, concrete_agent):
        """A concrete subclass implementing all abstract methods can be created."""
        assert isinstance(concrete_agent, BaseAgent)

    def test_default_chat_completions_returns_empty_list(self):
        """BaseAgent.chat_completions default implementation returns []."""
        agent = ConcreteAgent()
        result = BaseAgent.chat_completions.fget(agent)
        assert result == []

    def test_default_trajectory_returns_empty_trajectory(self):
        """BaseAgent.trajectory default returns an empty Trajectory."""
        result = BaseAgent.trajectory.fget(BaseAgent)
        assert isinstance(result, Trajectory)

    def test_concrete_agent_chat_completions(self, concrete_agent):
        """ConcreteAgent inherits default chat_completions (empty list)."""
        assert concrete_agent.chat_completions == []

    def test_concrete_agent_trajectory(self, concrete_agent):
        """ConcreteAgent returns its own trajectory."""
        traj = concrete_agent.trajectory
        assert isinstance(traj, Trajectory)
        assert traj.steps == []

    def test_get_current_state_no_steps(self, concrete_agent):
        """get_current_state returns None when trajectory has no steps."""
        assert concrete_agent.get_current_state() is None

    def test_get_current_state_with_steps(self, concrete_agent):
        """get_current_state returns the last step in trajectory."""
        step1 = Step(step_id=0, thought="first")
        step2 = Step(step_id=1, thought="second")
        concrete_agent._trajectory.steps = [step1, step2]
        result = concrete_agent.get_current_state()
        assert result is step2
        assert result.step_id == 1
        assert result.thought == "second"

    def test_get_current_state_single_step(self, concrete_agent):
        """get_current_state returns that sole step when only one exists."""
        step = Step(step_id=0, thought="only")
        concrete_agent._trajectory.steps = [step]
        assert concrete_agent.get_current_state() is step

    def test_update_from_model_returns_action(self, concrete_agent):
        """ConcreteAgent.update_from_model returns an Action."""
        result = concrete_agent.update_from_model("hello")
        assert isinstance(result, Action)
        assert result.action == "hello"

    def test_reset_clears_trajectory(self, concrete_agent):
        """reset should replace trajectory with a fresh empty one."""
        concrete_agent._trajectory.steps.append(Step(step_id=0))
        concrete_agent._trajectory.reward = 5.0
        concrete_agent.reset()
        assert concrete_agent.trajectory.steps == []
        assert concrete_agent.trajectory.reward == 0.0

    def test_abstract_update_from_env_raises(self):
        """Directly calling BaseAgent.update_from_env raises NotImplementedError."""
        agent = ConcreteAgent()
        with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
            BaseAgent.update_from_env(agent, "obs", 0.0, False, {})

    def test_abstract_update_from_model_raises(self):
        """Directly calling BaseAgent.update_from_model raises NotImplementedError."""
        agent = ConcreteAgent()
        with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
            BaseAgent.update_from_model(agent, "response")

    def test_missing_abstract_method_raises_type_error(self):
        """Omitting any abstract method should prevent instantiation."""
        class IncompleteAgent(BaseAgent):
            def update_from_env(self, observation, reward, done, info, **kwargs):
                pass
            def reset(self):
                pass

        with pytest.raises(TypeError):
            IncompleteAgent()
