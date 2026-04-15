# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for base/environment/base_env module: BaseEnv."""

from typing import Any

import pytest

from agentic_rl.runner.agent_engine_wrapper.base.environment.base_env import BaseEnv


# ---------------------------------------------------------------------------
# Concrete subclass for testing
# ---------------------------------------------------------------------------

class ConcreteEnv(BaseEnv):
    """Minimal concrete subclass implementing all abstract methods."""

    def __init__(self, task=None):
        self.task = task

    def reset(self):
        return {"obs": "initial"}, {}

    def step(self, action):
        return {"obs": "next"}, 1.0, False, {}

    @staticmethod
    def from_dict(info):
        return ConcreteEnv(task=info.get("task"))


@pytest.fixture
def env():
    return ConcreteEnv()


# ---------------------------------------------------------------------------
# Abstract class tests
# ---------------------------------------------------------------------------

class TestBaseEnvAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseEnv()

    def test_missing_reset_raises(self):
        class Incomplete(BaseEnv):
            def step(self, action):
                pass

            @staticmethod
            def from_dict(info):
                pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_missing_step_raises(self):
        class Incomplete(BaseEnv):
            def reset(self):
                pass

            @staticmethod
            def from_dict(info):
                pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_missing_from_dict_raises(self):
        class Incomplete(BaseEnv):
            def reset(self):
                pass

            def step(self, action):
                pass

        with pytest.raises(TypeError):
            Incomplete()


# ---------------------------------------------------------------------------
# idx property tests
# ---------------------------------------------------------------------------

class TestIdx:
    def test_idx_default_is_none(self, env):
        assert env.idx is None

    def test_idx_setter(self, env):
        env.idx = 42
        assert env.idx == 42

    def test_idx_setter_string(self, env):
        env.idx = "batch_0"
        assert env.idx == "batch_0"

    def test_idx_overwrite(self, env):
        env.idx = 1
        env.idx = 2
        assert env.idx == 2


# ---------------------------------------------------------------------------
# application_id property tests
# ---------------------------------------------------------------------------

class TestApplicationId:
    def test_application_id_default_is_none(self, env):
        assert env.application_id is None

    def test_application_id_setter(self, env):
        env.application_id = "app_123"
        assert env.application_id == "app_123"

    def test_application_id_overwrite(self, env):
        env.application_id = "first"
        env.application_id = "second"
        assert env.application_id == "second"


# ---------------------------------------------------------------------------
# Method tests
# ---------------------------------------------------------------------------

class TestConcreteEnvMethods:
    def test_reset_returns_tuple(self, env):
        obs, info = env.reset()
        assert obs == {"obs": "initial"}
        assert info == {}

    def test_step_returns_tuple(self, env):
        obs, reward, done, info = env.step("action")
        assert obs == {"obs": "next"}
        assert reward == 1.0
        assert done is False
        assert info == {}

    def test_from_dict(self):
        env = ConcreteEnv.from_dict({"task": "test_task"})
        assert env.task == "test_task"

    def test_close_does_not_raise(self, env):
        env.close()

    def test_is_multithread_safe_default(self):
        assert BaseEnv.is_multithread_safe() is True

    def test_is_multithread_safe_on_instance(self, env):
        assert env.is_multithread_safe() is True


# ---------------------------------------------------------------------------
# from_dict on BaseEnv raises
# ---------------------------------------------------------------------------

class TestBaseEnvFromDict:
    def test_base_env_from_dict_raises(self):
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            BaseEnv.from_dict({})
