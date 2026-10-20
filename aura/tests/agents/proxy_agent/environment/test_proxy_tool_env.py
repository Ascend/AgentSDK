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
from abc import ABC, abstractmethod
from unittest.mock import MagicMock, patch

import pytest


class MockBaseEnv(ABC):
    """Mock base class for BaseEnv that ProxyEnvironment can inherit from."""

    def __init__(self, *args, **kwargs):
        pass

    @property
    def idx(self):
        return getattr(self, "_idx", None)

    @idx.setter
    def idx(self, value):
        self._idx = value

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action, done, tool_outputs, raw_reward=None):
        pass


class TestProxyEnvironment:

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        # Mock base_env module with proper class
        mock_base_env = MagicMock()
        mock_base_env.BaseEnv = MockBaseEnv

        with patch.dict(sys.modules, {
            "aura.runner.agent_engine_wrapper.base.environment.base_env": mock_base_env,
        }):
            yield

    def test_import_external_reward_fn_success(self):
        from agents.proxy_agent.environment.tool_env import import_external_reward_fn

        mock_module = MagicMock()
        mock_reward_fn = MagicMock()
        mock_module.my_fn = mock_reward_fn

        with patch("importlib.import_module", return_value=mock_module):
            result = import_external_reward_fn("some.module.my_fn")
            assert result == mock_reward_fn

    def test_import_external_reward_fn_none(self):
        from agents.proxy_agent.environment.tool_env import import_external_reward_fn

        assert import_external_reward_fn(None) is None
        assert import_external_reward_fn("null") is None
        assert import_external_reward_fn("NULL") is None

    def test_import_external_reward_fn_invalid_format(self):
        from agents.proxy_agent.environment.tool_env import import_external_reward_fn

        with pytest.raises(ValueError, match="format is incorrect"):
            import_external_reward_fn("no_dot_separator")

    def test_import_external_reward_fn_import_error(self):
        from agents.proxy_agent.environment.tool_env import import_external_reward_fn

        with patch("importlib.import_module", side_effect=ImportError("Cannot find")):
            with pytest.raises(ImportError, match="Cannot find"):
                import_external_reward_fn("missing.module.fn")

    def test_import_external_reward_fn_attribute_error(self):
        from agents.proxy_agent.environment.tool_env import import_external_reward_fn

        mock_module = MagicMock()
        del mock_module.my_fn

        with patch("importlib.import_module", return_value=mock_module):
            with pytest.raises(AttributeError, match="does not contain"):
                import_external_reward_fn("some.module.my_fn")

    def test_proxy_environment_init(self):
        from agents.proxy_agent.environment.tool_env import ProxyEnvironment

        mock_module = MagicMock()
        mock_reward_fn = MagicMock()
        mock_module.tool_fn = mock_reward_fn
        mock_module.res_fn = mock_reward_fn

        with patch("importlib.import_module", return_value=mock_module):
            env = ProxyEnvironment(
                task={"problem": "test"},
                tools=["tool1"],
                tool_reward_fn_path="mod.tool_fn",
                res_reward_fn_path="mod.res_fn",
                max_steps=5,
            )
            assert env.max_steps == 5
            assert env.step_count == 0
            assert env.task == {"problem": "test"}

    def test_proxy_environment_init_with_tool_map(self):
        from agents.proxy_agent.environment.tool_env import ProxyEnvironment

        mock_module = MagicMock()
        mock_reward_fn = MagicMock()
        mock_module.res_fn = mock_reward_fn

        with patch("importlib.import_module", return_value=mock_module):
            env = ProxyEnvironment(
                task={"problem": "test"},
                tool_map={"tool1": MagicMock()},
                tool_reward_fn_path=None,
                res_reward_fn_path="mod.res_fn",
                max_steps=5,
            )
            assert env.max_steps == 5

    def test_proxy_environment_init_tools_and_tool_map_error(self):
        from agents.proxy_agent.environment.tool_env import ProxyEnvironment

        mock_module = MagicMock()
        mock_reward_fn = MagicMock()
        mock_module.res_fn = mock_reward_fn

        with patch("importlib.import_module", return_value=mock_module):
            with pytest.raises(ValueError, match="Cannot specify both"):
                ProxyEnvironment(
                    task={"problem": "test"},
                    tools=["tool1"],
                    tool_map={"tool1": MagicMock()},
                    tool_reward_fn_path=None,
                    res_reward_fn_path="mod.res_fn",
                )

    def test_proxy_environment_reset(self):
        from agents.proxy_agent.environment.tool_env import ProxyEnvironment

        mock_module = MagicMock()
        mock_reward_fn = MagicMock()
        mock_module.res_fn = mock_reward_fn

        with patch("importlib.import_module", return_value=mock_module):
            env = ProxyEnvironment(
                task={"problem": "test"},
                tool_reward_fn_path=None,
                res_reward_fn_path="mod.res_fn",
            )
            env.step_count = 5
            observation, info = env.reset()
            assert env.step_count == 0
            assert observation == {"problem": "test"}
            assert info == {}

    def test_proxy_environment_step_not_done(self):
        from agents.proxy_agent.environment.tool_env import ProxyEnvironment

        mock_module = MagicMock()
        mock_reward_output = MagicMock()
        mock_reward_output.reward = 0.5
        mock_reward_output.metadata = {"key": "val"}
        mock_tool_reward_fn = MagicMock(return_value=mock_reward_output)
        mock_module.tool_fn = mock_tool_reward_fn

        with patch("importlib.import_module", return_value=mock_module):
            env = ProxyEnvironment(
                task={"problem": "test"},
                tool_reward_fn_path="mod.tool_fn",
                res_reward_fn_path=None,
            )
            next_obs, reward, done, info = env.step(
                action=[{"id": "1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}],
                done=False,
                tool_outputs=[{"tool_call_id": "1", "content": "result"}],
            )
            assert reward == 0.5
            assert done is False
            assert "tool_outputs" in next_obs

    def test_proxy_environment_step_done_with_string(self):
        from agents.proxy_agent.environment.tool_env import ProxyEnvironment

        mock_module = MagicMock()
        mock_reward_output = MagicMock()
        mock_reward_output.reward = 1.0
        mock_reward_output.metadata = {}
        mock_res_reward_fn = MagicMock(return_value=mock_reward_output)
        mock_module.res_fn = mock_res_reward_fn

        with patch("importlib.import_module", return_value=mock_module):
            env = ProxyEnvironment(
                task={"problem": "test"},
                tool_reward_fn_path=None,
                res_reward_fn_path="mod.res_fn",
            )
            next_obs, reward, done, info = env.step(
                action="final answer",
                done=True,
                tool_outputs=[],
            )
            assert reward == 1.0
            assert done is True
            assert next_obs == {}

    def test_proxy_environment_step_done_with_finish_tool(self):
        from agents.proxy_agent.environment.tool_env import ProxyEnvironment

        mock_module = MagicMock()
        mock_reward_output = MagicMock()
        mock_reward_output.reward = 1.0
        mock_reward_output.metadata = {}
        mock_res_reward_fn = MagicMock(return_value=mock_reward_output)
        mock_module.res_fn = mock_res_reward_fn

        with patch("importlib.import_module", return_value=mock_module):
            env = ProxyEnvironment(
                task={"problem": "test"},
                tool_reward_fn_path=None,
                res_reward_fn_path="mod.res_fn",
            )
            next_obs, reward, done, info = env.step(
                action=[{
                    "id": "1",
                    "type": "function",
                    "function": {"name": "finish", "arguments": {"response": "the answer"}},
                }],
                done=True,
                tool_outputs=[],
            )
            assert reward == 1.0
            assert done is True

    def test_proxy_environment_step_raw_reward(self):
        from agents.proxy_agent.environment.tool_env import ProxyEnvironment

        mock_module = MagicMock()
        mock_reward_fn = MagicMock()
        mock_module.res_fn = mock_reward_fn

        with patch("importlib.import_module", return_value=mock_module):
            env = ProxyEnvironment(
                task={"problem": "test"},
                tool_reward_fn_path=None,
                res_reward_fn_path="mod.res_fn",
            )
            next_obs, reward, done, info = env.step(
                action="answer",
                done=True,
                tool_outputs=[],
                raw_reward=0.8,
            )
            assert reward == 0.8

    def test_proxy_environment_from_dict(self):
        from agents.proxy_agent.environment.tool_env import ProxyEnvironment

        mock_module = MagicMock()
        mock_reward_fn = MagicMock()
        mock_module.res_fn = mock_reward_fn

        with patch("importlib.import_module", return_value=mock_module):
            env = ProxyEnvironment.from_dict({
                "task": {"problem": "test"},
                "tools": ["tool1"],
                "max_steps": 5,
                "custom_reward_function": {
                    "tool_reward_fn_path": None,
                    "res_reward_fn_path": "mod.res_fn",
                },
            })
            assert env.max_steps == 5
            assert env.task == {"problem": "test"}

    def test_proxy_environment_from_dict_no_custom_reward(self):
        from agents.proxy_agent.environment.tool_env import ProxyEnvironment

        mock_module = MagicMock()
        mock_reward_fn = MagicMock()

        with patch("importlib.import_module", return_value=mock_module):
            env = ProxyEnvironment.from_dict({
                "task": {"problem": "test"},
                "tools": ["tool1"],
                "custom_reward_function": {},
            })
            assert env.task == {"problem": "test"}

    def test_proxy_environment_from_dict_missing_res_reward(self):
        from agents.proxy_agent.environment.tool_env import ProxyEnvironment

        mock_module = MagicMock()
        mock_reward_fn = MagicMock()

        with patch("importlib.import_module", return_value=mock_module):
            with pytest.raises(ValueError, match="res_reward_fn_path"):
                ProxyEnvironment.from_dict({
                    "task": {"problem": "test"},
                    "tools": ["tool1"],
                    "custom_reward_function": {
                        "tool_reward_fn_path": None,
                    },
                })
