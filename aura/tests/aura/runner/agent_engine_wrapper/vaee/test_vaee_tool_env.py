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
from unittest.mock import MagicMock, patch

import pytest


class TestToolEnvironment:

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        mock_vaee_types = MagicMock()
        mock_vaee_types.Step = MagicMock()

        mock_loggers = MagicMock()
        mock_loggers.get_logger = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {
            "aura.runner.agent_engine_wrapper.vaee.vaee_types": mock_vaee_types,
            "aura.base.log.loggers": MagicMock(Loggers=mock_loggers),
        }):
            yield

    def test_tool_env_init(self):
        from aura.runner.agent_engine_wrapper.vaee.tool_env import ToolEnvironment

        mock_reward_fn = MagicMock()
        task = {"problem": "test problem"}
        env = ToolEnvironment(task=task, res_reward_fn=mock_reward_fn, max_steps=5)

        assert env.max_steps == 5
        assert env.step_count == 0
        assert env.task == task
        assert env.res_reward_fn == mock_reward_fn

    def test_tool_env_reset(self):
        from aura.runner.agent_engine_wrapper.vaee.tool_env import ToolEnvironment

        mock_reward_fn = MagicMock()
        task = {"problem": "test problem"}
        env = ToolEnvironment(task=task, res_reward_fn=mock_reward_fn)
        env.step_count = 5

        observation, info = env.reset()
        assert env.step_count == 0
        assert observation == task
        assert info == {}

    def test_tool_env_step_not_last(self):
        from aura.runner.agent_engine_wrapper.vaee.tool_env import ToolEnvironment

        mock_reward_output = MagicMock()
        mock_reward_output.reward = 0.5
        mock_reward_output.is_correct = False
        mock_reward_output.metadata = {}
        mock_reward_fn = MagicMock(return_value=mock_reward_output)

        task = {"problem": "test"}
        env = ToolEnvironment(task=task, res_reward_fn=mock_reward_fn)

        cur_step = MagicMock()
        cur_step.action = "some_action"
        cur_step.tool_outputs = []

        reward, done = env.step(cur_step, [], is_last=False)
        assert reward == 0.5
        assert done is False
        assert env.step_count == 1

    def test_tool_env_step_last(self):
        from aura.runner.agent_engine_wrapper.vaee.tool_env import ToolEnvironment

        mock_reward_output = MagicMock()
        mock_reward_output.reward = 1.0
        mock_reward_output.is_correct = True
        mock_reward_output.metadata = {}
        mock_reward_fn = MagicMock(return_value=mock_reward_output)

        task = {"problem": "test"}
        env = ToolEnvironment(task=task, res_reward_fn=mock_reward_fn)

        cur_step = MagicMock()
        cur_step.action = "final_answer"
        cur_step.tool_outputs = []

        reward, done = env.step(cur_step, [], is_last=True)
        assert reward == 1.0
        assert done is True

    def test_tool_env_step_task_none(self):
        from aura.runner.agent_engine_wrapper.vaee.tool_env import ToolEnvironment

        mock_reward_output = MagicMock()
        mock_reward_output.reward = 0.0
        mock_reward_output.is_correct = False
        mock_reward_output.metadata = {}
        mock_reward_fn = MagicMock(return_value=mock_reward_output)

        env = ToolEnvironment(task=None, res_reward_fn=mock_reward_fn)

        cur_step = MagicMock()
        cur_step.action = "action"
        cur_step.tool_outputs = []

        reward, done = env.step(cur_step, [], is_last=False)
        assert reward == 0.0
