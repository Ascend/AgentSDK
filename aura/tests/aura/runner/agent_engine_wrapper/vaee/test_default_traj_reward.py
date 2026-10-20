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


class TestDefaultTrajReward:

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        mock_numpy = MagicMock()
        mock_numpy.mean = lambda x, *args, **kwargs: sum(x) / len(x) if x else 0.0
        mock_numpy.exp = MagicMock(return_value=1.0)
        mock_numpy.bool_ = bool

        mock_vaee_types = MagicMock()
        mock_vaee_types.Episode = MagicMock()
        mock_vaee_types.Trajectory = MagicMock()
        mock_vaee_types.Step = MagicMock()

        mock_loggers = MagicMock()
        mock_loggers.get_logger = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {
            "numpy": mock_numpy,
            "aura.runner.agent_engine_wrapper.vaee.vaee_types": mock_vaee_types,
            "aura.base.log.loggers": MagicMock(Loggers=mock_loggers),
        }):
            yield

    def test_truncate_data_string(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_reward import get_episode_data

        mock_episode = MagicMock()
        mock_episode.to_dict.return_value = {
            "id": "test_1",
            "trajectories": [{"steps": [{"step_id": 1, "reward": 0.5}]}],
        }

        result = get_episode_data(mock_episode.to_dict())
        assert isinstance(result, str)
        assert "test_1" in result

    def test_truncate_data_long_string(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_reward import get_episode_data

        data = {"key": "a" * 200}
        result = get_episode_data(data)
        assert "..." in result

    def test_truncate_data_long_list(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_reward import get_episode_data

        data = {"key": list(range(20))}
        result = get_episode_data(data)
        assert len(result) > 0

    def test_default_traj_reward_func_empty_episode(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_reward import default_traj_reward_func

        episode = MagicMock()
        episode.trajectories = []
        episode.to_dict.return_value = {"id": "empty"}

        result = default_traj_reward_func(episode)
        assert result == episode

    def test_default_traj_reward_func_with_steps(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_reward import default_traj_reward_func

        step_done = MagicMock()
        step_done.reward = 1.0
        step_done.done = True

        step_not_done = MagicMock()
        step_not_done.reward = 0.5
        step_not_done.done = False

        traj = MagicMock()
        traj.steps = [step_not_done, step_not_done, step_done]
        traj.toolcall_reward = 0.0
        traj.res_reward = 0.0
        traj.reward = 0.0

        episode = MagicMock()
        episode.trajectories = [traj]
        episode.to_dict.return_value = {"id": "test"}

        result = default_traj_reward_func(episode)
        assert traj.toolcall_reward == pytest.approx(0.5)
        assert traj.res_reward == 1.0
        assert traj.reward == pytest.approx(1.5)

    def test_default_traj_reward_func_no_done_steps(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_reward import default_traj_reward_func

        step1 = MagicMock()
        step1.reward = 0.5
        step1.done = False

        step2 = MagicMock()
        step2.reward = 0.3
        step2.done = False

        traj = MagicMock()
        traj.steps = [step1, step2]
        traj.toolcall_reward = 0.0
        traj.res_reward = 0.0
        traj.reward = 0.0

        episode = MagicMock()
        episode.trajectories = [traj]
        episode.to_dict.return_value = {"id": "test"}

        result = default_traj_reward_func(episode)
        assert traj.toolcall_reward == pytest.approx(0.4)
        assert traj.res_reward == -2.0
        assert traj.reward == pytest.approx(-1.6)

    def test_default_traj_reward_func_trajectory_with_no_steps(self):
        from aura.runner.agent_engine_wrapper.vaee.default_traj_reward import default_traj_reward_func

        traj = MagicMock()
        traj.steps = []

        episode = MagicMock()
        episode.trajectories = [traj]
        episode.to_dict.return_value = {"id": "test"}

        result = default_traj_reward_func(episode)
        assert result == episode
