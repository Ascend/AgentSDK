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

import json
import numpy as np
from typing import List, Optional, Any

from aura.runner.agent_engine_wrapper.vaee.vaee_types import Trajectory, Episode, RequestRecord, Step
from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

def get_episode_data(episode_dict):
    def truncate_data(data, max_str_len=100, max_list_len=10):
        """
        Recursively traverse dict/list and truncate long strings
        """
        if isinstance(data, dict):
            return {k: truncate_data(v, max_str_len, max_list_len) for k, v in data.items()}
        elif isinstance(data, list):
            return [truncate_data(i, max_str_len, max_list_len) for i in data[:max_list_len]]
        elif isinstance(data, str):
            if len(data) > max_str_len:
                return data[:max_str_len] + "..."
            return data
        else:
            return data

    return json.dumps(truncate_data(episode_dict), indent=4, ensure_ascii=False)


def default_traj_reward_func(
    episode: "Episode",
    answer: Optional[str] = None,
    *args, **kwargs
) -> "Episode":
    """
    Compute and update the reward for each trajectory in the Episode.
    Reference logic from compute_trajectory_reward:
    - toolcall_reward: Mean of all non-done step rewards.
    - res_reward: Reward of the last done step, defaults to -2 if none.
    """
    if not episode.trajectories:
        logger.info(f"default_traj_reward_func Reward completed (empty): {get_episode_data(episode.to_dict())}, answer: {answer}")
        return episode

    for trajectory in episode.trajectories:
        if not trajectory.steps:
            continue

        # 1. Compute tool call reward (mean of all non-done steps)
        toolcall_rewards = [step.reward for step in trajectory.steps if not step.done]
        toolcall_reward = np.mean(toolcall_rewards) if toolcall_rewards else 0.0

        # 2. Compute final result reward (reward of last done step)
        res_rewards = [step.reward for step in trajectory.steps if step.done]
        if res_rewards:
            res_reward = res_rewards[-1]
        else:
            res_reward = -2.0

        # 3. Update Trajectory object attributes
        trajectory.toolcall_reward = toolcall_reward
        trajectory.res_reward = res_reward
        trajectory.reward = res_reward + toolcall_reward

    logger.info(f"default_traj_reward_func Reward completed: {get_episode_data(episode.to_dict())}, answer: {answer}")
    return episode
