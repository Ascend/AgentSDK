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

from enum import Enum
from typing import Any

import ray

from agentic_rl.runner.agent_engine_wrapper.base.agent.base_agent import Trajectory


class TerminationReason(Enum):
    MAX_PROMPT_LENGTH_EXCEEDED = "max_prompt_length_exceeded"
    MAX_RESPONSE_LENGTH_EXCEEDED = "max_response_length_exceeded"
    ENV_DONE = "env_done"
    MAX_TURNS_EXCEEDED = "max_turns_exceeded"
    TIMEOUT = "timeout"


@ray.remote
class Episode:
    def __init__(self, episode_id: str = None):
        self.episode_id = episode_id
        self.task: Any = None
        self.termination_reason = None
        # self.is_correct: bool = False
        self.trajectories: list[tuple[str, Trajectory]] = []

    def to_dict(self):
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "termination_reason": self.termination_reason,
            # "is_correct": bool(self.is_correct),
            "trajectories": [(agent_name, trajectory.to_dict()) for agent_name, trajectory in self.trajectories],
        }

    def set_task(self, task: Any):
        self.task = task

    def set_termination_reason(self, termination_reason):
        self.termination_reason = termination_reason

    def add_trajectory(self, agent_name: str, trajectory: Trajectory):
        self.trajectories.append((agent_name, trajectory))

    def get_trajectory_by_agent_name(self, agent_name: str):
        return [trajectory for name, trajectory in self.trajectories if name == agent_name]

    def remove_trajectory_by_agent_name(self, agent_name: str):
        self.trajectories = [trajectory for name, trajectory in self.trajectories if name != agent_name]
