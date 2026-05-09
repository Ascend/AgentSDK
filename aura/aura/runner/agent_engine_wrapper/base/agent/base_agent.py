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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    chat_completions: list[dict[str, str]] = field(default_factory=list)

    thought: str = ""
    action: Any = None
    observation: Any = None
    model_response: str = ""
    info: dict = field(default_factory=dict)  # Store any additional info.

    # field below are filled by the engine
    reward: float = 0.0
    done: bool = False
    mc_return: float = 0.0

    step_id: int = 0

    def to_dict(self):
        return {
            "chat_completions": self.chat_completions,
            "reward": self.reward,
            "mc_return": float(self.mc_return),
            "done": self.done,
            "step_id": self.step_id,
        }


@dataclass
class Action:
    action: Any = None


@dataclass
class Trajectory:
    task: Any = None
    steps: list[Step] = field(default_factory=list)
    reward: float = 0.0
    toolcall_reward: float = 0.0
    res_reward: float = 0.0
    prompt_id: int = 0
    data_id: str = None,
    training_id: str = None
    epoch_id = 0
    iteration_id = 0
    sample_id = 0
    trajectory_id = 0
    application_id = ""
    termination_reason: str = "unknown"

    def to_dict(self):
        return {
            "task": self.task,
            "steps": [step.to_dict() for step in self.steps],
            "reward": float(self.reward),
            "prompt_id": self.prompt_id,
            "data_id": self.data_id,
            "training_id": self.training_id,
            "epoch_id": self.epoch_id,
            "iteration_id": self.iteration_id,
            "sample_id": self.sample_id,
            "trajectory_id": self.trajectory_id,
            "application_id": self.application_id,
            "termination_reason": self.termination_reason,
        }

    def to_info_dict(self):
        return {
            "task": self.task,
            "data_id": self.data_id,
            "training_id": self.training_id,
            "epoch_id": self.epoch_id,
            "iteration_id": self.iteration_id,
            "sample_id": self.sample_id,
            "trajectory_id": self.trajectory_id,
            "application_id": self.application_id,
            "total_steps": len(self.steps),
            "termination_reason": self.termination_reason,
        }


class BaseAgent(ABC):
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Converts agent's internal state into a list of OAI chat completions."""
        return []

    @property
    def trajectory(self) -> Trajectory:
        """Converts agent's internal state into a Trajectory object."""
        return Trajectory()

    @abstractmethod
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.

        Args:
            observation (Any): The observation after stepping through environment.
            reward (float): The reward received after taking the action.
            done (bool): Whether the episode has ended due to termination.
            info (dict): Additional metadata from the environment.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's internal state after the model generates a response.

        Args:
            response (str): The response from the model.

        Returns:
            None
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def reset(self):
        """
        Resets the agent's internal state, typically called at the beginning of a new episode.

        This function should clear any stored history or state information necessary
        for a fresh interaction.

        Returns:
            None
        """
        return

    def get_current_state(self) -> Step | None:
        """
        Returns the agent's current state as a dictionary.

        This method provides access to the agent's internal state at the current step,
        which can be useful for debugging, logging, or state management.

        Returns:
            Step: The agent's current state.
        """
        if not self.trajectory.steps:
            return None
        return self.trajectory.steps[-1]
