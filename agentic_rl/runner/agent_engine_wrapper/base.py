#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
from typing import Any
from dataclasses import dataclass, field

import torch

from agentic_rl.base.utils.checker import TrajectoryChecker


@dataclass
class Step:
    chat_completions: list[dict[str, str]] = field(default_factory=list)

    # Response content inside <think> tag
    thought: str = ""
    # Response content inside <tool call> tag
    action: Any = None
    # Environment feedback from the previous action, provided as a 'tool' or 'user' message
    observation: Any = None
    model_response: str = ""
    info: dict = field(default_factory=dict)

    reward: float = 0.0
    done: bool = False
    mc_return: float = 0.0

    def __post_init__(self):
        TrajectoryChecker.validate_step(self.chat_completions, self.thought, self.model_response,
                                        self.info, self.reward, self.done, self.mc_return)


@dataclass
class Trajectory:
    prompt_tokens: torch.Tensor
    response_tokens: torch.Tensor
    response_masks: torch.Tensor
    idx: int = 0
    trajectory_reward: float | int = 0.0
    chat_completions: list[dict[str, str]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=lambda: {"steps": 0,
                                                             "toolcall_reward": 0.0,
                                                             "res_reward": 0.0,
                                                             "reward_time": 0.0,
                                                             "env_time": 0.0,
                                                             "llm_time": 0.0,
                                                             "total_time": 0.0})

    def __post_init__(self):
        TrajectoryChecker.validate_param({"prompt_tokens": self.prompt_tokens,
                                          "response_tokens": self.response_tokens,
                                          "response_masks": self.response_masks,
                                          "idx": self.idx,
                                          "trajectory_reward": self.trajectory_reward,
                                          "chat_completions": self.chat_completions,
                                          "metrics": self.metrics})


@dataclass
class StepTrajectory(Trajectory):
    task: Any = None
    steps: list[Step] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if (not isinstance(self.steps, list) or len(self.steps) == 0 or
                any(not isinstance(step, Step) for step in self.steps)):
            raise ValueError("steps must be a non empty list of Step instances")