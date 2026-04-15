#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

import concurrent.futures
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from agentic_rl.runner.agent_engine_wrapper.base.agent.base_agent import Trajectory


def compute_trajectory_reward_raw(trajectory: Trajectory) -> Trajectory:
    """
    Add trajectory reward to the dict of each interaction.

    Args:
        trajectory: List of dictionaries representing each step in the trajectory.

    Returns:
        The updated trajectory with trajectory_reward added to each step.
    """
    if not trajectory:
        return trajectory

    trajectory_reward = np.sum([step.reward for step in trajectory.steps])
    trajectory.reward = trajectory_reward
    return trajectory


def compute_trajectory_reward(trajectory: Trajectory) -> Trajectory:
    """
    Add trajectory reward to the dict of each interaction.

    Args:
        trajectory: List of dictionaries representing each step in the trajectory.

    Returns:
        The updated trajectory with trajectory_reward added to each step.
    """
    if not trajectory:
        return trajectory

    toolcall_rewards = [step.reward for step in trajectory.steps if not step.done]
    toolcall_reward = np.mean(toolcall_rewards) if toolcall_rewards else 0

    res_rewards = [step.reward for step in trajectory.steps if step.done]
    if res_rewards:
        res_reward = res_rewards[-1]
    else:
        res_reward = -2

    trajectory.toolcall_reward = toolcall_reward
    trajectory.res_reward = res_reward
    trajectory.reward = toolcall_reward + res_reward
    return trajectory


def compute_mc_return(trajectory: Trajectory, gamma: float = 0.95) -> Trajectory:
    """
    In-place Monte Carlo returns for a Trajectory dataclass.

    G_t = R_{t+1} + γ * G_{t+1}

    Args:
        trajectory: Trajectory object whose .steps is a list of Step objects.
        gamma: Discount factor.

    Returns:
        The same Trajectory, with each step.mc_return filled.
    """
    mc_return_value = 0.0
    for step in reversed(trajectory.steps):
        mc_return_value = step.reward + gamma * mc_return_value
        step.mc_return = mc_return_value
    return trajectory


@contextmanager
def parallel_task_manager(func: Callable, items: list[Any], max_workers: int = 32) -> Iterator[list[tuple[int, Any]]]:
    """
    Execute a function in parallel for all items and collect results.

    Args:
        func: Function to execute
        items: List of items to process
        max_workers: Maximum number of workers

    Yields:
        List of (idx, result) tuples
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(func, *item): i for i, item in enumerate(items)}
        for future in concurrent.futures.as_completed(future_to_item):
            idx = future_to_item[future]
            result = future.result()
            results.append((idx, result))
    yield results