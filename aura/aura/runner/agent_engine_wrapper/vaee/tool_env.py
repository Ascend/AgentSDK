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

from aura.runner.agent_engine_wrapper.vaee.vaee_types import Step
from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

class ToolEnvironment():
    """
    A simple environment for tool-based agents that provides questions and evaluates responses.
    """

    def __init__(self, task, res_reward_fn, max_steps=10):
        """
        Initialize the ToolEnvironment.

        Args:
            task: Task information for the environment.
            reward_fn: Reward function to use for evaluation.
            max_steps: Maximum number of steps allowed in the environment.
        """

        self.step_count = 0
        self.max_steps = max_steps

        self.task = task
        self.res_reward_fn = res_reward_fn

    def reset(self):
        """Reset the environment and return initial observations."""
        self.step_count = 0
        return self.task, {}

    def step(self, cur_step: Step, steps: list[Step], is_last: bool = False):
        """
        Take a step in the environment based on the action.

        """
        action = {
                    "assistant_message": cur_step.action,
                    "tool_outputs": cur_step.tool_outputs,
                    "step_idx": self.step_count,
                    "is_last_step": is_last,
                    "max_steps": self.max_steps
                 }
        task_info = self.task if self.task is not None else {}
        logger.info(f"[tool_env] >>> CALLING res_reward_fn: step_idx={self.step_count}, is_last={is_last}, "
                     f"action_keys={list(action.keys())}, action.has_finish_tool={'tool_calls' in (cur_step.action or {}) if isinstance(cur_step.action, dict) else False}")
        reward_output = self.res_reward_fn(action=action, task_info=task_info)
        logger.info(f"[tool_env] <<< res_reward_fn RETURNED: step_idx={self.step_count}, "
                     f"reward={reward_output.reward}, is_correct={reward_output.is_correct}, "
                     f"metadata={reward_output.metadata}")
        logger.info(f"[tool_env] step completed: step_idx={action['step_idx']}, is_last_step={action['is_last_step']}, "
                     f"max_steps={action['max_steps']}, done_before={action['step_idx'] >= (action['max_steps'] - 1) or is_last}, "
                     f"reward_returned={reward_output.reward}")
        done = is_last
        self.step_count += 1
        return reward_output.reward, done
