# -*- coding: utf-8 -*-
#
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
# 
"""Registry of available agent configurations and lookup utility."""

from typing import Optional

from agentic_rl.runner.agent_engine_wrapper.base.environment.env_utils import compute_trajectory_reward
from agents.math_agent.environment.tool_env import ToolEnvironment
from agents.math_agent.reward.reward_fn import math_reward_fn
from agents.math_agent.tool_agent import ToolAgent

AGENTS_MAPPING = [
    {
        "name": "math",
        "env_class": ToolEnvironment,
        "env_args": {
            "tools": ["python"],
            "reward_fn": math_reward_fn,
            "tool_timeout": 120,
            "max_steps": 5,
        },
        "agent_class": ToolAgent,
        "agent_args": {
            "tools": ["python"],
            "parser_name": "qwen",
            "system_prompt": (
                "You are a math assistant that can write Python code to solve math problems. "
                "When you provide the final answer, "
                "ensure that it is wrapped in the LaTeX syntax: \\boxed{final_answer}. "
                "For example, if the answer is 42, you should return: \\boxed{42}. "
            ),
        },
        "compute_trajectory_reward_fn": compute_trajectory_reward,
    }
]


def get_agent_by_name(name: str) -> Optional[dict]:
    """
    Look up an agent configuration by its registered name.

    Args:
        name: The registered name of the agent to retrieve.

    Returns:
        The agent configuration dict, or None if no match is found.
    """
    for agent_config in AGENTS_MAPPING:
        if name == agent_config.get("name", ""):
            return agent_config

    return None
