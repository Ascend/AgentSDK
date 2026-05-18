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

from aura.runner.agent_engine_wrapper.base.environment.env_utils import compute_trajectory_reward

from agents.proxy_agent.extern_agent import ProxyAgent
from agents.proxy_agent.environment.tool_env import ProxyEnvironment

from agents.math_agent.environment.tool_env import ToolEnvironment
from agents.math_agent.reward.reward_fn import math_reward_fn
from agents.math_agent.tool_agent import ToolAgent

from agents.tools_mapping import TOOLS_MAP
from agents.search_r1_agent.prompt import SEARCH_R1_PROMPT
from agents.search_r1_agent.reward.reward_fn import search_r1_res_reward_fn
from agents.search_r1_agent.reward.search_r1_reward import compute_search_r1_trajectory_reward
from agents.search_r1_agent.search_r1_agent import SearchR1Agent
from agents.search_r1_agent.environment.search_r1_env import SearchR1Environment
from agents.search_r1_agent.parser.chat_template import SearchR1ChatTemplateParser

AGENTS_MAPPING = [
    {
        "name": "math",
        "env_class": ToolEnvironment,
        "env_args": {
            "tools": ["python"],
            "reward_fn": math_reward_fn,
        },
        "agent_class": ToolAgent,
        "agent_args": {
            "tools": ["python"],
            "parser_name": "qwen",
            "system_prompt": "You are a math assistant that can write Python code to solve math problems. "
            "When you provide the final answer, "
            "ensure that it is wrapped in the LaTeX syntax: \\boxed{final_answer}. "
            "For example, if the answer is 42, you should return: \\boxed{42}. ",
        },
        "compute_trajectory_reward_fn": compute_trajectory_reward,
    },
    {
        "name": "proxy",
        "env_class": ProxyEnvironment,
        "env_args": {
            # tool_map 和 reward_fn 通过配置文件传入
        },
        "agent_class": ProxyAgent,
        "agent_args": {
            "parser_name": "qwen",
        },
    },
    {
        "name": "search_r1",
        "env_class": SearchR1Environment,
        "chat_parser": SearchR1ChatTemplateParser,
        "env_args": {
            "tool_map": TOOLS_MAP["search_r1"],
            "reward_fn": search_r1_res_reward_fn,
        },
        "agent_class": SearchR1Agent,
        "agent_args": {
            "tool_map": TOOLS_MAP["search_r1"],
            "parser_name": "qwen",
            "system_prompt": SEARCH_R1_PROMPT,
        },
        "compute_trajectory_reward_fn": compute_search_r1_trajectory_reward,
    },
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
