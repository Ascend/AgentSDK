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

from agents.math_agent.environment.tool_env import ToolEnvironment
from agents.math_agent.reward.reward_fn import math_reward_fn
from agents.math_agent.tool_agent import ToolAgent


class _LazyImport:
    """Delay optional agent imports until the agent is selected."""

    def __init__(self, module_path: str, object_name: str):
        self.module_path = module_path
        self.object_name = object_name
        self._cached = None

    def load(self):
        if self._cached is None:
            module = __import__(self.module_path, fromlist=[self.object_name])
            self._cached = getattr(module, self.object_name)
        return self._cached


def _resolve_lazy_config(agent_config: dict) -> dict:
    resolved = dict(agent_config)
    for key in ("env_class", "agent_class"):
        value = resolved.get(key)
        if isinstance(value, _LazyImport):
            resolved[key] = value.load()

    env_args = dict(resolved.get("env_args", {}))
    reward_fn = env_args.get("reward_fn")
    if isinstance(reward_fn, _LazyImport):
        env_args["reward_fn"] = reward_fn.load()
    resolved["env_args"] = env_args
    resolved["agent_args"] = dict(resolved.get("agent_args", {}))
    return resolved

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
        "name": "webwalker",
        "env_class": _LazyImport(
            "agents.webwalker_agent.environment.webwalker_env",
            "WebWalkerEnvironment",
        ),
        "env_args": {
            "reward_fn": _LazyImport(
                "agents.webwalker_agent.reward.reward_fn",
                "webwalker_reward_fn",
            ),
        },
        "agent_class": _LazyImport(
            "agents.webwalker_agent.webwalker_agent",
            "WebWalkerAgent",
        ),
        "agent_args": {
            "parser_name": "webwalker",
        },
        "compute_trajectory_reward_fn": compute_trajectory_reward,
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
            return _resolve_lazy_config(agent_config)

    return None
