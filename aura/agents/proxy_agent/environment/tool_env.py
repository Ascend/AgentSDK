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
from typing import Any
import importlib
import logging

from rllm.tools.tool_base import Tool
from aura.runner.agent_engine_wrapper.base.environment.base_env import BaseEnv

logger = logging.getLogger(__name__)

def import_external_reward_fn(fn_path: str):
    """
    Dynamic function importer inspired by the Verl style
    Args:
        fn_path: a string in the format 'agents.dtn.reward.my_tool_fn'
    Returns:
        Callable Python function object
    """
    if fn_path is None or str(fn_path).strip().lower() == "null":
        return None
    try:
        module_path, fn_name = fn_path.rsplit('.', 1)

        module = importlib.import_module(module_path)

        reward_fn = getattr(module, fn_name)

        logger.info(f"Successfully loaded custom reward function: {fn_name} from {module_path}")
        return reward_fn

    except ValueError:
        raise ValueError(f"[Import Error] The path format is incorrect, it must follow the 'module.func' format: {fn_path}")
    except ImportError as e:
        raise ImportError(f"[Import Error] Cannot find module {module_path}, please check PYTHONPATH. Error details: {e}")
    except AttributeError as e:
        raise AttributeError(f"[Import Error] Module {module_path} does not contain function {fn_name}. Error details: {e}")

class ProxyEnvironment(BaseEnv):
    """
    A simple environment for tool-based agents that provides questions and evaluates responses.
    """

    def __init__(self, task: dict | None = None, tools: list[str] | None = None, tool_map: dict[str, type[Tool]] | None = None,  tool_reward_fn_path :str| None = None, res_reward_fn_path: str | None = None, max_steps=10):
        """
        Initialize the ToolEnvironment.

        Args:
            task: Task information for the environment.
            tools: List of tool names to look up in the registry (legacy behavior).
            tool_map: Dictionary mapping tool names to Tool classes (new behavior).
            reward_fn: Reward function to use for evaluation.
            max_steps: Maximum number of steps allowed in the environment.
        """
        if tool_map is not None and tools is not None:
            raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters")

        self.step_count = 0
        self.max_steps = max_steps

        self.task = task
        self.tool_reward_fn = import_external_reward_fn(tool_reward_fn_path)
        self.res_reward_fn = import_external_reward_fn(res_reward_fn_path)

    def reset(self):
        """Reset the environment and return initial observations."""
        self.step_count = 0

        return self.task, {}

    def step(self, action: list[dict] | str | dict, done: bool, tool_outputs: str|list[dict], raw_reward = None):
        """
        Take a step in the environment based on the action.

        Args:
            actions: List containing a single action string from the agent

        Returns:
            next_observations, rewards, terminateds, infos
        """
        # Check if we should terminate

        if not done:
            tool_outputs_dict = {}
            if isinstance(tool_outputs, str):
                tool_outputs_dict["mock_id"] = tool_outputs
            else:
                for call in tool_outputs:
                    tool_call_id = call["tool_call_id"]
                    tool_outputs_dict[tool_call_id] = call["content"]
            next_obs = {"tool_outputs": tool_outputs_dict}
            tool_actions = {"tool_calls": action, "tool_outputs": tool_outputs_dict}
            task_info = self.task if self.task is not None else {}
            if raw_reward is not None:
                return next_obs, raw_reward, done, {"response": action, "metadata": {}}
            if self.tool_reward_fn is not None:
                reward_output = self.tool_reward_fn(action=tool_actions, task_info=task_info)
                reward_val = reward_output.reward
                metadata = reward_output.metadata
            else:
                reward_val = None
                metadata = {}
                logger.warning("No reward function provided, returning None reward")
            return next_obs, reward_val, done, {"response": action, "metadata": metadata}
        else:
            if isinstance(action, str):
                llm_response = action
            elif isinstance(action, list):
                # Find the finish tool call
                finish_action = None
                for tool_call in action:
                    if tool_call.get("function", {}).get("name") == "finish":
                        finish_action = tool_call
                        break
                if finish_action:
                    arguments = finish_action.get("function", {}).get("arguments", {})
                    llm_response = arguments.get("response", "")
                else:
                    # No finish tool call found, use the action itself
                    llm_response = str(action)
            task_info = self.task if self.task is not None else {}
            if raw_reward is not None:
                return {}, raw_reward, done, {"response": action, "metadata": {}}
            reward_output = self.res_reward_fn(action=llm_response, task_info=task_info)
            return  {}, reward_output.reward, done, {"response": action, "metadata": reward_output.metadata}


    @staticmethod
    def from_dict(env_args: dict) -> "ProxyEnvironment":
        """
        Create a ProxyEnvironment instance from a dictionary of arguments.

        Extracts and removes environment-specific keys (custom_reward_function, tools, task,
        tool_map, max_steps) from the dictionary, then passes the remaining args to the
        ProxyEnvironment constructor.

        Args:
            env_args: Dictionary of environment configuration arguments.

        Returns:
            ProxyEnvironment: A new environment instance.
        """
        custom_reward_function = env_args.pop("custom_reward_function", None)
        tools = env_args.pop("tools", None)
        task = env_args.pop("task", None)
        tool_map = env_args.pop("tool_map", None)
        max_steps = env_args.pop("max_steps", 10)
        if custom_reward_function:
            tool_reward_fn_path = custom_reward_function.get("tool_reward_fn_path", None)
            res_reward_fn_path = custom_reward_function.get("res_reward_fn_path", None)
            if res_reward_fn_path is None:
                raise ValueError("res_reward_fn_path cannot be None, please configure it in custom_reward_function")
            if tool_reward_fn_path is None:
                logger.warning("tool_reward_fn_path is not configured, tool call rewards will not be calculated")
        else:
            logger.error("custom_reward_function is required, please configure it in env_args")
            tool_reward_fn_path = None
            res_reward_fn_path = None
        return ProxyEnvironment(task=task, tools=tools, tool_map=tool_map, max_steps=max_steps, tool_reward_fn_path=tool_reward_fn_path, res_reward_fn_path=res_reward_fn_path)
