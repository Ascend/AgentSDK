#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import importlib
import logging

from rllm.tools.tool_base import Tool
from aura.runner.agent_engine_wrapper.base.environment.base_env import BaseEnv

logger = logging.getLogger(__name__)


def import_external_reward_fn(fn_path: str):
    """
    参考 verl 风格的动态函数导入器
    Args:
        fn_path: 形如 'agents.dtn.reward.my_tool_fn' 的字符串
    Returns:
        可被调用的 python 函数对象
    """
    # 兼容 yaml 解析出来的 null (Python 的 None) 或者 "null" 字符串
    if fn_path is None or str(fn_path).strip().lower() == "null":
        return None
    try:
        # 分离出模块路径和具体的函数名
        # 例如: 'agents.dtn_code_agent.reward.dtn_code_reward' 和 'dtn_code_tool_reward_fn'
        module_path, fn_name = fn_path.rsplit('.', 1)

        # 动态导入该模块
        module = importlib.import_module(module_path)

        # 从模块中获取函数对象
        reward_fn = getattr(module, fn_name)

        logger.info(f"Successfully loaded custom reward function: {fn_name} from {module_path}")
        return reward_fn

    except ValueError:
        raise ValueError(f"[Import Error] 路径格式错误，需满足 'module.func' 格式: {fn_path}")
    except ImportError as e:
        raise ImportError(f"[Import Error] 找不到模块 {module_path}，请检查 PYTHONPATH。错误详情: {e}")
    except AttributeError as e:
        raise AttributeError(f"[Import Error] 模块 {module_path} 中不存在函数 {fn_name}。错误详情: {e}")


class ProxyEnvironment(BaseEnv):
    """
    A simple environment for tool-based agents that provides questions and evaluates responses.
    """

    def __init__(
        self,
        task: dict | None = None,
        tools: list[str] | None = None,
        tool_map: dict[str, type[Tool]] | None = None,
        tool_reward_fn_path: str | None = None,
        res_reward_fn_path: str | None = None,
        max_steps=10,
    ):
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
        # 不需要调用工具，因此不管
        self.step_count = 0
        self.max_steps = max_steps

        self.task = task
        self.tool_reward_fn = import_external_reward_fn(tool_reward_fn_path)
        self.res_reward_fn = import_external_reward_fn(res_reward_fn_path)

    def reset(self):
        """Reset the environment and return initial observations."""
        self.step_count = 0

        return self.task, {}

    def step(self, action: list[dict] | str | dict, done: bool, tool_outputs: str | list[dict], raw_reward=None):
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
                # 因为没法真正按 ID 跑，我们直接把整段观测赋给解析出的工具
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
            return {}, reward_output.reward, done, {"response": action, "metadata": reward_output.metadata}

    @staticmethod
    def from_dict(env_args: dict) -> "ProxyEnvironment":
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
            raise ValueError("custom_reward_function is required, please configure it in env_args")
        return ProxyEnvironment(
            task=task,
            tools=tools,
            tool_map=tool_map,
            max_steps=max_steps,
            tool_reward_fn_path=tool_reward_fn_path,
            res_reward_fn_path=res_reward_fn_path,
        )
