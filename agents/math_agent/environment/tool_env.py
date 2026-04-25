# -*- coding: utf-8 -*-
#
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Wenxuan Huang.
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
import json
import queue
import threading
import warnings
from typing import Any, Optional

from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import Tool

from agents.math_agent.reward.reward_fn import RewardFunction, zero_reward
from agentic_rl.runner.agent_engine_wrapper.base.environment.base_env import BaseEnv
from agentic_rl.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


class ToolEnvironment(BaseEnv):
    """
    A simple environment for tool-based agents that provides questions
    and evaluates responses.
    """

    def __init__(
        self,
        task: Optional[dict] = None,
        tools: Optional[list[str]] = None,
        tool_map: Optional[dict[str, type[Tool]]] = None,
        reward_fn: Optional[RewardFunction] = None,
        max_steps: int = 10,
    ) -> None:
        """
        Initialize the ToolEnvironment.

        Args:
            task: Task information for the environment.
            tools: List of tool names to look up in the registry (legacy behavior).
            tool_map: Dictionary mapping tool names to Tool classes (new behavior).
            reward_fn: Reward function to use for evaluation.
            max_steps: Maximum number of steps allowed in the environment.

        Raises:
            ValueError: If both tools and tool_map are specified.
        """
        if tool_map is not None and tools is not None:
            raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters")

        self.step_count = 0
        self.max_steps = max_steps

        if tool_map is not None:
            self.tools = MultiTool(tool_map=tool_map)
        elif tools is not None:
            self.tools = MultiTool(tools=tools)
        else:
            self.tools = MultiTool(tools=[])

        self.task = task

        if reward_fn is None:
            warnings.warn("No reward function specified, will get 0 reward.", stacklevel=2)
            self.reward_fn = zero_reward
        else:
            self.reward_fn = reward_fn

    def reset(self) -> tuple[Any, dict]:
        """Reset the environment and return initial observations."""
        self.step_count = 0
        return self.task, {}

    def step(self, action: list[dict] | str | dict) -> tuple[dict, float, bool, dict]:
        """
        Take a step in the environment based on the action.

        Args:
            action: A tool-call list, a single tool-call dict, or a plain
                string response from the agent.

        Returns:
            A tuple of (next_observations, reward, done, info).
        """
        if isinstance(action, dict):
            action = [action]
        self.step_count += 1

        reward = 0
        done = self.step_count >= self.max_steps or isinstance(action, str)

        if isinstance(action, list) and action:
            for tool_call in action:
                if tool_call.get("function", {}).get("name") == "finish":
                    done = True
                    break

        if done:
            if isinstance(action, str):
                llm_response = action
            elif isinstance(action, list):
                finish_action = None
                for tool_call in action:
                    if tool_call.get("function", {}).get("name") == "finish":
                        finish_action = tool_call
                        break

                if finish_action:
                    arguments = finish_action.get("function", {}).get("arguments", {})
                    llm_response = arguments.get("response", "")
                else:
                    llm_response = str(action)

            task_info = self.task if self.task is not None else {}
            reward_output = self.reward_fn(task_info=task_info, action=llm_response)
            return (
                {},
                reward_output.reward,
                done,
                {"response": action, "metadata": reward_output.metadata},
            )

        tool_calls = action
        if not isinstance(tool_calls, list):
            raise TypeError(f"Expected tool_calls to be a list, got {type(tool_calls)}")

        tool_outputs = self._execute_tool_calls(tool_calls)
        next_obs = {"tool_outputs": tool_outputs}

        return next_obs, reward, done, {"response": action, "metadata": {}}

    def _execute_tool_calls(self, tool_calls: list[dict[Any, Any]]) -> dict[str, str]:
        """
        Execute tool calls concurrently in threads.

        Args:
            tool_calls: List of tool-call dicts with 'id' and 'function' keys.

        Returns:
            Mapping from tool-call id to the string output of each tool.
        """
        tool_outputs: dict[str, str] = {}
        output_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        threads = []

        def execute_tool(tool_call):
            tool_name = tool_call["function"]["name"]
            raw_args = tool_call["function"]["arguments"]
            tool_args = None
            if isinstance(raw_args, dict):
                tool_args = raw_args
            elif isinstance(raw_args, str):
                try:
                    tool_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    tool_args = {"code": raw_args}
            else:
                raise ValueError(f"Unsupported arguments type: {type(raw_args)}")
            tool_output = self.tools(tool_name=tool_name, **tool_args)
            tool_output_str = tool_output.to_string()

            output_queue.put((tool_call["id"], tool_output_str))

        for tool_call in tool_calls:
            thread = threading.Thread(target=execute_tool, args=(tool_call,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Collect results and store in order
        while not output_queue.empty():
            tool_call_id, output_str = output_queue.get()
            tool_outputs[tool_call_id] = output_str

        return tool_outputs

    @staticmethod
    def from_dict(env_args: dict) -> "ToolEnvironment":
        """
        Construct a ToolEnvironment from a configuration dict.

        The original dict is not mutated; recognised keys are extracted
        from a shallow copy and the remainder is passed as the task.

        Args:
            env_args: Configuration dictionary.

        Returns:
            A new ToolEnvironment instance.
        """
        args = dict(env_args)
        tools = args.pop("tools", None)
        tool_map = args.pop("tool_map", None)
        reward_fn = args.pop("reward_fn", None)
        max_steps = args.pop("max_steps", 10)

        return ToolEnvironment(
            task=args,
            tools=tools,
            tool_map=tool_map,
            max_steps=max_steps,
            reward_fn=reward_fn,
        )
