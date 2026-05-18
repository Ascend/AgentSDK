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

import copy
import json
import logging
import uuid
from typing import Any

from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import Tool
from agents.search_r1_agent.prompt import SEARCH_R1_PROMPT
from aura.runner.agent_engine_wrapper.base.agent.base_agent import Action, BaseAgent, Step, Trajectory
from agents.search_r1_agent.parser.search_r1_tool_parser import QwenToolParser


logger = logging.getLogger(__name__)


class SearchR1Agent(BaseAgent):
    """
    An tool agent that can use tools to interact with the environment,
    refactored to follow the BaseAgent abstraction.
    """

    def __init__(
        self,
        parser_name="qwen",
        system_prompt: str = SEARCH_R1_PROMPT,
        tools: list[str] | None = None,
        tool_map: dict[str, type[Tool]] | None = None,
    ):
        """
        Initialize the SearchR1Agent.

        Args:
            system_prompt: System prompt for the agent.
            parser_name: Name of the parser to use for tool calls.
            tools: List of tool names available to the agent (legacy behavior).
            tool_map: Dictionary mapping tool names to Tool classes (new behavior).
        """
        if tool_map is not None and tools is not None:
            raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters")

        # Initialize MultiTool with either tools or tool_map
        if tool_map is not None:
            self.tools = MultiTool(tool_map=tool_map)
        elif tools is not None:
            self.tools = MultiTool(tools=tools)
        else:
            self.tools = MultiTool(tools=[])

        self.tool_parser = QwenToolParser()

        self.tools_prompt = self.tool_parser.get_tool_prompt(json.dumps(self.tools.json, indent=2))

        # Initialize state according to BaseAgent
        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.current_observation = None
        self.reset()  # Call reset to set initial state

    def _format_observation_as_messages(self, obs: Any) -> list[dict]:
        """Helper to format observation into messages."""
        messages = []
        if isinstance(obs, dict):
            if "task" in obs and "problem" in obs["task"]:
                messages.append({"role": "user", "content": obs["task"]["problem"]})
            elif "task" in obs and "question" in obs["task"]:  # 针对MATH场景训练
                messages.append({"role": "user", "content": obs["task"]["question"]})
            elif "problem" in obs:
                messages.append({"role": "user", "content": obs["problem"]})
            elif "tool_outputs" in obs:
                # Format tool outputs from environment observation
                for tool_call_id, tool_output_str in obs["tool_outputs"].items():
                    messages.append(
                        {
                            "role": "tool",
                            "content": tool_output_str,
                            "tool_call_id": tool_call_id,
                        }
                    )
        elif isinstance(obs, str):
            messages.append({"role": "user", "content": obs})
        elif obs:
            messages.append({"role": "user", "content": str(obs)})

        return messages

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's state based on environment feedback.
        Formats observation and updates the trajectory.
        """

        # Format the observation for the next model call
        obs_messages = self._format_observation_as_messages(observation)
        self.messages.extend(obs_messages)
        self.current_observation = observation

    def truncate_poll_output(self, text: str) -> str:
        if '</search>' in text:
            return text.split('</search>')[0] + '</search>'
        elif '</answer>' in text:
            return text.split('</answer>')[0] + '</answer>'
        else:
            return text

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's state based on the model's response.
        Parses the response, updates messages, and the current step in the trajectory.
        """
        tool_calls_dict = []
        assistant_content = response
        assistant_content = self.truncate_poll_output(assistant_content)  # 截断防止幻觉
        # Attempt to parse tool calls from string response
        try:
            tool_calls = self.tool_parser.parse(response)
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": tool_call.to_dict(),
                }
                for tool_call in tool_calls
            ]

        except Exception as e:
            logger.error(f"Failed to parse tool calls from string response: {e}")
            tool_calls_dict = []  # Indicate no valid tool calls parsed

        # Append assistant message to chat history
        assistant_message = {"role": "assistant", "content": assistant_content}
        if len(tool_calls_dict) > 0:
            # Ensure arguments within tool_calls_dict are strings if needed by downstream processing
            for call in tool_calls_dict:
                if isinstance(call.get("function", {}).get("arguments"), dict):
                    call["function"]["arguments"] = json.dumps(call["function"]["arguments"], ensure_ascii=False)
        else:
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": {
                            "response": assistant_content,
                        },
                    },
                }
            ]

        self.messages.append(assistant_message)

        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=tool_calls_dict,
            model_response=response,
            observation=self.current_observation,
        )
        self._trajectory.steps.append(new_step)

        return Action(action=tool_calls_dict)

    def reset(self):
        """Resets the agent's state for a new episode."""
        self._trajectory = Trajectory()
        self.messages = []

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Returns the current message history for the model."""
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """Returns the trajectory recorded so far."""
        return self._trajectory
