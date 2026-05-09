#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
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
from rllm.tools import tool_registry
from rllm.tools.tool_base import Tool, ToolOutput


class MultiTool(Tool):
    def __init__(self, tools: list[str] | None = None, tool_map: dict[str, type[Tool]] | None = None):
        """
        Initialize MultiTool with either tool names or a tool_map directly.

        Args:
            tools: List of tool names to look up in the registry (legacy behavior)
            tool_map: Dictionary mapping tool names to Tool classes (new behavior)
        """
        if tool_map is not None and tools is not None:
            raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters")

        if tool_map is not None:
            # New behavior: use provided tool_map with tool classes
            self.tools = list(tool_map.keys())
            # Instantiate tool classes with the name parameter
            self.tool_map = {}
            for name, tool_cls in tool_map.items():
                self.tool_map[name] = tool_cls(name=name)
        elif tools is not None:
            if not all(tool in tool_registry for tool in tools):
                raise ValueError("All tools must be in the registry")
            self.tools = tools
            # Filter out None values from tool_registry.instantiate
            self.tool_map = {}
            for tool in tools:
                tool_instance = tool_registry.instantiate(tool)
                if tool_instance is not None:
                    self.tool_map[tool] = tool_instance
        else:
            # Default to empty
            self.tools = []
            self.tool_map = {}

    @property
    def json(self):
        return [tool.json for tool in self.tool_map.values()]

    def forward(self, *args, tool_name: str, **kwargs) -> ToolOutput:
        if tool_name not in self.tool_map:
            return ToolOutput(name=tool_name, output=f"Tool {tool_name} not found in tool map")
        tool = self.tool_map[tool_name]
        return tool(*args, **kwargs)
