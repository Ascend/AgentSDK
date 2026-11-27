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
