#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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

import json
from typing import Any

from rllm.parser.tool_parser.tool_parser_base import ToolParser
from rllm.tools.tool_base import ToolCall


class DeepSeekToolParser(ToolParser):
    def __init__(self):
        """
        Initialize the parser with DeepSeek-specific token markers.
        """
        self.think_end = "</think>"
        self.tool_calls_begin = "<｜tool▁calls▁begin｜>"
        self.tool_calls_end = "<｜tool▁calls▁end｜>"
        self.tool_call_begin = "<｜tool▁call▁begin｜>"
        self.tool_call_end = "<｜tool▁call▁end｜>"
        self.tool_sep = "<｜tool▁sep｜>"

    def parse(self, model_response: str) -> list[ToolCall]:
        """Parse tool calls from model output.

        Args:
            model_response (str): Text containing tool calls

        Returns:
            List of ToolCall objects extracted from the response.
        """
        tool_calls_dicts = self.parse_deepseek_tool_calls(model_response)
        tool_calls = [
            ToolCall(name=tc["name"], arguments=tc["arguments"]) 
            for tc in tool_calls_dicts
        ]
        return tool_calls

    def parse_deepseek_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """
        Parse tool calls from text using the DeepSeek special token format.
        """
        tool_calls = []

        call_idx = 0
        while True:
            call_idx = text.find(self.tool_call_begin, call_idx)
            if call_idx == -1:
                break

            call_start = call_idx + len(self.tool_call_begin)
            call_end = text.find(self.tool_call_end, call_start)
            if call_end == -1:
                break

            call_content = text[call_start:call_end].strip()

            func_sep_idx = call_content.find(self.tool_sep)

            if func_sep_idx == -1:
                call_idx = call_end + len(self.tool_call_end)
                continue

            function_name = call_content[:func_sep_idx].strip()

            args_str = call_content[func_sep_idx + len(self.tool_sep):].strip()

            try:
                args_json = json.loads(args_str)
            except json.JSONDecodeError:
                call_idx = call_end + len(self.tool_call_end)
                continue

            tool_calls.append({"name": function_name, "arguments": args_json})

            call_idx = call_end + len(self.tool_call_end)

        return tool_calls

    def get_tool_prompt(self, tools_schema: str) -> str:
        return f"""
# Tools

You may call one or more functions to assist with the user query.
<tools>
{tools_schema}
</tools>

For function call returns, you should first print <｜tool▁calls▁begin｜>

For each function call, you should return object like:
<｜tool▁call▁begin｜>function<｜tool▁sep｜><function_name>
"""