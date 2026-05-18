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

import json
from typing import Any

from rllm.parser.tool_parser.tool_parser_base import ToolParser
from rllm.tools.tool_base import ToolCall


class QwenToolParser(ToolParser):
    def __init__(self):
        """Initialize the parser with specified type and model.

        Args:
            model (str): Model name for tokenizer (optional)
            parser_type (str): Type of parser to use ('qwen' or other parsers you might add)
        """
        self.tool_call_begin = "<search>"
        self.tool_call_end = "</search>"
        self.tool_output_begin = "<information>"
        self.tool_output_end = "</information>"

    def parse(self, model_response: str) -> list[ToolCall]:
        """Parse tool calls from model output.

        Args:
            model_output (str): Text containing tool calls

        Returns:
            ToolInputs: Parsed tool calls
        """
        tool_calls_dicts = self.parse_qwen_search_calls(model_response)
        tool_calls = [ToolCall(name=tc["name"], arguments=tc["arguments"]) for tc in tool_calls_dicts]
        return tool_calls

    def parse_qwen_search_calls(self, text: str) -> list[dict[str, Any]]:
        """Parse tool calls from text using a simple token format.

        Format:
        <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>

        Returns:
            list[dict]: List of parsed tool calls, each containing 'name' and 'parameters'
        """

        tool_calls: list[dict[str, Any]] = []

        # Return empty list if no tool calls found
        if self.tool_call_begin not in text:
            return tool_calls

        # Process all tool calls in the text
        while self.tool_call_begin in text:
            start = text.find(self.tool_call_begin) + len(self.tool_call_begin)
            end = text.find(self.tool_call_end)
            if end == -1:
                break

            # Extract and parse the JSON content
            search_content = text[start:end].strip()
            json_content = json.dumps({"name": "search", "arguments": {"query_list": [search_content]}})
            try:
                call_data = json.loads(json_content)
                # Convert to common format matching parse_tool_calls output
                tool_calls.append({"name": call_data["name"], "arguments": call_data["arguments"]})
            except json.JSONDecodeError:
                print(f"Error parsing tool call: {json_content}")
                text = text[end + len(self.tool_call_end) :]
                continue

            # Move to next potential tool call
            text = text[end + len(self.tool_call_end) :]

        return tool_calls

    def get_tool_prompt(self, tools_schema: str) -> str:
        return f"""
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_schema}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
"""
