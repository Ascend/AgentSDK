#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------
# Copyright 2025 the LlamaFactory team.
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
import pytest
from agentic_rl.base.utils.formatter import (
    default_tool_formatter,
    default_tool_extractor,
    EmptyFormatter,
    StringFormatter,
    FunctionFormatter,
    ToolFormatter
)


class TestDefaultToolFormatter:
    def test_single_tool(self):
        tools = [
            {
                "name": "search",
                "description": "Search the web for information",
                "parameters": {
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        result = default_tool_formatter(tools)
        assert "Tool Name: search" in result
        assert "Tool Description: Search the web for information" in result
        assert "query (string, required): The search query" in result
        assert "num_results (integer): Number of results to return" in result
        assert "Action: tool name (one of [search])" in result

    def test_multiple_tools(self):
        tools = [
            {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "calculator",
                "description": "Perform calculations",
                "parameters": {
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ]
        result = default_tool_formatter(tools)
        assert "Tool Name: search" in result
        assert "Tool Name: calculator" in result
        assert "Action: tool name (one of [search, calculator])" in result

    def test_tool_with_enum(self):
        tools = [
            {
                "name": "weather",
                "description": "Get weather information",
                "parameters": {
                    "properties": {
                        "unit": {
                            "type": "string",
                            "description": "Temperature unit",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["unit"]
                }
            }
        ]
        result = default_tool_formatter(tools)
        assert "unit (string, required): Temperature unit, should be one of [celsius, fahrenheit]" in result

    def test_tool_with_array(self):
        tools = [
            {
                "name": "batch_process",
                "description": "Process multiple items",
                "parameters": {
                    "properties": {
                        "items": {
                            "type": "array",
                            "description": "List of items to process",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["items"]
                }
            }
        ]
        result = default_tool_formatter(tools)
        assert "items (array, required): List of items to process, where each item should be string" in result


class TestDefaultToolExtractor:
    def test_extract_valid_tool_call(self):
        content = """Action: search.
Action Input: {"query": "test"}"""
        result = default_tool_extractor(content)
        assert result == ("search", json.dumps({"query": "test"}, ensure_ascii=False))

    def test_extract_with_code_blocks(self):
        content = """Action: search
Action Input: ```{"query": "test"}```"""
        result = default_tool_extractor(content)
        assert result == ("search", json.dumps({"query": "test"}, ensure_ascii=False))

    def test_extract_with_quotes(self):
        content = '''Action: search
Action Input: "{\"query\": \"test\"}"'''  # Use triple single quotes to avoid nesting issues
        result = default_tool_extractor(content)
        assert result == ("search", json.dumps({"query": "test"}, ensure_ascii=False))

    def test_extract_with_extra_whitespace(self):
        content = """Action:    search   
Action Input:    {"query": "test"}   """
        result = default_tool_extractor(content)
        assert result == ("search", json.dumps({"query": "test"}, ensure_ascii=False))

    def test_extract_with_dotall(self):
        content = """Action: search.
Some intermediate text.
Action Input: {"query": "test"}"""
        result = default_tool_extractor(content)
        assert result == ("search", json.dumps({"query": "test"}, ensure_ascii=False))

    def test_extract_invalid_json(self):
        content = """Action: search
Action Input: invalid json"""
        result = default_tool_extractor(content)
        assert result == content

    def test_extract_no_action(self):
        content = "No action here"
        result = default_tool_extractor(content)
        assert result == content

    def test_extract_no_input(self):
        content = """Action: search
Action Input:"""
        result = default_tool_extractor(content)
        assert result == content


class TestEmptyFormatter:
    def test_no_placeholders(self):
        formatter = EmptyFormatter(slots=["Hello", "World"])
        result = formatter.apply()
        assert result == ["Hello", "World"]

    def test_with_dict_and_set(self):
        formatter = EmptyFormatter(slots=["Hello", {"key": "value"}, {"set_item"}])
        result = formatter.apply()
        assert result == ["Hello", {"key": "value"}, {"set_item"}]

    def test_with_placeholder_raises_error(self):
        with pytest.raises(ValueError, match="Empty formatter should not contain any placeholder."):
            EmptyFormatter(slots=["Hello {{name}}"])


class TestStringFormatter:
    def test_single_placeholder(self):
        formatter = StringFormatter(slots=["Hello {{name}}"])
        result = formatter.apply(name="World")
        assert result == ["Hello World"]

    def test_multiple_placeholders(self):
        formatter = StringFormatter(slots=["Hello {{name}}, you are {{age}} years old"])
        result = formatter.apply(name="John", age="30")
        assert result == ["Hello John, you are 30 years old"]

    def test_multiple_slots(self):
        formatter = StringFormatter(slots=["Hello {{name}}", "You are {{age}} years old"])
        result = formatter.apply(name="John", age="30")
        assert result == ["Hello John", "You are 30 years old"]

    def test_with_dict_and_set(self):
        formatter = StringFormatter(slots=["Hello {{name}}", {"key": "value"}, {"set_item"}])
        result = formatter.apply(name="World")
        assert result == ["Hello World", {"key": "value"}, {"set_item"}]

    def test_no_placeholder_raises_error(self):
        with pytest.raises(ValueError, match="A placeholder is required in the string formatter."):
            StringFormatter(slots=["Hello World"])

    def test_non_string_value_raises_error(self):
        formatter = StringFormatter(slots=["Hello {{name}}"])
        with pytest.raises(RuntimeError, match="Expected a string, got 123"):
            formatter.apply(name=123)

    def test_unknown_placeholder(self):
        formatter = StringFormatter(slots=["Hello {{name}}"])
        result = formatter.apply(name="World", unknown="value")
        assert result == ["Hello World"]

    def test_duplicate_placeholder(self):
        formatter = StringFormatter(slots=["{{name}} {{name}}"])
        result = formatter.apply(name="World")
        assert result == ["World {{name}}"]  # Only replaces first occurrence


class TestFunctionFormatter:
    def test_valid_function(self):
        formatter = FunctionFormatter(slots=["Function: {{name}}, Args: {{arguments}}"])
        content = json.dumps({
            "name": "search",
            "arguments": {"query": "test"}
        })
        result = formatter.apply(content=content)
        assert result == ["Function: search, Args: {\"query\": \"test\"}"]

    def test_invalid_json(self):
        formatter = FunctionFormatter(slots=["Function: {{name}}, Args: {{arguments}}"])
        content = "invalid json"
        result = formatter.apply(content=content)
        assert result == ["Function: , Args: "]

    def test_missing_name_or_arguments(self):
        with pytest.raises(ValueError, match="Name and arguments placeholders are required in the function formatter."):
            FunctionFormatter(slots=["Function: {{name}}"])

    def test_multiple_slots(self):
        formatter = FunctionFormatter(slots=[
            "Function: {{name}}",
            "Args: {{arguments}}",
            {"type": "function"}
        ])
        content = json.dumps({
            "name": "search",
            "arguments": {"query": "test"}
        })
        result = formatter.apply(content=content)
        assert result == [
            "Function: search",
            "Args: {\"query\": \"test\"}",
            {"type": "function"}
        ]

    def test_missing_name_placeholder(self):
        with pytest.raises(ValueError, match="Name and arguments placeholders are required in the function formatter."):
            FunctionFormatter(slots=["Args: {{arguments}}"])

    def test_missing_arguments_placeholder(self):
        with pytest.raises(ValueError, match="Name and arguments placeholders are required in the function formatter."):
            FunctionFormatter(slots=["Function: {{name}}"])


class TestToolFormatter:
    def test_valid_tools(self):
        formatter = ToolFormatter(slots=["{{tools}}"], tool_format="default")
        tools = [
            {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        content = json.dumps(tools)
        result = formatter.apply(content=content)
        assert "Tool Name: search" in result[0]
        assert "Action: tool name (one of [search])" in result[0]

    def test_empty_tools(self):
        formatter = ToolFormatter(slots=["{{tools}}"], tool_format="default")
        content = json.dumps([])
        result = formatter.apply(content=content)
        assert result == [""]

    def test_invalid_json(self):
        formatter = ToolFormatter(slots=["{{tools}}"], tool_format="default")
        content = "invalid json"
        result = formatter.apply(content=content)
        assert result == [""]

    def test_missing_tool_format(self):
        with pytest.raises(ValueError, match="Tool format was not found."):
            ToolFormatter(slots=["{{tools}}"])

    def test_extract_tool_call(self):
        formatter = ToolFormatter(slots=["{{tools}}"], tool_format="default")
        content = """Action: search.
Action Input: {"query": "test"}"""
        result = formatter.extract(content)
        assert result == ("search", json.dumps({"query": "test"}, ensure_ascii=False))

    def test_extract_invalid_tool_call(self):
        formatter = ToolFormatter(slots=["{{tools}}"], tool_format="default")
        content = "No tool call here"
        result = formatter.extract(content)
        assert result == content