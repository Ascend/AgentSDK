#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import sys
import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies():
    """Mock all external dependencies for r1_tool_parser tests."""
    yield {}


class TestR1ToolParserDetailed:
    """Detailed tests for R1ToolParser class."""

    def test_parse_with_multiline_json(self, mock_dependencies):
        """Test parsing with multiline JSON arguments."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        text = '''<｜tool▁call▁begin｜>function<｜tool▁sep｜>python
```json
{
    "code": "def f():\\n    return 1",
    "timeout": 10
}
```
<｜tool▁call▁end｜>'''
        
        result = parser.parse(text)
        
        assert len(result) == 1
        assert result[0].name == "python"

    def test_parse_with_special_characters(self, mock_dependencies):
        """Test parsing with special characters in arguments."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        text = '''<｜tool▁call▁begin｜>function<｜tool▁sep｜>search
```json
{"query": "hello world!"}
```
<｜tool▁call▁end｜>'''
        
        result = parser.parse(text)
        
        assert len(result) == 1
        assert result[0].name == "search"

    def test_parse_with_empty_arguments(self, mock_dependencies):
        """Test parsing with empty arguments."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        text = '''<｜tool▁call▁begin｜>function<｜tool▁sep｜>finish
```json
{}
```
<｜tool▁call▁end｜>'''
        
        result = parser.parse(text)
        
        assert len(result) == 1
        assert result[0].name == "finish"
        assert result[0].arguments == {}

    def test_parse_multiple_tool_calls(self, mock_dependencies):
        """Test parsing multiple tool calls."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        text = '''<｜tool▁call▁begin｜>function<｜tool▁sep｜>python
```json
{"code": "a = 1"}
```
<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>calculator
```json
{"expr": "1+1"}
```
<｜tool▁call▁end｜>'''
        
        result = parser.parse(text)
        
        assert len(result) == 2
        assert result[0].name == "python"
        assert result[1].name == "calculator"

    def test_parse_with_unicode(self, mock_dependencies):
        """Test parsing with unicode characters."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        text = '''<｜tool▁call▁begin｜>function<｜tool▁sep｜>translate
```json
{"text": "你好世界"}
```
<｜tool▁call▁end｜>'''
        
        result = parser.parse(text)
        
        assert len(result) == 1
        assert result[0].name == "translate"

    def test_parse_mixed_content(self, mock_dependencies):
        """Test parsing text with mixed content."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        text = '''Let me think about this problem.
First, I'll use Python to calculate.
<｜tool▁call▁begin｜>function<｜tool▁sep｜>python
```json
{"code": "print(42)"}
```
<｜tool▁call▁end｜>
The answer is 42.'''
        
        result = parser.parse(text)
        
        assert len(result) == 1
        assert result[0].name == "python"

    def test_parse_without_json_block_marker(self, mock_dependencies):
        """Test parsing without proper JSON block marker."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        text = '''<｜tool▁call▁begin｜>function<｜tool▁sep｜>python
{"code": "print(1)"}
<｜tool▁call▁end｜>'''
        
        result = parser.parse(text)
        
        assert len(result) == 0

    def test_parse_r1_tool_calls_malformed_json(self, mock_dependencies):
        """Test parse_r1_tool_calls with malformed JSON."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        text = '''<｜tool▁call▁begin｜>function<｜tool▁sep｜>python
```json
{name: "invalid"}
```
<｜tool▁call▁end｜>'''
        
        result = parser.parse_r1_tool_calls(text)
        
        assert len(result) == 0

    def test_parse_r1_tool_calls_missing_function_prefix(self, mock_dependencies):
        """Test parse_r1_tool_calls with missing function prefix."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        text = '''<｜tool▁call▁begin｜>python
```json
{"code": "print(1)"}
```
<｜tool▁call▁end｜>'''
        
        result = parser.parse_r1_tool_calls(text)
        
        assert len(result) == 0

    def test_get_tool_prompt_format(self, mock_dependencies):
        """Test get_tool_prompt output format."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        tools_schema = '[{"type": "function", "function": {"name": "python"}}]'
        
        result = parser.get_tool_prompt(tools_schema)
        
        assert "<tools>" in result
        assert "</tools>" in result
        assert "function" in result.lower()

    def test_parse_with_nested_json(self, mock_dependencies):
        """Test parsing with nested JSON arguments."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        text = '''<｜tool▁call▁begin｜>function<｜tool▁sep｜>python
```json
{"config": {"timeout": 10, "retries": 3}, "code": "print(1)"}
```
<｜tool▁call▁end｜>'''
        
        result = parser.parse(text)
        
        assert len(result) == 1
        assert result[0].name == "python"
        assert "config" in result[0].arguments

    def test_parse_with_function_name_on_newline(self, mock_dependencies):
        """Test parsing with function name on same line as separator."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        text = '''<｜tool▁call▁begin｜>function<｜tool▁sep｜>python
```json
{"code": "print(1)"}
```
<｜tool▁call▁end｜>'''
        
        result = parser.parse(text)
        
        assert len(result) == 1
        assert result[0].name == "python"
