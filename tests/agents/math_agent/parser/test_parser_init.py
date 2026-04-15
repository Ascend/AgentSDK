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
import pytest


class TestParserRegistry:
    """Tests for parser registry."""

    def test_parser_registry_has_r1(self):
        """Test that PARSER_REGISTRY has r1 parser."""
        from agents.math_agent.parser import PARSER_REGISTRY
        
        assert "r1" in PARSER_REGISTRY

    def test_parser_registry_has_qwen(self):
        """Test that PARSER_REGISTRY has qwen parser."""
        from agents.math_agent.parser import PARSER_REGISTRY
        
        assert "qwen" in PARSER_REGISTRY

    def test_get_tool_parser_r1(self):
        """Test get_tool_parser returns R1ToolParser for r1."""
        from agents.math_agent.parser import get_tool_parser, R1ToolParser
        
        result = get_tool_parser("r1")
        
        assert result == R1ToolParser

    def test_get_tool_parser_qwen(self):
        """Test get_tool_parser returns QwenToolParser for qwen."""
        from agents.math_agent.parser import get_tool_parser, QwenToolParser
        
        result = get_tool_parser("qwen")
        
        assert result == QwenToolParser

    def test_get_tool_parser_invalid(self):
        """Test get_tool_parser raises error for invalid parser name."""
        from agents.math_agent.parser import get_tool_parser
        
        with pytest.raises(ValueError):
            get_tool_parser("invalid_parser")


class TestQwenToolParser:
    """Tests for QwenToolParser class."""

    def test_init(self):
        """Test QwenToolParser initialization."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        
        assert parser.tool_call_begin == parser.tool_call_begin
        assert parser.tool_call_end == parser.tool_call_end
        assert parser.tool_output_begin == parser.tool_output_begin
        assert parser.tool_output_end == parser.tool_output_end

    def test_parse_single_tool_call(self):
        """Test parsing single tool call."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        text = f'Let me solve this.{parser.tool_call_begin}{{"name": "python", "arguments": {{"code": "print(1)"}}}}{parser.tool_call_end}'
        
        result = parser.parse(text)
        
        assert len(result) == 1
        assert result[0].name == "python"
        assert result[0].arguments == {"code": "print(1)"}

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        text = f'{parser.tool_call_begin}{{"name": "python", "arguments": {{"code": "a"}}}}{parser.tool_call_end}{parser.tool_call_begin}{{"name": "calculator", "arguments": {{"expr": "1+1"}}}}{parser.tool_call_end}'
        
        result = parser.parse(text)
        
        assert len(result) == 2
        assert result[0].name == "python"
        assert result[1].name == "calculator"

    def test_parse_no_tool_calls(self):
        """Test parsing text without tool calls."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        text = "This is just regular text without tool calls."
        
        result = parser.parse(text)
        
        assert len(result) == 0

    def test_parse_incomplete_tool_call(self):
        """Test parsing incomplete tool call."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        begin = parser.tool_call_begin
        text = f'{begin}{{"name": "python", "arguments": {{"code": "print(1)"}}}}'
        
        result = parser.parse(text)
        
        assert len(result) == 0

    def test_parse_invalid_json(self):
        """Test parsing with invalid JSON."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        text = f'{parser.tool_call_begin}{{invalid json}}{parser.tool_call_end}'
        
        result = parser.parse(text)
        
        assert len(result) == 0

    def test_get_tool_prompt(self):
        """Test get_tool_prompt method."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        tools_schema = '{"name": "python"}'
        
        result = parser.get_tool_prompt(tools_schema)
        
        assert "python" in result
        assert "<tools>" in result
        assert "</tools>" in result

    def test_parse_qwen_tool_calls(self):
        """Test parse_qwen_tool_calls method."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        text = f'{parser.tool_call_begin}{{"name": "test", "arguments": {{"a": 1}}}}{parser.tool_call_end}'
        
        result = parser.parse_qwen_tool_calls(text)
        
        assert len(result) == 1
        assert result[0]["name"] == "test"
        assert result[0]["arguments"] == {"a": 1}


class TestR1ToolParser:
    """Tests for R1ToolParser class."""

    def test_init(self):
        """Test R1ToolParser initialization."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        
        assert parser.tool_calls_begin == "<｜tool▁call▁begin｜>"
        assert parser.tool_calls_end == "<｜tool▁calls▁end｜>"
        assert parser.tool_call_begin == "<｜tool▁call▁begin｜>"
        assert parser.tool_call_end == "<｜tool▁call▁end｜>"
        assert parser.tool_sep == "<｜tool▁sep｜>"

    def test_parse_single_tool_call(self):
        """Test parsing single R1 tool call."""
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
        assert result[0].arguments == {"code": "print(1)"}

    def test_parse_no_tool_calls(self):
        """Test parsing text without R1 tool calls."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        text = "This is regular text without tool calls."
        
        result = parser.parse(text)
        
        assert len(result) == 0

    def test_parse_incomplete_tool_call(self):
        """Test parsing incomplete R1 tool call."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        text = "<｜tool▁call▁begin｜>function<｜tool▁sep｜>python"
        
        result = parser.parse(text)
        
        assert len(result) == 0

    def test_get_tool_prompt(self):
        """Test get_tool_prompt method."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        tools_schema = '{"name": "python"}'
        
        result = parser.get_tool_prompt(tools_schema)
        
        assert "python" in result
        assert "<tools>" in result
        assert "</tools>" in result

    def test_parse_r1_tool_calls_with_json_block(self):
        """Test parse_r1_tool_calls with JSON block."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        text = '''<｜tool▁call▁begin｜>function<｜tool▁sep｜>calculator
```json
{"expr": "1+1"}
```
<｜tool▁call▁end｜>'''
        
        result = parser.parse_r1_tool_calls(text)
        
        assert len(result) == 1
        assert result[0]["name"] == "calculator"
        assert result[0]["arguments"] == {"expr": "1+1"}

    def test_parse_r1_tool_calls_missing_json(self):
        """Test parse_r1_tool_calls when JSON block is missing."""
        from agents.math_agent.parser.tool_parser.r1_tool_parser import R1ToolParser
        
        parser = R1ToolParser()
        text = "<｜tool▁call▁begin｜>function<｜tool▁sep｜>python<｜tool▁call▁end｜>"
        
        result = parser.parse_r1_tool_calls(text)
        
        assert len(result) == 0
