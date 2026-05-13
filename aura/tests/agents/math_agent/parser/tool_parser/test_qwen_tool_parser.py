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


class TestQwenToolParserDetailed:
    """Detailed tests for QwenToolParser class."""

    def test_parse_with_nested_json(self):
        """Test parsing with nested JSON arguments."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        text = f'{parser.tool_call_begin}{{"name": "python", "arguments": {{"code": "def f():\\n    return 1", "timeout": 10}}}}{parser.tool_call_end}'
        
        result = parser.parse(text)
        
        assert len(result) == 1
        assert result[0].name == "python"

    def test_parse_with_special_characters(self):
        """Test parsing with special characters in arguments."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        text = f'{parser.tool_call_begin}{{"name": "search", "arguments": {{"query": "hello world!"}}}}{parser.tool_call_end}'
        
        result = parser.parse(text)
        
        assert len(result) == 1
        assert result[0].name == "search"

    def test_parse_with_empty_arguments(self):
        """Test parsing with empty arguments."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        text = f'{parser.tool_call_begin}{{"name": "finish", "arguments": {{}}}}{parser.tool_call_end}'
        
        result = parser.parse(text)
        
        assert len(result) == 1
        assert result[0].name == "finish"
        assert result[0].arguments == {}

    def test_parse_with_unicode(self):
        """Test parsing with unicode characters."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        text = f'{parser.tool_call_begin}{{"name": "translate", "arguments": {{"text": "你好世界"}}}}{parser.tool_call_end}'
        
        result = parser.parse(text)
        
        assert len(result) == 1
        assert result[0].name == "translate"

    def test_parse_mixed_content(self):
        """Test parsing text with mixed content."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        text = f'''Let me think about this problem.
First, I'll use Python to calculate.
{parser.tool_call_begin}{{"name": "python", "arguments": {{"code": "print(42)"}}}}{parser.tool_call_end}
The answer is 42.'''
        
        result = parser.parse(text)
        
        assert len(result) == 1
        assert result[0].name == "python"

    def test_parse_consecutive_tool_calls(self):
        """Test parsing consecutive tool calls without text between."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        text = f'{parser.tool_call_begin}{{"name": "a", "arguments": {{}}}}{parser.tool_call_end}{parser.tool_call_begin}{{"name": "b", "arguments": {{}}}}{parser.tool_call_end}'
        
        result = parser.parse(text)
        
        assert len(result) == 2
        assert result[0].name == "a"
        assert result[1].name == "b"

    def test_parse_qwen_tool_calls_malformed_json(self):
        """Test parse_qwen_tool_calls with malformed JSON."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        text = f'{parser.tool_call_begin}{{name: "invalid"}}{parser.tool_call_end}'
        
        result = parser.parse_qwen_tool_calls(text)
        
        assert len(result) == 0

    def test_parse_qwen_tool_calls_missing_name(self):
        """Test parse_qwen_tool_calls with missing name field."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        text = f'{parser.tool_call_begin}{{"arguments": {{"a": 1}}}}{parser.tool_call_end}'
        
        with pytest.raises(KeyError):
            result = parser.parse_qwen_tool_calls(text)

    def test_get_tool_prompt_format(self):
        """Test get_tool_prompt output format."""
        from agents.math_agent.parser.tool_parser.qwen_tool_parser import QwenToolParser
        
        parser = QwenToolParser()
        tools_schema = '[{"type": "function", "function": {"name": "python"}}]'
        
        result = parser.get_tool_prompt(tools_schema)
        
        assert parser.tool_call_begin in result
        assert parser.tool_call_end in result
        assert "function" in result.lower()
