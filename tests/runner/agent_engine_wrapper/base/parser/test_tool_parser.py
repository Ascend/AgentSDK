# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for base/parser/tool_parser module: DeepSeekToolParser."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agentic_rl.runner.agent_engine_wrapper.base.parser.tool_parser import DeepSeekToolParser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return DeepSeekToolParser()


def _build_tool_call(name, arguments):
    """Helper to build a well-formed tool call string."""
    return (
        f"<｜tool▁call▁begin｜>{name}<｜tool▁sep｜>"
        f"{json.dumps(arguments)}<｜tool▁call▁end｜>"
    )


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestDeepSeekToolParserInit:
    def test_special_tokens_set(self, parser):
        assert parser.think_end == "</think>"
        assert parser.tool_calls_begin == "<｜tool▁calls▁begin｜>"
        assert parser.tool_calls_end == "<｜tool▁calls▁end｜>"
        assert parser.tool_call_begin == "<｜tool▁call▁begin｜>"
        assert parser.tool_call_end == "<｜tool▁call▁end｜>"
        assert parser.tool_sep == "<｜tool▁sep｜>"


# ---------------------------------------------------------------------------
# parse_deepseek_tool_calls tests
# ---------------------------------------------------------------------------

class TestParseDeepseekToolCalls:
    def test_single_tool_call(self, parser):
        text = _build_tool_call("search", {"query": "python"})
        result = parser.parse_deepseek_tool_calls(text)
        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["arguments"] == {"query": "python"}

    def test_multiple_tool_calls(self, parser):
        text = (
            _build_tool_call("search", {"query": "python"})
            + _build_tool_call("execute", {"code": "print(1)"})
        )
        result = parser.parse_deepseek_tool_calls(text)
        assert len(result) == 2
        assert result[0]["name"] == "search"
        assert result[1]["name"] == "execute"

    def test_no_tool_calls(self, parser):
        result = parser.parse_deepseek_tool_calls("No tool calls here")
        assert result == []

    def test_empty_string(self, parser):
        result = parser.parse_deepseek_tool_calls("")
        assert result == []

    def test_text_before_and_after_tool_call(self, parser):
        text = "Some thinking text " + _build_tool_call("fn", {"a": 1}) + " more text"
        result = parser.parse_deepseek_tool_calls(text)
        assert len(result) == 1
        assert result[0]["name"] == "fn"
        assert result[0]["arguments"] == {"a": 1}

    def test_missing_tool_sep_skips_call(self, parser):
        text = "<｜tool▁call▁begin｜>no_separator_here<｜tool▁call▁end｜>"
        result = parser.parse_deepseek_tool_calls(text)
        assert result == []

    def test_invalid_json_skips_call(self, parser):
        text = "<｜tool▁call▁begin｜>func<｜tool▁sep｜>not_valid_json<｜tool▁call▁end｜>"
        result = parser.parse_deepseek_tool_calls(text)
        assert result == []

    def test_missing_end_token(self, parser):
        text = "<｜tool▁call▁begin｜>func<｜tool▁sep｜>{\"a\": 1}"
        result = parser.parse_deepseek_tool_calls(text)
        assert result == []

    def test_nested_json_arguments(self, parser):
        args = {"config": {"nested": {"deep": True}}, "list": [1, 2, 3]}
        text = _build_tool_call("complex_fn", args)
        result = parser.parse_deepseek_tool_calls(text)
        assert len(result) == 1
        assert result[0]["arguments"] == args

    def test_empty_arguments(self, parser):
        text = _build_tool_call("no_args_fn", {})
        result = parser.parse_deepseek_tool_calls(text)
        assert len(result) == 1
        assert result[0]["arguments"] == {}

    def test_tool_call_with_whitespace(self, parser):
        text = (
            "<｜tool▁call▁begin｜>  search  <｜tool▁sep｜>"
            '  {"query": "test"}  <｜tool▁call▁end｜>'
        )
        result = parser.parse_deepseek_tool_calls(text)
        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["arguments"] == {"query": "test"}

    def test_mixed_valid_and_invalid_calls(self, parser):
        valid = _build_tool_call("good_fn", {"x": 1})
        invalid = "<｜tool▁call▁begin｜>bad_fn<｜tool▁sep｜>not_json<｜tool▁call▁end｜>"
        text = valid + invalid
        result = parser.parse_deepseek_tool_calls(text)
        assert len(result) == 1
        assert result[0]["name"] == "good_fn"


# ---------------------------------------------------------------------------
# parse tests
# ---------------------------------------------------------------------------

class TestParse:
    @patch("agentic_rl.runner.agent_engine_wrapper.base.parser.tool_parser.ToolCall")
    def test_parse_returns_tool_call_objects(self, MockToolCall, parser):
        MockToolCall.side_effect = lambda name, arguments: MagicMock(name=name, arguments=arguments)
        text = _build_tool_call("search", {"query": "python"})
        result = parser.parse(text)
        assert len(result) == 1
        MockToolCall.assert_called_once_with(name="search", arguments={"query": "python"})

    @patch("agentic_rl.runner.agent_engine_wrapper.base.parser.tool_parser.ToolCall")
    def test_parse_empty_returns_empty_list(self, MockToolCall, parser):
        result = parser.parse("no tools")
        assert result == []
        MockToolCall.assert_not_called()

    @patch("agentic_rl.runner.agent_engine_wrapper.base.parser.tool_parser.ToolCall")
    def test_parse_multiple_calls(self, MockToolCall, parser):
        MockToolCall.side_effect = lambda name, arguments: MagicMock(name=name, arguments=arguments)
        text = _build_tool_call("fn1", {"a": 1}) + _build_tool_call("fn2", {"b": 2})
        result = parser.parse(text)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# get_tool_prompt tests
# ---------------------------------------------------------------------------

class TestGetToolPrompt:
    def test_returns_string_with_schema(self, parser):
        schema = '[{"name": "search", "parameters": {}}]'
        result = parser.get_tool_prompt(schema)
        assert schema in result
        assert "Tools" in result

    def test_contains_format_instructions(self, parser):
        result = parser.get_tool_prompt("[]")
        assert "<｜tool▁calls▁begin｜>" in result
        assert "<｜tool▁call▁begin｜>" in result
        assert "<｜tool▁sep｜>" in result
