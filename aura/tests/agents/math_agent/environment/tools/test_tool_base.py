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
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies():
    """Mock all external dependencies for tool_base tests."""
    mock_function_to_dict = MagicMock()
    mock_function_to_dict.return_value = {
        "type": "function",
        "function": {
            "name": "test_function",
            "description": "A test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
            },
        },
    }

    with (
        patch("rllm.tools.utils.function_to_dict", mock_function_to_dict),
    ):
        yield {
            "function_to_dict": mock_function_to_dict,
        }


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self, mock_dependencies):
        """Test ToolCall creation with name and arguments."""
        from agents.math_agent.environment.tools.tool_base import ToolCall
        
        tool_call = ToolCall(name="test_tool", arguments={"a": 1, "b": 2})
        
        assert tool_call.name == "test_tool"
        assert tool_call.arguments == {"a": 1, "b": 2}

    def test_tool_call_to_dict(self, mock_dependencies):
        """Test ToolCall to_dict method."""
        from agents.math_agent.environment.tools.tool_base import ToolCall
        
        tool_call = ToolCall(name="test_tool", arguments={"a": 1, "b": 2})
        result = tool_call.to_dict()
        
        assert result["name"] == "test_tool"
        assert result["arguments"] == {"a": 1, "b": 2}

    def test_tool_call_empty_arguments(self, mock_dependencies):
        """Test ToolCall with empty arguments."""
        from agents.math_agent.environment.tools.tool_base import ToolCall
        
        tool_call = ToolCall(name="empty_tool", arguments={})
        
        assert tool_call.name == "empty_tool"
        assert tool_call.arguments == {}


class TestToolOutput:
    """Tests for ToolOutput dataclass."""

    def test_tool_output_creation_with_output(self, mock_dependencies):
        """Test ToolOutput creation with output."""
        from agents.math_agent.environment.tools.tool_base import ToolOutput
        
        output = ToolOutput(name="test_tool", output="result")
        
        assert output.name == "test_tool"
        assert output.output == "result"
        assert output.error is None

    def test_tool_output_creation_with_error(self, mock_dependencies):
        """Test ToolOutput creation with error."""
        from agents.math_agent.environment.tools.tool_base import ToolOutput
        
        output = ToolOutput(name="test_tool", error="Something went wrong")
        
        assert output.name == "test_tool"
        assert output.output is None
        assert output.error == "Something went wrong"

    def test_tool_output_str_with_error(self, mock_dependencies):
        """Test ToolOutput __str__ with error."""
        from agents.math_agent.environment.tools.tool_base import ToolOutput
        
        output = ToolOutput(name="test_tool", error="Error message")
        result = str(output)
        
        assert result == "Error: Error message"

    def test_tool_output_str_with_none_output(self, mock_dependencies):
        """Test ToolOutput __str__ with None output."""
        from agents.math_agent.environment.tools.tool_base import ToolOutput
        
        output = ToolOutput(name="test_tool", output=None)
        result = str(output)
        
        assert result == ""

    def test_tool_output_str_with_dict_output(self, mock_dependencies):
        """Test ToolOutput __str__ with dict output."""
        from agents.math_agent.environment.tools.tool_base import ToolOutput
        
        output = ToolOutput(name="test_tool", output={"key": "value"})
        result = str(output)
        
        assert '"key": "value"' in result

    def test_tool_output_str_with_list_output(self, mock_dependencies):
        """Test ToolOutput __str__ with list output."""
        from agents.math_agent.environment.tools.tool_base import ToolOutput
        
        output = ToolOutput(name="test_tool", output=[1, 2, 3])
        result = str(output)
        
        assert "[1, 2, 3]" in result

    def test_tool_output_str_with_string_output(self, mock_dependencies):
        """Test ToolOutput __str__ with string output."""
        from agents.math_agent.environment.tools.tool_base import ToolOutput
        
        output = ToolOutput(name="test_tool", output="simple string")
        result = str(output)
        
        assert result == "simple string"

    def test_tool_output_to_string(self, mock_dependencies):
        """Test ToolOutput to_string method."""
        from agents.math_agent.environment.tools.tool_base import ToolOutput
        
        output = ToolOutput(name="test_tool", output="result")
        result = output.to_string()
        
        assert result == "result"


class TestTool:
    """Tests for Tool class."""

    def test_tool_init_with_name_and_description(self, mock_dependencies):
        """Test Tool initialization with name and description."""
        from agents.math_agent.environment.tools.tool_base import Tool
        
        class TestTool(Tool):
            @property
            def json(self):
                return {
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "description": self.description,
                        "parameters": {},
                    },
                }
        
        tool = TestTool(name="test_tool", description="A test tool")
        
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"

    def test_tool_init_with_function(self, mock_dependencies):
        """Test Tool initialization with function."""
        from agents.math_agent.environment.tools.tool_base import Tool
        
        def test_func(a: int, b: int) -> int:
            return a + b
        
        tool = Tool(function=test_func)
        
        assert tool.name == "test_function"
        assert tool.description == "A test function"

    def test_tool_forward_with_function(self, mock_dependencies):
        """Test Tool forward with function."""
        from agents.math_agent.environment.tools.tool_base import Tool
        
        def add(a: int, b: int) -> int:
            return a + b
        
        tool = Tool(function=add)
        result = tool.forward(a=1, b=2)
        
        assert result.name == "test_function"
        assert result.output == 3

    def test_tool_forward_with_error(self, mock_dependencies):
        """Test Tool forward with error in function."""
        from agents.math_agent.environment.tools.tool_base import Tool
        
        def error_func():
            raise ValueError("Test error")
        
        tool = Tool(function=error_func)
        result = tool.forward()
        
        assert result.error is not None
        assert "ValueError" in result.error

    def test_tool_forward_not_implemented(self, mock_dependencies):
        """Test Tool forward without implementation."""
        from agents.math_agent.environment.tools.tool_base import Tool
        
        class TestTool(Tool):
            @property
            def json(self):
                return {"type": "function", "function": {"name": "test", "description": "test"}}
        
        tool = TestTool(name="test", description="test")
        
        with pytest.raises(NotImplementedError):
            tool.forward()

    def test_tool_call_sync(self, mock_dependencies):
        """Test Tool __call__ for sync execution."""
        from agents.math_agent.environment.tools.tool_base import Tool
        
        def add(a: int, b: int) -> int:
            return a + b
        
        tool = Tool(function=add)
        result = tool(a=1, b=2, use_async=False)
        
        assert result.output == 3

    def test_tool_call_auto_detect_sync(self, mock_dependencies):
        """Test Tool __call__ auto-detects sync implementation."""
        from agents.math_agent.environment.tools.tool_base import Tool
        
        def add(a: int, b: int) -> int:
            return a + b
        
        tool = Tool(function=add)
        result = tool(a=1, b=2)
        
        assert result.output == 3

    @pytest.mark.asyncio
    async def test_tool_call_with_use_async_true_no_async_impl(self, mock_dependencies):
        """Test Tool __call__ with use_async=True and sync implementation."""
        from agents.math_agent.environment.tools.tool_base import Tool
        
        def sync_only():
            return "sync result"
        
        tool = Tool(function=sync_only)
        
        result = await tool(use_async=True)
        assert result.output == "sync result"

    def test_tool_json_property(self, mock_dependencies):
        """Test Tool json property."""
        from agents.math_agent.environment.tools.tool_base import Tool
        
        def test_func(a: int) -> int:
            return a * 2
        
        tool = Tool(function=test_func)
        result = tool.json
        
        assert result["type"] == "function"
        assert result["function"]["name"] == "test_function"

    def test_tool_requires_name_or_function(self, mock_dependencies):
        """Test Tool requires name when no function provided."""
        from agents.math_agent.environment.tools.tool_base import Tool
        
        class TestTool(Tool):
            @property
            def json(self):
                return {"type": "function", "function": {"name": "test", "description": "test"}}
        
        with pytest.raises(ValueError):
            TestTool()

    def test_tool_requires_description_or_function(self, mock_dependencies):
        """Test Tool requires description when no function provided."""
        from agents.math_agent.environment.tools.tool_base import Tool
        
        class TestTool(Tool):
            @property
            def json(self):
                return {"type": "function", "function": {"name": "test", "description": "test"}}
        
        with pytest.raises(ValueError):
            TestTool(name="test")
