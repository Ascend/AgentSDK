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
from unittest.mock import Mock, MagicMock, patch, AsyncMock


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies():
    """Mock all external dependencies for mcp_tool tests."""
    
    class MockToolOutput:
        def __init__(self, name="", output=None, error=None):
            self.name = name
            self.output = output
            self.error = error
    
    class MockTool:
        def __init__(self, name="", description=""):
            self.name = name
            self.description = description
    
    with (
        patch("rllm.tools.tool_base.Tool", MockTool),
        patch("rllm.tools.tool_base.ToolOutput", MockToolOutput),
    ):
        yield {
            "tool_class": MockTool,
            "tool_output_class": MockToolOutput,
        }


class TestMCPTool:
    """Tests for MCPTool class."""

    def test_mcp_tool_init(self, mock_dependencies):
        """Test MCPTool initialization."""
        from agents.math_agent.environment.tools.mcp_tool import MCPTool
        
        mock_session = MagicMock()
        tool_schema = {"type": "object", "properties": {"query": {"type": "string"}}}
        
        tool = MCPTool(
            session=mock_session,
            tool_name="search_tool",
            tool_description="A search tool",
            tool_schema=tool_schema,
        )
        
        assert tool.name == "search_tool"
        assert tool.description == "A search tool"
        assert tool.session == mock_session

    def test_mcp_tool_json_property(self, mock_dependencies):
        """Test MCPTool json property."""
        from agents.math_agent.environment.tools.mcp_tool import MCPTool
        
        mock_session = MagicMock()
        tool_schema = {"type": "object", "properties": {"query": {"type": "string"}}}
        
        tool = MCPTool(
            session=mock_session,
            tool_name="search_tool",
            tool_description="A search tool",
            tool_schema=tool_schema,
        )
        
        result = tool.json
        
        assert result["type"] == "function"
        assert result["function"]["name"] == "search_tool"
        assert result["function"]["description"] == "A search tool"
        assert result["function"]["parameters"] == tool_schema

    @pytest.mark.asyncio
    async def test_async_forward_success(self, mock_dependencies):
        """Test MCPTool async_forward with successful call."""
        from agents.math_agent.environment.tools.mcp_tool import MCPTool
        
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = MagicMock()
        mock_result.content.text = "tool result"
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        tool_schema = {"type": "object", "properties": {}}
        
        tool = MCPTool(
            session=mock_session,
            tool_name="test_tool",
            tool_description="Test tool",
            tool_schema=tool_schema,
        )
        
        result = await tool.async_forward(query="test query")
        
        assert result.output == "tool result"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_async_forward_with_list_content(self, mock_dependencies):
        """Test MCPTool async_forward with list content."""
        from agents.math_agent.environment.tools.mcp_tool import MCPTool
        
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_content_item = MagicMock()
        mock_content_item.text = "list item result"
        mock_result.content = [mock_content_item]
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        tool_schema = {"type": "object", "properties": {}}
        
        tool = MCPTool(
            session=mock_session,
            tool_name="test_tool",
            tool_description="Test tool",
            tool_schema=tool_schema,
        )
        
        result = await tool.async_forward(query="test")
        
        assert result.output == "list item result"

    @pytest.mark.asyncio
    async def test_async_forward_with_str_content(self, mock_dependencies):
        """Test MCPTool async_forward with string content."""
        from agents.math_agent.environment.tools.mcp_tool import MCPTool
        
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = "raw string content"
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        tool_schema = {"type": "object", "properties": {}}
        
        tool = MCPTool(
            session=mock_session,
            tool_name="test_tool",
            tool_description="Test tool",
            tool_schema=tool_schema,
        )
        
        result = await tool.async_forward(query="test")
        
        assert result.output == "raw string content"

    @pytest.mark.asyncio
    async def test_async_forward_with_error(self, mock_dependencies):
        """Test MCPTool async_forward with error."""
        from agents.math_agent.environment.tools.mcp_tool import MCPTool
        
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=Exception("Connection error"))
        
        tool_schema = {"type": "object", "properties": {}}
        
        tool = MCPTool(
            session=mock_session,
            tool_name="test_tool",
            tool_description="Test tool",
            tool_schema=tool_schema,
        )
        
        result = await tool.async_forward(query="test")
        
        assert result.error is not None
        assert "Connection error" in result.error

    @pytest.mark.asyncio
    async def test_async_forward_with_no_content_attr(self, mock_dependencies):
        """Test MCPTool async_forward when result has no content attribute."""
        from agents.math_agent.environment.tools.mcp_tool import MCPTool
        
        mock_session = AsyncMock()
        mock_result = "plain string result"
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        
        tool_schema = {"type": "object", "properties": {}}
        
        tool = MCPTool(
            session=mock_session,
            tool_name="test_tool",
            tool_description="Test tool",
            tool_schema=tool_schema,
        )
        
        result = await tool.async_forward(query="test")
        
        assert result.output == "plain string result"
