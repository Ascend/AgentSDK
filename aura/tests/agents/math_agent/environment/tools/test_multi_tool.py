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


@pytest.fixture(autouse=True, scope="function")
def mock_dependencies():
    """Mock all external dependencies for multi_tool tests."""
    yield {}


class TestMultiTool:
    """Tests for MultiTool class."""

    def test_init_with_tools_list(self, mock_dependencies):
        """Test MultiTool initialization with tools list."""
        from agents.math_agent.environment.tools.multi_tool import MultiTool
        
        tool = MultiTool(tools=["python"])
        
        assert "python" in tool.tools

    def test_init_with_tool_map(self, mock_dependencies):
        """Test MultiTool initialization with tool_map."""
        from agents.math_agent.environment.tools.multi_tool import MultiTool
        
        mock_tool_class = MagicMock()
        mock_tool_instance = MagicMock()
        mock_tool_instance.json = {"type": "function", "function": {"name": "custom_tool"}}
        mock_tool_class.return_value = mock_tool_instance
        
        tool = MultiTool(tool_map={"custom_tool": mock_tool_class})
        
        assert "custom_tool" in tool.tools
        assert "custom_tool" in tool.tool_map

    def test_init_raises_error_when_both_params_provided(self, mock_dependencies):
        """Test MultiTool raises error when both tools and tool_map provided."""
        from agents.math_agent.environment.tools.multi_tool import MultiTool
        
        mock_tool_class = MagicMock()
        
        with pytest.raises(ValueError, match="Cannot specify both"):
            MultiTool(tools=["python"], tool_map={"custom": mock_tool_class})

    def test_init_with_empty_params(self, mock_dependencies):
        """Test MultiTool initialization with no params."""
        from agents.math_agent.environment.tools.multi_tool import MultiTool
        
        tool = MultiTool()
        
        assert tool.tools == []
        assert tool.tool_map == {}

    def test_json_property(self, mock_dependencies):
        """Test MultiTool json property."""
        from agents.math_agent.environment.tools.multi_tool import MultiTool
        
        mock_tool_class = MagicMock()
        mock_tool_instance = MagicMock()
        mock_tool_instance.json = {"type": "function", "function": {"name": "test_tool"}}
        mock_tool_class.return_value = mock_tool_instance
        
        tool = MultiTool(tool_map={"test_tool": mock_tool_class})
        result = tool.json
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "test_tool"

    def test_json_property_empty(self, mock_dependencies):
        """Test MultiTool json property with no tools."""
        from agents.math_agent.environment.tools.multi_tool import MultiTool
        
        tool = MultiTool()
        result = tool.json
        
        assert result == []

    def test_forward_existing_tool(self, mock_dependencies):
        """Test MultiTool forward with existing tool."""
        from agents.math_agent.environment.tools.multi_tool import MultiTool
        
        mock_tool_instance = MagicMock()
        mock_tool_instance.return_value = MagicMock(output="result", to_string=lambda: "result")
        mock_tool_class = MagicMock(return_value=mock_tool_instance)
        
        tool = MultiTool(tool_map={"test_tool": mock_tool_class})
        result = tool(tool_name="test_tool", arg1="value1")
        
        mock_tool_instance.assert_called_once()

    def test_forward_nonexistent_tool(self, mock_dependencies):
        """Test MultiTool forward with nonexistent tool."""
        from agents.math_agent.environment.tools.multi_tool import MultiTool
        
        mock_tool_class = MagicMock()
        mock_tool_instance = MagicMock()
        mock_tool_instance.json = {"type": "function", "function": {"name": "existing_tool"}}
        mock_tool_class.return_value = mock_tool_instance
        
        tool = MultiTool(tool_map={"existing_tool": mock_tool_class})
        result = tool(tool_name="nonexistent_tool")
        
        assert "not found" in result.output

    def test_init_with_registry_tools(self, mock_dependencies):
        """Test MultiTool initialization with tools from registry."""
        from agents.math_agent.environment.tools.multi_tool import MultiTool
        
        tool = MultiTool(tools=["python"])
        
        assert "python" in tool.tools

    def test_tool_map_instantiation(self, mock_dependencies):
        """Test that tool_map tools are instantiated correctly."""
        from agents.math_agent.environment.tools.multi_tool import MultiTool
        
        mock_tool_class = MagicMock()
        mock_tool_instance = MagicMock()
        mock_tool_instance.json = {"type": "function", "function": {"name": "my_tool"}}
        mock_tool_class.return_value = mock_tool_instance
        
        tool = MultiTool(tool_map={"my_tool": mock_tool_class})
        
        mock_tool_class.assert_called_once_with(name="my_tool")
