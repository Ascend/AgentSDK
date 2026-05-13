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


class TestAgentsMapping:
    """Tests for agents_mapping module."""

    def test_agents_mapping_structure(self):
        """Test AGENTS_MAPPING has expected structure."""
        from agents.agents_mapping import AGENTS_MAPPING
        
        assert isinstance(AGENTS_MAPPING, list)
        assert len(AGENTS_MAPPING) > 0

    def test_agents_mapping_math_agent_exists(self):
        """Test that math agent exists in mapping."""
        from agents.agents_mapping import AGENTS_MAPPING
        
        math_agent = None
        for agent in AGENTS_MAPPING:
            if agent.get("name") == "math":
                math_agent = agent
                break
        
        assert math_agent is not None
        assert "env_class" in math_agent
        assert "agent_class" in math_agent
        assert "env_args" in math_agent
        assert "agent_args" in math_agent

    def test_agents_mapping_math_agent_config(self):
        """Test math agent configuration."""
        from agents.agents_mapping import AGENTS_MAPPING
        
        math_agent = None
        for agent in AGENTS_MAPPING:
            if agent.get("name") == "math":
                math_agent = agent
                break
        
        assert math_agent is not None
        env_args = math_agent.get("env_args", {})
        assert "tools" in env_args
        assert "reward_fn" in env_args
        assert "tool_timeout" in env_args
        assert "max_steps" in env_args

    def test_get_agent_by_name_found(self):
        """Test get_agent_by_name returns agent when found."""
        from agents.agents_mapping import get_agent_by_name
        
        result = get_agent_by_name("math")
        
        assert result is not None
        assert result.get("name") == "math"

    def test_get_agent_by_name_not_found(self):
        """Test get_agent_by_name returns None when not found."""
        from agents.agents_mapping import get_agent_by_name
        
        result = get_agent_by_name("nonexistent_agent")
        
        assert result is None

    def test_get_agent_by_name_empty_string(self):
        """Test get_agent_by_name with empty string."""
        from agents.agents_mapping import get_agent_by_name
        
        result = get_agent_by_name("")
        
        assert result is None

    def test_agents_mapping_has_compute_trajectory_reward_fn(self):
        """Test that math agent has compute_trajectory_reward_fn."""
        from agents.agents_mapping import AGENTS_MAPPING
        
        math_agent = None
        for agent in AGENTS_MAPPING:
            if agent.get("name") == "math":
                math_agent = agent
                break
        
        assert math_agent is not None
        assert "compute_trajectory_reward_fn" in math_agent
