#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

        http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import pytest


class TestAgenticEnvConfig:

    def test_default_config(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.agentic_env import AgenticEnvConfig
        config = AgenticEnvConfig({})
        assert config.namespace == "agentic_raygroup"
        assert config.rollout_output_path == "./outputs"
        assert config.agent_name == "netopt"
        assert config.max_steps == 5
        assert config.max_tool_length == 4096
        assert config.use_sse is False
        assert config.tool_timeout == 2000
        assert config.trajectory_timeout == 7200

    def test_custom_config(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.agentic_env import AgenticEnvConfig
        config_dict = {
            'namespace': 'test_namespace',
            'rollout_output_path': '/custom/output',
            'agent_name': 'test_agent',
            'max_steps': 10,
            'max_tool_length': 8192,
            'use_sse': True,
            'tool_timeout': 5000,
            'trajectory_timeout': 3600,
        }
        config = AgenticEnvConfig(config_dict)
        assert config.namespace == 'test_namespace'
        assert config.rollout_output_path == '/custom/output'
        assert config.agent_name == 'test_agent'
        assert config.max_steps == 10
        assert config.max_tool_length == 8192
        assert config.use_sse is True
        assert config.tool_timeout == 5000
        assert config.trajectory_timeout == 3600

    def test_mcp_server_config(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.agentic_env import AgenticEnvConfig
        config_dict = {
            'mcp_server_url': 'http://localhost:8080',
            'mcp_server_command': 'python',
            'mcp_server_args': ['-m', 'mcp.server'],
            'mcp_server_env': {'ENV': 'test'},
        }
        config = AgenticEnvConfig(config_dict)
        assert config.mcp_server_url == 'http://localhost:8080'
        assert config.mcp_server_command == 'python'
        assert config.mcp_server_args == ['-m', 'mcp.server']
        assert config.mcp_server_env == {'ENV': 'test'}
