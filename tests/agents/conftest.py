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
from unittest.mock import MagicMock, patch
import sys

@pytest.fixture(autouse=True, scope="function")
def mock_rllm_dependencies():
    """Setup mock for rllm module as a pytest fixture."""
    
    mock_rllm = MagicMock()

    # Mock parser hierarchy
    mock_rllm.parser = MagicMock()
    mock_rllm.parser.tool_parser = MagicMock()
    mock_rllm.parser.tool_parser.tool_parser_base = MagicMock()
    mock_rllm.parser.tool_parser.tool_parser_base.ToolParser = object
    
    # Mock ToolCall class
    class MockToolCall:
        def __init__(self, name="", arguments=None):
            self.name = name
            self.arguments = arguments if arguments is not None else {}
    
    # Mock Tool class
    class MockTool:
        def __init__(self, name="", description=""):
            self.name = name
            self.description = description
        
        def forward(self, *args, **kwargs):
            return MockToolOutput(name=self.name, output="success")
        
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
    
    # Mock ToolOutput class
    class MockToolOutput:
        def __init__(self, name="", output=None, error=None):
            self.name = name
            self.output = output
            self.error = error
    
    # Setup tools hierarchy
    mock_rllm.tools = MagicMock()
    mock_rllm.tools.tool_base = MagicMock()
    mock_rllm.tools.tool_base.Tool = MockTool
    mock_rllm.tools.tool_base.ToolCall = MockToolCall
    mock_rllm.tools.tool_base.ToolOutput = MockToolOutput
    
    # Mock MultiTool class
    class MockMultiTool(MockTool):
        def __init__(self, tools=None, tool_map=None):
            super().__init__()
            if tool_map is not None and tools is not None:
                raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters")
            
            if tool_map is not None:
                self.tools = list(tool_map.keys())
                self.tool_map = {}
                for name, tool_cls in tool_map.items():
                    self.tool_map[name] = tool_cls(name=name)
            elif tools is not None:
                self.tools = tools
                self.tool_map = {tool: MockTool(name=tool) for tool in tools}
            else:
                self.tools = []
                self.tool_map = {}
        
        @property
        def json(self):
            return [{"type": "function", "function": {"name": name}} for name in self.tools]
        
        def forward(self, tool_name, *args, **kwargs):
            if tool_name not in self.tool_map:
                return MockToolOutput(name=tool_name, output=f"Tool {tool_name} not found")
            return MockToolOutput(name=tool_name, output="success")
    
    mock_rllm.tools.multi_tool = MagicMock()
    mock_rllm.tools.multi_tool.MultiTool = MockMultiTool
    
    # Mock ToolRegistry
    class MockToolRegistry:
        def __init__(self):
            self._tools = {"python", "calculator", "search"}
        
        def __contains__(self, tool_name):
            return tool_name in self._tools
        
        def instantiate(self, tool_name):
            if tool_name in self._tools:
                return MockTool(name=tool_name)
            return None
    
    mock_rllm.tools.tool_registry = MockToolRegistry()
    mock_rllm.tools.utils = MagicMock()
    mock_rllm.tools.utils.function_to_dict = MagicMock(return_value={
        "type": "function",
        "function": {
            "name": "test_function",
            "description": "A test function",
            "parameters": {"type": "object", "properties": {}},
        },
    })

    # Mock rewards hierarchy
    mock_rllm.rewards = MagicMock()
    mock_rllm.rewards.code_utils = MagicMock()
    mock_rllm.rewards.code_utils.firejail_exec = MagicMock()
    mock_rllm.rewards.code_utils.firejail_exec.code_exec_firejail = MagicMock()
    mock_rllm.rewards.code_utils.humanevalplus = MagicMock()
    mock_rllm.rewards.code_utils.humanevalplus.get_num_test_cases = MagicMock(return_value=5)
    mock_rllm.rewards.code_utils.humanevalplus.run_test = MagicMock(return_value=[True])
    mock_rllm.rewards.code_utils.kodcode = MagicMock()
    mock_rllm.rewards.code_utils.kodcode.code_exec = MagicMock(return_value=(True, {}))
    mock_rllm.rewards.code_utils.livecodebench = MagicMock()
    mock_rllm.rewards.code_utils.livecodebench.run_test = MagicMock(return_value=[True])
    mock_rllm.rewards.code_utils.taco = MagicMock()
    mock_rllm.rewards.code_utils.taco.run_test = MagicMock(return_value=[True])
    
    mock_rllm.rewards.math_utils = MagicMock()
    mock_rllm.rewards.math_utils.utils = MagicMock()
    mock_rllm.rewards.math_utils.utils.extract_answer = MagicMock(return_value="42")
    mock_rllm.rewards.math_utils.utils.grade_answer_mathd = MagicMock(return_value=False)
    mock_rllm.rewards.math_utils.utils.grade_answer_sympy = MagicMock(return_value=False)

    # Mock code tools
    mock_rllm.tools.code_tools = MagicMock()
    mock_rllm.tools.code_tools.code_tool = MagicMock()
    mock_rllm.tools.code_tools.code_tool.CodeTool = MagicMock()
    mock_rllm.tools.code_tools.together_tool = MagicMock()
    mock_rllm.tools.code_tools.together_tool.TogetherCodeTool = MagicMock()

    # Mock globals and utils
    mock_rllm.globals = MagicMock()
    mock_rllm.globals.OAI_RM_MODEL = "test-model"
    mock_rllm.globals.THOUGHT_DELIMITER_END = "</think"
    
    mock_rllm.utils = MagicMock()
    mock_rllm.utils.call_gemini_llm = MagicMock(return_value="[[NO]]")
    mock_rllm.utils.call_oai_rm_llm = MagicMock(return_value="[[NO]]")
    
    mock_rllm.system_prompts = MagicMock()
    mock_rllm.system_prompts.ORM_PROMPT = "test prompt"

    # Store all mocked modules
    modules_to_mock = {
        "rllm": mock_rllm,
        "rllm.parser": mock_rllm.parser,
        "rllm.parser.tool_parser": mock_rllm.parser.tool_parser,
        "rllm.parser.tool_parser.tool_parser_base": mock_rllm.parser.tool_parser.tool_parser_base,
        "rllm.tools": mock_rllm.tools,
        "rllm.tools.tool_base": mock_rllm.tools.tool_base,
        "rllm.tools.multi_tool": mock_rllm.tools.multi_tool,
        "rllm.tools.tool_registry": mock_rllm.tools.tool_registry,
        "rllm.tools.utils": mock_rllm.tools.utils,
        "rllm.tools.code_tools": mock_rllm.tools.code_tools,
        "rllm.tools.code_tools.code_tool": mock_rllm.tools.code_tools.code_tool,
        "rllm.tools.code_tools.together_tool": mock_rllm.tools.code_tools.together_tool,
        "rllm.rewards": mock_rllm.rewards,
        "rllm.rewards.code_utils": mock_rllm.rewards.code_utils,
        "rllm.rewards.code_utils.firejail_exec": mock_rllm.rewards.code_utils.firejail_exec,
        "rllm.rewards.code_utils.humanevalplus": mock_rllm.rewards.code_utils.humanevalplus,
        "rllm.rewards.code_utils.kodcode": mock_rllm.rewards.code_utils.kodcode,
        "rllm.rewards.code_utils.livecodebench": mock_rllm.rewards.code_utils.livecodebench,
        "rllm.rewards.code_utils.taco": mock_rllm.rewards.code_utils.taco,
        "rllm.rewards.math_utils": mock_rllm.rewards.math_utils,
        "rllm.rewards.math_utils.utils": mock_rllm.rewards.math_utils.utils,
        "rllm.globals": mock_rllm.globals,
        "rllm.utils": mock_rllm.utils,
        "rllm.system_prompts": mock_rllm.system_prompts,
    }
    
    # Save original modules
    original_modules = {}
    for module_name, mock_module in modules_to_mock.items():
        original_modules[module_name] = sys.modules.get(module_name)
        sys.modules[module_name] = mock_module
    
    yield mock_rllm
    
    # Restore original modules
    for module_name, original in original_modules.items():
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original


@pytest.fixture(autouse=True, scope="function")
def mock_agentic_rl_dependencies():
    """Mock agentic_rl and its submodules for testing"""
    
    # 创建主要的 mock 对象
    mock_agentic_rl = MagicMock()
    
    # 构建嵌套结构
    # agentic_rl.runner.agent_engine_wrapper.base.agent.base_agent
    mock_base_agent = MagicMock()
    mock_base_agent.Trajectory = MagicMock()
    mock_base_agent.Step = MagicMock()
    mock_base_agent.Action = MagicMock()
    mock_base_agent.BaseAgent = MagicMock()
    
    mock_agent = MagicMock()
    mock_agent.base_agent = mock_base_agent
    
    mock_base = MagicMock()
    mock_base.agent = mock_agent
    
    mock_agent_engine_wrapper = MagicMock()
    mock_agent_engine_wrapper.base = mock_base
    
    mock_runner = MagicMock()
    mock_runner.agent_engine_wrapper = mock_agent_engine_wrapper
    
    mock_agentic_rl.runner = mock_runner
    
    # Mock BaseEnv class
    class MockBaseEnv:
        def __init__(self):
            self.step_count = 0
            self.task = None
        
        def reset(self):
            self.step_count = 0
            return self.task
        
        def step(self, action):
            self.step_count += 1
            return None, 0, False, {}
    
    # Create mock base_env module with BaseEnv attribute
    mock_base_env_module = MagicMock()
    mock_base_env_module.BaseEnv = MockBaseEnv
    
    # Create mock env_utils module with compute_trajectory_reward attribute
    mock_env_utils_module = MagicMock()
    mock_env_utils_module.compute_trajectory_reward = MagicMock(return_value=1.0)
    
    mock_environment = MagicMock()
    mock_environment.base_env = mock_base_env_module
    mock_environment.env_utils = mock_env_utils_module
    
    mock_base.environment = mock_environment
    
    # 使用 patch.dict 批量替换 sys.modules
    with patch.dict(
        sys.modules,
        {
            "agentic_rl": mock_agentic_rl,
            "agentic_rl.runner": mock_runner,
            "agentic_rl.runner.agent_engine_wrapper": mock_agent_engine_wrapper,
            "agentic_rl.runner.agent_engine_wrapper.base": mock_base,
            "agentic_rl.runner.agent_engine_wrapper.base.agent": mock_agent,
            "agentic_rl.runner.agent_engine_wrapper.base.agent.base_agent": mock_base_agent,
            "agentic_rl.runner.agent_engine_wrapper.base.environment": mock_environment,
            "agentic_rl.runner.agent_engine_wrapper.base.environment.base_env": mock_base_env_module,
            "agentic_rl.runner.agent_engine_wrapper.base.environment.env_utils": mock_env_utils_module,
        },
    ):
        yield


@pytest.fixture(autouse=True, scope="function")
def mock_ray_dependencies():
    """Setup mock for ray module as a pytest fixture."""
    
    mock_ray = MagicMock()
    
    def mock_remote(func_or_cls=None, **kwargs):
        if func_or_cls is not None:
            if isinstance(func_or_cls, type):
                original_cls = func_or_cls
                class MockRemoteClass:
                    def __init__(self, *args, **kwargs):
                        self._instance = original_cls(*args, **kwargs)
                    
                    def __getattr__(self, name):
                        return getattr(self._instance, name)
                
                MockRemoteClass.__name__ = original_cls.__name__
                MockRemoteClass.__module__ = original_cls.__module__
                return MockRemoteClass
            return func_or_cls
        def decorator(func_or_cls):
            return func_or_cls
        return decorator
    
    mock_ray.remote = mock_remote
    
    # 保存原始模块
    original_ray = sys.modules.get("ray")
    
    # 替换模块
    sys.modules["ray"] = mock_ray
    
    yield mock_ray
    
    # 恢复原始模块
    if original_ray is None:
        sys.modules.pop("ray", None)
    else:
        sys.modules["ray"] = original_ray
