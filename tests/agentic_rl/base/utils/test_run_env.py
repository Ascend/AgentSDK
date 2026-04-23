#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import os
import pytest
import unittest.mock as mock

# Mock ExtendedGenerateConfig before importing run_env
mock_extended_generate_config = mock.MagicMock()
mock_extended_generate_config_class = mock.MagicMock(return_value=mock_extended_generate_config)

# Use mock.patch to mock the entire module
mock_config_cls = mock.MagicMock()
mock_config_cls.ExtendedGenerateConfig = mock_extended_generate_config_class

# Insert mock module into sys.modules
import sys
sys.modules["agentic_rl.trainer.train_adapter.mindspeed_rl.config_cls"] = mock_config_cls

from agentic_rl.base.utils.run_env import (
    CURRENT_PATH,
    ROOT_PATH,
    THIRD_PARTY_PATH,
    MINDSPEED_RL_PATH,
    MINDSPEED_PATH,
    MINDSPEED_LLM_PATH,
    MEGATRON_PATH,
    VLLM_PATH,
    VLLM_ASCEND_PATH,
    RLLM__PATH,
    CONFIGS_PATH,
    get_thirty_path,
    load_runtime_env,
    get_runtime_env,
    get_vllm_version
)


class TestRunEnv:
    def test_path_constants(self):
        """Test that path constants are properly defined."""
        # Verify CURRENT_PATH is an existing directory
        assert os.path.exists(CURRENT_PATH)
        assert os.path.isdir(CURRENT_PATH)
        
        # Verify other paths are built based on CURRENT_PATH
        assert ROOT_PATH in THIRD_PARTY_PATH
        assert THIRD_PARTY_PATH in MINDSPEED_RL_PATH
        assert THIRD_PARTY_PATH in MINDSPEED_PATH
        assert THIRD_PARTY_PATH in MINDSPEED_LLM_PATH
        assert THIRD_PARTY_PATH in MEGATRON_PATH
        assert THIRD_PARTY_PATH in VLLM_PATH
        assert THIRD_PARTY_PATH in VLLM_ASCEND_PATH
        assert THIRD_PARTY_PATH in RLLM__PATH
        assert ROOT_PATH in CONFIGS_PATH

    def test_get_thirty_path(self):
        """Test get_thirty_path function."""
        run_env = get_thirty_path()
        
        # Verify return value structure
        assert isinstance(run_env, dict)
        assert "env_vars" in run_env
        assert isinstance(run_env["env_vars"], dict)
        assert "PYTHONPATH" in run_env["env_vars"]
        
        # Verify PYTHONPATH contains all necessary paths
        pythonpath = run_env["env_vars"]["PYTHONPATH"]
        assert MINDSPEED_RL_PATH in pythonpath
        assert MINDSPEED_PATH in pythonpath
        assert MINDSPEED_LLM_PATH in pythonpath
        assert MEGATRON_PATH in pythonpath
        assert VLLM_PATH in pythonpath
        assert VLLM_ASCEND_PATH in pythonpath
        assert RLLM__PATH in pythonpath
        assert THIRD_PARTY_PATH in pythonpath
        assert "$PYTHONPATH" in pythonpath

    @mock.patch("yaml.safe_load")
    @mock.patch("builtins.open")
    def test_load_runtime_env(self, mock_open, mock_safe_load):
        """Test load_runtime_env function."""
        # Mock yaml file content
        mock_yaml_content = {
            "env_vars": {
                "TASK_QUEUE_ENABLE": "2",
                "VLLM_VERSION": "v0.3.0"
            }
        }
        mock_safe_load.return_value = mock_yaml_content
        
        # Call function
        runtime_env = load_runtime_env()
        
        # Verify file opening and yaml loading
        mock_open.assert_called_once_with(CONFIGS_PATH + "/envs/runtime_env.yaml")
        mock_safe_load.assert_called_once()
        
        # Verify return value
        assert runtime_env == mock_yaml_content

    @mock.patch("agentic_rl.base.utils.run_env.load_runtime_env")
    @mock.patch("agentic_rl.base.utils.run_env.logger")
    @mock.patch("agentic_rl.base.utils.run_env.ExtendedGenerateConfig")
    def test_get_runtime_env_task_queue_disabled(self, mock_extended_config, mock_logger, mock_load_runtime_env):
        """Test get_runtime_env when TASK_QUEUE_ENABLE should be changed to 1."""
        # Mock runtime_env
        original_runtime_env = {
            "env_vars": {
                "TASK_QUEUE_ENABLE": "2",
                "VLLM_VERSION": "v0.3.0"
            }
        }
        
        # Create a copy so when the function modifies it, it won't affect our assertions
        mock_runtime_env = original_runtime_env.copy()
        mock_runtime_env["env_vars"] = original_runtime_env["env_vars"].copy()
        
        mock_load_runtime_env.return_value = mock_runtime_env
        
        # Mock ExtendedGenerateConfig
        mock_generate_config = mock.Mock()
        mock_generate_config.enforce_eager = False
        mock_generate_config.enable_sleep_mode = True
        mock_extended_config.return_value = mock_generate_config
        
        # Call function
        config = {"generate_config": {"enforce_eager": False, "enable_sleep_mode": True}}
        result = get_runtime_env(config)
        
        # Verify function calls
        mock_load_runtime_env.assert_called_once()
        # Assert logger calls with original values because logger is called before modification
        mock_logger.info.assert_any_call(f"ray init with runtime_env: {original_runtime_env}")
        mock_logger.info.assert_any_call("change TASK_QUEUE_ENABLE to 1 because enforce_eager is False")
        
        # Verify result
        assert result["env_vars"]["TASK_QUEUE_ENABLE"] == "1"

    @mock.patch("agentic_rl.base.utils.run_env.load_runtime_env")
    @mock.patch("agentic_rl.base.utils.run_env.logger")
    @mock.patch("agentic_rl.base.utils.run_env.ExtendedGenerateConfig")
    def test_get_runtime_env_task_queue_unchanged(self, mock_extended_config, mock_logger, mock_load_runtime_env):
        """Test get_runtime_env when TASK_QUEUE_ENABLE should remain unchanged."""
        # Mock runtime_env
        mock_runtime_env = {
            "env_vars": {
                "TASK_QUEUE_ENABLE": "2",
                "VLLM_VERSION": "v0.3.0"
            }
        }
        mock_load_runtime_env.return_value = mock_runtime_env
        
        # Mock ExtendedGenerateConfig
        mock_generate_config = mock.Mock()
        mock_generate_config.enforce_eager = True
        mock_generate_config.enable_sleep_mode = False
        mock_extended_config.return_value = mock_generate_config
        
        # Call function
        config = {"generate_config": {"enforce_eager": True, "enable_sleep_mode": False}}
        result = get_runtime_env(config)
        
        # Verify function calls
        mock_load_runtime_env.assert_called_once()
        mock_logger.info.assert_called_once_with(f"ray init with runtime_env: {mock_runtime_env}")
        
        # Verify result
        assert result["env_vars"]["TASK_QUEUE_ENABLE"] == "2"

    @mock.patch("agentic_rl.base.utils.run_env.load_runtime_env")
    @mock.patch("agentic_rl.base.utils.run_env.logger")
    @mock.patch("agentic_rl.base.utils.run_env.ExtendedGenerateConfig")
    def test_get_runtime_env_sleep_mode_error(self, mock_extended_config, mock_logger, mock_load_runtime_env):
        """Test get_runtime_env when enable_sleep_mode is false but enforce_eager is also false."""
        # Mock runtime_env
        mock_runtime_env = {
            "env_vars": {
                "TASK_QUEUE_ENABLE": "1",
                "VLLM_VERSION": "v0.3.0"
            }
        }
        mock_load_runtime_env.return_value = mock_runtime_env
        
        # Mock ExtendedGenerateConfig
        mock_generate_config = mock.Mock()
        mock_generate_config.enforce_eager = False
        mock_generate_config.enable_sleep_mode = False
        mock_extended_config.return_value = mock_generate_config
        
        # Calling function should raise TypeError because original code raises a string instead of an exception object
        config = {"generate_config": {"enforce_eager": False, "enable_sleep_mode": False}}
        with pytest.raises(TypeError) as excinfo:
            get_runtime_env(config)
        
        # Verify exception message contains expected content
        assert "exceptions must derive from BaseException" in str(excinfo.value)

    @mock.patch("agentic_rl.base.utils.run_env.load_runtime_env")
    def test_get_vllm_version(self, mock_load_runtime_env):
        """Test get_vllm_version function."""
        # Mock runtime_env
        mock_vllm_version = "v0.3.0"
        mock_runtime_env = {
            "env_vars": {
                "TASK_QUEUE_ENABLE": "2",
                "VLLM_VERSION": mock_vllm_version
            }
        }
        mock_load_runtime_env.return_value = mock_runtime_env
        
        # Call function
        vllm_version = get_vllm_version()
        
        # Verify function call and return value
        mock_load_runtime_env.assert_called_once()
        assert vllm_version == mock_vllm_version