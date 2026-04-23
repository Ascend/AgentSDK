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

import os
import pytest
from unittest.mock import patch

from agentic_rl.base.conf.conf import AgenticRLConf


class TestAgenticRLConf:
    """
    Tests for AgenticRLConf class.

    Covers:
      - Configuration loading from string and environment variable
      - Configuration filtering based on whitelist keys
      - Various input scenarios (empty, None, nested structures)
    """

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """
        Save and restore environment variable around each test.
        """
        original_env = os.environ.get(AgenticRLConf.CONF_ENV)
        yield
        if original_env is not None:
            os.environ[AgenticRLConf.CONF_ENV] = original_env
        elif AgenticRLConf.CONF_ENV in os.environ:
            del os.environ[AgenticRLConf.CONF_ENV]

    @pytest.mark.parametrize("conf_input", [None, ""])
    @patch('agentic_rl.base.conf.conf.logger')
    def test_load_config_with_empty_inputs(self, mock_logger, conf_input):
        """
        Test with None input and empty string input.
        """
        # Ensure env var is not set for these tests
        if AgenticRLConf.CONF_ENV in os.environ:
            del os.environ[AgenticRLConf.CONF_ENV]
        
        conf = AgenticRLConf.load_config(conf_input)
        assert dict(conf) == {}
        mock_logger.warning.assert_called_once()

    @patch('agentic_rl.base.conf.conf.logger')
    def test_load_config_with_valid_conf_string(self, mock_logger):
        """
        Test with valid configuration string that includes both whitelisted and non-whitelisted keys.
        """
        conf_str = '''{
            "agentic_ai": {
                "model": "test_model",
                "api_key": "test_key"
            },
            "serve_conf": {
                "port": 8080,
                "host": "localhost"
            },
            "non_whitelisted_key": "should_be_ignored"
        }'''
        
        conf = AgenticRLConf.load_config(conf_str)
        expected = {
            "agentic_ai": {
                "model": "test_model",
                "api_key": "test_key"
            },
            "serve_conf": {
                "port": 8080,
                "host": "localhost"
            }
        }
        assert dict(conf) == expected
        # Verify dot notation access works
        assert conf.agentic_ai.model == "test_model"
        assert conf.serve_conf.port == 8080
        mock_logger.debug.assert_called_once()

    @patch('agentic_rl.base.conf.conf.logger')
    def test_load_config_with_all_whitelisted_keys(self, mock_logger):
        """
        Test with all whitelisted keys included in the configuration.
        """
        conf_str = '''{
            "agentic_ai": {},
            "serve_conf": {},
            "direct_conf": {},
            "train_instances": [],
            "agent_instances": [],
            "infer_instances": []
        }'''
        
        conf = AgenticRLConf.load_config(conf_str)
        assert set(conf.keys()) == AgenticRLConf.WHITELIST_KEYS
        mock_logger.debug.assert_called_once()

    @patch('agentic_rl.base.conf.conf.logger')
    def test_load_config_with_env_variable(self, mock_logger):
        """
        Test loading configuration from environment variable.
        """
        conf_str = '''{
            "agentic_ai": {
                "model": "env_model"
            }
        }'''
        os.environ[AgenticRLConf.CONF_ENV] = conf_str
        
        conf = AgenticRLConf.load_config()
        assert conf.agentic_ai.model == "env_model"
        mock_logger.debug.assert_called_once()

    @patch('agentic_rl.base.conf.conf.logger')
    def test_load_config_with_nested_configurations(self, mock_logger):
        """
        Test with deeply nested configurations to ensure dot notation works at all levels.
        """
        conf_str = '''{
            "agentic_ai": {
                "model": {
                    "name": "nested_model",
                    "version": "1.0",
                    "params": {
                        "param1": "value1",
                        "param2": 100
                    }
                },
                "api": {
                    "url": "http://example.com/api"
                }
            }
        }'''
        
        conf = AgenticRLConf.load_config(conf_str)
        assert conf.agentic_ai.model.name == "nested_model"
        assert conf.agentic_ai.model.params.param1 == "value1"
        assert conf.agentic_ai.api.url == "http://example.com/api"
        mock_logger.debug.assert_called_once()