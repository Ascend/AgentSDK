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

from aura.base.conf.conf import AgenticRLConf


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

    @patch('aura.base.conf.conf.logger')
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
