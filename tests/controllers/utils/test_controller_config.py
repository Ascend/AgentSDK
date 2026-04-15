#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
import unittest
from unittest.mock import patch

from agentic_rl.controllers.utils.controller_config import ControllerConfig


class TestControllerConfig(unittest.TestCase):
    """Tests for ControllerConfig class."""

    def test_default_values(self):
        """Verify default configuration when no environment variables are set."""
        with patch.dict(os.environ, {}, clear=True):
            config = ControllerConfig()
            self.assertEqual(config.controller_base_port, 4001)
            self.assertEqual(config.rollout_server_addr, "0.0.0.0:4001")
            self.assertEqual(config.train_server_addr, "0.0.0.0:4002")

    def test_custom_port(self):
        """Verify custom CONTROLLER_BASE_PORT overrides default port."""
        with patch.dict(os.environ, {"CONTROLLER_BASE_PORT": "5000"}, clear=True):
            config = ControllerConfig()
            self.assertEqual(config.controller_base_port, 5000)
            self.assertEqual(config.rollout_server_addr, "0.0.0.0:5000")
            self.assertEqual(config.train_server_addr, "0.0.0.0:5001")

    def test_custom_rollout_node(self):
        """Verify custom ROLLOUT_NODE overrides default rollout address."""
        with patch.dict(os.environ, {"ROLLOUT_NODE": "192.168.1.10"}, clear=True):
            config = ControllerConfig()
            self.assertEqual(config.controller_base_port, 4001)
            self.assertEqual(config.rollout_server_addr, "192.168.1.10:4001")
            self.assertEqual(config.train_server_addr, "0.0.0.0:4002")

    def test_custom_train_node(self):
        """Verify custom TRAIN_NODE overrides default train address."""
        with patch.dict(os.environ, {"TRAIN_NODE": "192.168.1.20"}, clear=True):
            config = ControllerConfig()
            self.assertEqual(config.controller_base_port, 4001)
            self.assertEqual(config.rollout_server_addr, "0.0.0.0:4001")
            self.assertEqual(config.train_server_addr, "192.168.1.20:4002")

    def test_all_custom(self):
        """Verify all environment variables override defaults simultaneously."""
        env_vars = {
            "CONTROLLER_BASE_PORT": "8000",
            "ROLLOUT_NODE": "10.0.0.1",
            "TRAIN_NODE": "10.0.0.2",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = ControllerConfig()
            self.assertEqual(config.controller_base_port, 8000)
            self.assertEqual(config.rollout_server_addr, "10.0.0.1:8000")
            self.assertEqual(config.train_server_addr, "10.0.0.2:8001")


if __name__ == '__main__':
    unittest.main()
