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

import sys
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Module-level mocks  (BEFORE importing the code under test)
#
# data_registry.py imports:
#   - InferDataManager  (from infer_data -> Loggers, RolloutClient + transitive deps)
#   - MindSpeedRLDataManager (from mindspeed_rl_data -> ray, Loggers, data_transform)
#   - VerlDataManager (from verl_data -> torch, numpy, verl)
# ---------------------------------------------------------------------------
mock_ray = MagicMock()
mock_torch = MagicMock()
mock_np = MagicMock()
mock_verl = MagicMock()
mock_requests = MagicMock()

mock_loggers_module = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = MagicMock()

mock_base_utils_module = MagicMock()
mock_base_utils_module.singleton = lambda cls: cls

mock_rollout_queue_module = MagicMock()
mock_rollout_client_module = MagicMock()
mock_controller_config_module = MagicMock()
mock_http_status_module = MagicMock()

mock_controller_utils_module = MagicMock()
mock_controller_utils_module.DEFAULT_SLEEP_TIME = 2
mock_controller_utils_module.READ_TIMEOUT = 600
mock_controller_utils_module.DEFAULT_URL_METHOD = "http"
mock_controller_utils_module.MAX_TIMEOUT = 1800

mock_data_transform_module = MagicMock()

with patch.dict(sys.modules, {
    'ray': mock_ray,
    'torch': mock_torch,
    'torch.distributed': mock_torch.distributed,
    'numpy': mock_np,
    'verl': mock_verl,
    'requests': mock_requests,
    'agentic_rl.base.log.loggers': mock_loggers_module,
    'agentic_rl.base.utils.utils': mock_base_utils_module,
    'agentic_rl.controllers.rollout_controller.rollout_queue': mock_rollout_queue_module,
    'agentic_rl.controllers.rollout_controller.rollout_client': mock_rollout_client_module,
    'agentic_rl.controllers.utils.controller_config': mock_controller_config_module,
    'agentic_rl.controllers.utils.http_status': mock_http_status_module,
    'agentic_rl.controllers.utils.utils': mock_controller_utils_module,
    'agentic_rl.data_manager.data_transform': mock_data_transform_module,
}):
    from agentic_rl.data_manager.data_registry import (
        DataManagerRegistry,
        registry,
        data_manager_class,
    )
    from agentic_rl.data_manager.infer_data import InferDataManager
    from agentic_rl.data_manager.mindspeed_rl_data import MindSpeedRLDataManager
    from agentic_rl.data_manager.verl_data import VerlDataManager


class TestDataManagerRegistry(unittest.TestCase):
    """Tests for DataManagerRegistry, module-level registry, and data_manager_class."""

    # ---- DataManagerRegistry basics ----------------------------------------

    def test_registry_init_empty(self):
        """A fresh registry has an empty internal dict."""
        r = DataManagerRegistry()
        self.assertEqual(r._registry, {})

    def test_register_creates_backend_entry(self):
        """register creates a nested dict for a new backend."""
        r = DataManagerRegistry()
        r.register("my_backend", "train", MagicMock)
        self.assertIn("my_backend", r._registry)
        self.assertIn("train", r._registry["my_backend"])

    def test_get_class_returns_registered_class(self):
        """get_class returns the class registered for a backend + mode."""
        r = DataManagerRegistry()
        sentinel = MagicMock
        r.register("b", "m", sentinel)
        self.assertIs(r.get_class("b", "m"), sentinel)

    # ---- module-level registry pre-populated entries -----------------------

    def test_registry_mindspeed_rl_train(self):
        """Module-level registry maps mindspeed_rl/train to MindSpeedRLDataManager."""
        cls = registry.get_class("mindspeed_rl", "train")
        self.assertIs(cls, MindSpeedRLDataManager)

    def test_registry_mindspeed_rl_infer(self):
        """Module-level registry maps mindspeed_rl/infer to InferDataManager."""
        cls = registry.get_class("mindspeed_rl", "infer")
        self.assertIs(cls, InferDataManager)

    def test_registry_verl_train(self):
        """Module-level registry maps verl/train to VerlDataManager."""
        cls = registry.get_class("verl", "train")
        self.assertIs(cls, VerlDataManager)

    def test_registry_verl_infer(self):
        """Module-level registry maps verl/infer to InferDataManager."""
        cls = registry.get_class("verl", "infer")
        self.assertIs(cls, InferDataManager)

    # ---- data_manager_class helper -----------------------------------------

    def test_data_manager_class_function(self):
        """data_manager_class returns the correct class via the module-level registry."""
        cls = data_manager_class("verl", "train")
        self.assertIs(cls, VerlDataManager)

    # ---- unknown backend ---------------------------------------------------

    def test_get_class_unknown_backend_raises(self):
        """get_class raises AttributeError for an unregistered backend (returns None)."""
        with self.assertRaises(AttributeError):
            registry.get_class("unknown_backend", "train")


if __name__ == '__main__':
    unittest.main()
