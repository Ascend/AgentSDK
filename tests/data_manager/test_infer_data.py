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
# ---------------------------------------------------------------------------
mock_torch = MagicMock()

mock_loggers_module = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = MagicMock()

mock_rollout_client_module = MagicMock()

# RolloutClient's transitive dependencies also need mocking:
#   rollout_client.py imports: io, time, ray, requests, torch,
#     agentic_rl.base.log.loggers, agentic_rl.base.utils.utils (singleton),
#     agentic_rl.controllers.rollout_controller.rollout_queue,
#     agentic_rl.controllers.utils.controller_config,
#     agentic_rl.controllers.utils.http_status,
#     agentic_rl.controllers.utils.utils

mock_base_utils_module = MagicMock()
mock_base_utils_module.singleton = lambda cls: cls  # passthrough decorator

mock_rollout_queue_module = MagicMock()
mock_controller_config_module = MagicMock()
mock_http_status_module = MagicMock()
mock_controller_utils_module = MagicMock()
mock_controller_utils_module.DEFAULT_SLEEP_TIME = 2
mock_controller_utils_module.READ_TIMEOUT = 600
mock_controller_utils_module.DEFAULT_URL_METHOD = "http"
mock_controller_utils_module.MAX_TIMEOUT = 1800

mock_ray = MagicMock()
mock_requests = MagicMock()

with patch.dict(sys.modules, {
    'torch': mock_torch,
    'torch.distributed': mock_torch.distributed,
    'ray': mock_ray,
    'requests': mock_requests,
    'agentic_rl.base.log.loggers': mock_loggers_module,
    'agentic_rl.base.utils.utils': mock_base_utils_module,
    'agentic_rl.controllers.rollout_controller.rollout_queue': mock_rollout_queue_module,
    'agentic_rl.controllers.rollout_controller.rollout_client': mock_rollout_client_module,
    'agentic_rl.controllers.utils.controller_config': mock_controller_config_module,
    'agentic_rl.controllers.utils.http_status': mock_http_status_module,
    'agentic_rl.controllers.utils.utils': mock_controller_utils_module,
}):
    from agentic_rl.data_manager.infer_data import InferDataManager


class TestInferDataManager(unittest.TestCase):
    """Tests for InferDataManager covering all public methods."""

    def setUp(self):
        self.dm = InferDataManager()

    def tearDown(self):
        mock_torch.reset_mock()
        mock_loggers_module.reset_mock()
        mock_rollout_client_module.reset_mock()

    # ---- __init__ -----------------------------------------------------------

    def test_init_creates_empty_deque(self):
        """data_manager should be an empty deque after init."""
        self.assertEqual(len(self.dm.data_manager), 0)

    # ---- all_consumed -------------------------------------------------------

    def test_all_consumed_empty(self):
        """all_consumed returns 0 when queue is empty."""
        result = self.dm.all_consumed("train")
        self.assertEqual(result, 0)

    def test_all_consumed_non_empty(self):
        """all_consumed returns the number of items in the deque."""
        self.dm.data_manager.append(("batch1", [0]))
        self.dm.data_manager.append(("batch2", [1]))
        result = self.dm.all_consumed("train")
        self.assertEqual(result, 2)

    # ---- put_experience / get_data ------------------------------------------

    def test_put_experience_adds_to_deque(self):
        """put_experience appends (batch_dict, indexes) tuple to the deque."""
        self.dm.put_experience({"ids": [1, 2]}, [0, 1])
        self.assertEqual(len(self.dm.data_manager), 1)
        item = self.dm.data_manager[0]
        self.assertEqual(item, ({"ids": [1, 2]}, [0, 1]))

    def test_get_data_returns_batch_when_index_truthy(self):
        """get_data pops and returns batch_data when index is truthy."""
        batch = MagicMock()
        batch.keys.return_value = ["input_ids"]
        self.dm.data_manager.append((batch, [0, 1]))

        result_batch, result_index = self.dm.get_data("train", None, 16)
        self.assertIs(result_batch, batch)
        self.assertEqual(result_index, [0, 1])

    def test_get_data_returns_empty_when_index_falsy(self):
        """get_data returns ({}, []) when popped index is empty."""
        batch = MagicMock()
        batch.keys.return_value = ["input_ids"]
        self.dm.data_manager.append((batch, []))

        result_batch, result_index = self.dm.get_data("train", None, 16)
        self.assertEqual(result_batch, {})
        self.assertEqual(result_index, [])

    # ---- put_data -----------------------------------------------------------

    def test_put_data_creates_rollout_client(self):
        """put_data instantiates RolloutClient and calls send_outputs_to_train_server."""
        mock_client_instance = MagicMock()
        mock_rollout_client_module.RolloutClient.return_value = mock_client_instance

        output = {"result": [1]}
        self.dm.put_data(output, [0], metric={"loss": 0.1})

        mock_rollout_client_module.RolloutClient.assert_called_once()
        mock_client_instance.send_outputs_to_train_server.assert_called_once_with(output, {"loss": 0.1})

    # ---- update_metrics -----------------------------------------------------

    def test_update_metrics_is_noop(self):
        """update_metrics does nothing (pass)."""
        # Should not raise
        self.dm.update_metrics("loss", 0.5, True)


if __name__ == '__main__':
    unittest.main()
