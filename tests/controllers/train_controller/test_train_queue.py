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
# train_queue.py imports:
#   - agentic_rl.base.log.loggers.Loggers  (which imports torch, torch.distributed)
#   - agentic_rl.controllers.utils.utils.DEFAULT_SLEEP_TIME, MAX_TIMEOUT
#     (which imports ray, requests, torch, Loggers)
# ---------------------------------------------------------------------------
mock_torch = MagicMock()
mock_ray = MagicMock()
mock_requests = MagicMock()

mock_loggers_module = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = MagicMock()

mock_controller_utils_module = MagicMock()
mock_controller_utils_module.DEFAULT_SLEEP_TIME = 2
mock_controller_utils_module.MAX_TIMEOUT = 1800

with patch.dict(sys.modules, {
    'torch': mock_torch,
    'torch.distributed': mock_torch.distributed,
    'ray': mock_ray,
    'requests': mock_requests,
    'agentic_rl.base.log.loggers': mock_loggers_module,
    'agentic_rl.controllers.utils.utils': mock_controller_utils_module,
}):
    from agentic_rl.controllers.train_controller.train_queue import TrainQueue


class TestTrainQueue(unittest.TestCase):
    """Tests for TrainQueue covering init, add, pop, size, len, getitem."""

    def setUp(self):
        self.tq = TrainQueue(max_queue_size=2, global_batch_size=4, n_samples_per_prompt=1)

    def tearDown(self):
        mock_torch.reset_mock()
        mock_ray.reset_mock()
        mock_requests.reset_mock()
        mock_loggers_module.reset_mock()

    # ---- __init__ -----------------------------------------------------------

    def test_init_attributes(self):
        """Verify all attributes are set correctly on construction."""
        self.assertEqual(self.tq.n_samples_per_prompt, 1)
        self.assertEqual(self.tq.global_batch_size, 4)
        self.assertEqual(self.tq.max_queue_len, 2 * 4 * 1)  # 8
        self.assertEqual(self.tq.timeout, 1800)
        self.assertEqual(self.tq.sleep_time, 2)
        self.assertEqual(len(self.tq.queue), 0)
        self.assertEqual(len(self.tq.metric_queue), 0)

    def test_init_max_queue_len_calculation(self):
        """max_queue_len = max_queue_size * global_batch_size * n_samples_per_prompt."""
        tq = TrainQueue(max_queue_size=3, global_batch_size=8, n_samples_per_prompt=2)
        self.assertEqual(tq.max_queue_len, 3 * 8 * 2)

    # ---- add_minibatch ------------------------------------------------------

    def test_add_minibatch_returns_true_under_limit(self):
        """add_minibatch returns True when queue size <= max_queue_len."""
        result = self.tq.add_minibatch({"data": 1}, {"loss": 0.1})
        self.assertTrue(result)
        self.assertEqual(len(self.tq.queue), 1)
        self.assertEqual(len(self.tq.metric_queue), 1)

    def test_add_minibatch_returns_true_at_limit(self):
        """add_minibatch returns True when queue size exactly equals max_queue_len."""
        # max_queue_len = 8, so add exactly 8
        for i in range(8):
            result = self.tq.add_minibatch({"data": i}, {"loss": i})
        self.assertTrue(result)
        self.assertEqual(len(self.tq.queue), 8)

    def test_add_minibatch_returns_false_over_limit(self):
        """add_minibatch returns False when queue exceeds max_queue_len."""
        for i in range(8):
            self.tq.add_minibatch({"data": i}, {"loss": i})
        result = self.tq.add_minibatch({"data": 9}, {"loss": 9})
        self.assertFalse(result)
        self.assertEqual(len(self.tq.queue), 9)

    # ---- pop_batch ----------------------------------------------------------

    def test_pop_batch_returns_first_added(self):
        """pop_batch returns the first item (FIFO) and its metric."""
        self.tq.add_minibatch({"data": "first"}, {"m": 1})
        self.tq.add_minibatch({"data": "second"}, {"m": 2})

        out, metric = self.tq.pop_batch()
        self.assertEqual(out, {"data": "first"})
        self.assertEqual(metric, {"m": 1})
        self.assertEqual(len(self.tq.queue), 1)

    def test_pop_batch_empty_raises(self):
        """pop_batch on an empty queue raises IndexError."""
        with self.assertRaises(IndexError):
            self.tq.pop_batch()

    # ---- size ---------------------------------------------------------------

    def test_size_empty(self):
        """size returns 0 for empty queue."""
        self.assertEqual(self.tq.size(), 0)

    def test_size_after_additions(self):
        """size returns number of items added."""
        self.tq.add_minibatch("a", "m1")
        self.tq.add_minibatch("b", "m2")
        self.assertEqual(self.tq.size(), 2)

    # ---- get_max_len --------------------------------------------------------

    def test_get_max_len(self):
        """get_max_len returns the computed max_queue_len."""
        self.assertEqual(self.tq.get_max_len(), 8)

    # ---- __len__ ------------------------------------------------------------

    def test_len_empty(self):
        """len() returns 0 for empty queue."""
        self.assertEqual(len(self.tq), 0)

    def test_len_non_empty(self):
        """len() returns the number of items."""
        self.tq.add_minibatch("a", "m")
        self.assertEqual(len(self.tq), 1)

    # ---- __getitem__ --------------------------------------------------------

    def test_getitem_valid_index(self):
        """__getitem__ does not raise for valid index (note: no return value specified)."""
        self.tq.add_minibatch("data_0", "metric_0")
        self.tq.add_minibatch("data_1", "metric_1")
        # __getitem__ does not explicitly return a value in the source
        # but should not raise for valid indices
        try:
            self.tq[0]
            self.tq[1]
        except IndexError:
            self.fail("__getitem__ raised IndexError for a valid index")

    def test_getitem_negative_index_raises(self):
        """__getitem__ raises IndexError for negative index."""
        self.tq.add_minibatch("data", "metric")
        with self.assertRaises(IndexError):
            self.tq[-1]

    def test_getitem_out_of_bounds_raises(self):
        """__getitem__ raises IndexError for index >= len."""
        self.tq.add_minibatch("data", "metric")
        with self.assertRaises(IndexError):
            self.tq[1]


if __name__ == '__main__':
    unittest.main()
