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
# train_weight_updater.py imports:
#   - time, dataclasses (stdlib -- no mock needed)
#   - ray  (used as @ray.remote decorator on WeightUpdateActor)
#   - agentic_rl.base.log.loggers.Loggers  (which imports torch, torch.distributed)
# ---------------------------------------------------------------------------
mock_ray = MagicMock()
mock_torch = MagicMock()

# Make @ray.remote a passthrough decorator so the class is importable
mock_ray.remote = MagicMock(side_effect=lambda cls: cls)

mock_loggers_module = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = MagicMock()

with patch.dict(sys.modules, {
    'ray': mock_ray,
    'torch': mock_torch,
    'torch.distributed': mock_torch.distributed,
    'agentic_rl.base.log.loggers': mock_loggers_module,
}):
    from agentic_rl.controllers.train_controller.train_weight_updater import (
        WeightUpdateActor,
        _ExportTracker,
    )


class TestWeightUpdateActor(unittest.TestCase):
    """Tests for WeightUpdateActor covering all methods."""

    def setUp(self):
        self.dispatch_actor = MagicMock()
        self.handler_a = MagicMock()
        self.handler_b = MagicMock()
        self.actor = WeightUpdateActor(
            dispatch_actor=self.dispatch_actor,
            actor_handlers=[self.handler_a, self.handler_b],
        )

    def tearDown(self):
        mock_ray.reset_mock()
        mock_torch.reset_mock()
        mock_loggers_module.reset_mock()

    # ---- __init__ -----------------------------------------------------------

    def test_init_attributes(self):
        """Verify all attributes are set correctly on construction."""
        self.assertIs(self.actor.dispatch_actor, self.dispatch_actor)
        self.assertEqual(self.actor.current_training_iter, 1)
        self.assertEqual(self.actor._exports, {})
        self.assertEqual(self.actor.weight_export_events, [])
        self.assertEqual(self.actor.export_durations, [])
        self.assertEqual(self.actor.finish_delays, [])
        self.assertFalse(self.actor.update_finished)

    # ---- _update_weights_async ---------------------------------------------

    def test_update_weights_async_list_handlers(self):
        """_update_weights_async creates a tracker and calls remote on list handlers."""
        self.actor._update_weights_async("/path/weights", iteration=5)

        self.assertEqual(self.actor.current_training_iter, 5)
        self.assertIn("/path/weights", self.actor._exports)
        tracker = self.actor._exports["/path/weights"]
        self.assertEqual(tracker.iteration, 5)
        self.assertEqual(tracker.expected, 2)
        self.assertEqual(tracker.seen, 0)

        self.handler_a.prepare_infer_params_to_cpu.remote.assert_called_once_with("/path/weights")
        self.handler_b.prepare_infer_params_to_cpu.remote.assert_called_once_with("/path/weights")

    def test_update_weights_async_non_list_handlers(self):
        """_update_weights_async calls prepare_infer_params_to_cpu directly for non-list."""
        handler_pool = MagicMock()
        handler_pool._workers = [1, 2, 3]
        actor = WeightUpdateActor(
            dispatch_actor=self.dispatch_actor,
            actor_handlers=handler_pool,
        )

        actor._update_weights_async("/path2", iteration=10)

        handler_pool.prepare_infer_params_to_cpu.assert_called_once_with("/path2")
        tracker = actor._exports["/path2"]
        self.assertEqual(tracker.expected, 3)

    # ---- _update_metrics ---------------------------------------------------

    def test_update_metrics_records_duration_and_delay(self):
        """_update_metrics appends duration, delay, and event info."""
        import time as real_time
        start_ts = real_time.time() - 5.0  # simulate 5 seconds ago
        self.actor.current_training_iter = 10

        self.actor._update_metrics("/path", start_ts, iteration=8)

        self.assertEqual(len(self.actor.export_durations), 1)
        self.assertGreater(self.actor.export_durations[0], 0)
        self.assertEqual(len(self.actor.finish_delays), 1)
        self.assertEqual(self.actor.finish_delays[0], 2)  # max(0, 10 - 8) = 2
        self.assertEqual(len(self.actor.weight_export_events), 1)
        event = self.actor.weight_export_events[0]
        self.assertEqual(event["weight_save_dir"], "/path")
        self.assertEqual(event["iteration"], 8)
        self.assertEqual(event["status"], "ok")

    def test_update_metrics_cleans_up_exports(self):
        """_update_metrics removes the entry from _exports."""
        self.actor._exports["/path"] = _ExportTracker(iteration=1, start_ts=0.0, expected=1)

        self.actor._update_metrics("/path", 0.0, iteration=1)

        self.assertNotIn("/path", self.actor._exports)

    # ---- _finalise_export --------------------------------------------------

    def test_finalise_export_notifies_and_updates_metrics(self):
        """_finalise_export calls dispatch_actor.notify_weights_update.remote and _update_metrics."""
        import time as real_time
        start_ts = real_time.time()
        self.actor.current_training_iter = 5

        self.actor._finalise_export("/dir", start_ts, iteration=5)

        self.dispatch_actor.notify_weights_update.remote.assert_called_once_with("/dir")
        self.assertEqual(len(self.actor.export_durations), 1)

    # ---- update_weights_to_file --------------------------------------------

    def test_update_weights_to_file(self):
        """update_weights_to_file delegates to _update_weights_async."""
        self.actor.update_weights_to_file("/w", iteration=3)

        self.assertEqual(self.actor.current_training_iter, 3)
        self.assertIn("/w", self.actor._exports)

    # ---- weight_saved ------------------------------------------------------

    def test_weight_saved_unknown_dir_is_noop(self):
        """weight_saved is a no-op when the dir has no tracker."""
        self.actor.weight_saved("/unknown")
        self.assertFalse(self.actor.update_finished)

    def test_weight_saved_increments_seen(self):
        """weight_saved increments seen on the tracker."""
        self.actor._update_weights_async("/path", iteration=1)
        tracker = self.actor._exports["/path"]
        self.assertEqual(tracker.seen, 0)

        self.actor.weight_saved("/path")
        self.assertEqual(tracker.seen, 1)

    def test_weight_saved_finalises_when_all_seen(self):
        """weight_saved calls _finalise_export when all shards are seen."""
        self.actor._update_weights_async("/path", iteration=2)
        # expected = 2 (two handlers)

        self.actor.weight_saved("/path")  # seen = 1
        self.assertFalse(self.actor.update_finished)

        self.actor.weight_saved("/path")  # seen = 2 >= expected
        self.assertTrue(self.actor.update_finished)
        self.dispatch_actor.notify_weights_update.remote.assert_called_once_with("/path")

    # ---- update_weights_finished -------------------------------------------

    def test_update_weights_finished_returns_false_initially(self):
        """update_weights_finished returns False when no update has finished."""
        result = self.actor.update_weights_finished()
        self.assertFalse(result)

    def test_update_weights_finished_returns_true_and_resets(self):
        """update_weights_finished returns True once and resets to False."""
        self.actor.update_finished = True
        result = self.actor.update_weights_finished()
        self.assertTrue(result)
        self.assertFalse(self.actor.update_finished)

        result2 = self.actor.update_weights_finished()
        self.assertFalse(result2)

    # ---- init_done ---------------------------------------------------------

    def test_init_done_is_noop(self):
        """init_done does nothing and does not raise."""
        self.actor.init_done()


if __name__ == '__main__':
    unittest.main()
