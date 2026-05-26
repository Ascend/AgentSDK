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
import types
from unittest.mock import MagicMock, patch
import pytest
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Fixture: fake module tree for train_weight_updater
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_updater_env():
    """Construct isolated fake modules and return the module under test."""

    # ---- fake time ----
    fake_time = types.ModuleType("time")
    fake_time.time = MagicMock(return_value=1000.0)

    # ---- fake ray ----
    fake_ray = types.ModuleType("ray")

    # ray.remote decorator that simply returns the class unchanged
    def ray_remote(cls):
        return cls

    fake_ray.remote = ray_remote
    fake_ray.get = MagicMock()

    # ---- fake loggers ----
    fake_loggers_mod = types.ModuleType("aura.base.log.loggers")
    mock_logger = MagicMock()
    fake_loggers_mod.Loggers = MagicMock(
        return_value=MagicMock(get_logger=MagicMock(return_value=mock_logger))
    )

    # ---- aura packages to locate the real file ----
    import os
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_log = types.ModuleType("aura.base.log")
    fake_aura_base_log.__path__ = []
    fake_aura_controllers = types.ModuleType("aura.controllers")
    fake_aura_controllers.__path__ = []
    fake_aura_controllers_train_controller = types.ModuleType(
        "aura.controllers.train_controller"
    )
    fake_aura_controllers_train_controller.__path__ = [
        os.path.join(base, "controllers/train_controller")
    ]

    fakes = {
        "time": fake_time,
        "ray": fake_ray,
        "aura.base.log.loggers": fake_loggers_mod,
        "aura": fake_aura,
        "aura.base": fake_aura_base,
        "aura.base.log": fake_aura_base_log,
        "aura.controllers": fake_aura_controllers,
        "aura.controllers.train_controller": fake_aura_controllers_train_controller,
    }

    target = "aura.controllers.train_controller.train_weight_updater"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.controllers.train_controller.train_weight_updater as mod
        yield {
            "mod": mod,
            "fake_ray": fake_ray,
            "fake_time": fake_time,
            "mock_logger": mock_logger,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_worker_with_actor_handlers():
    """Create a mock actor_worker that has actor_handlers attribute."""
    worker = MagicMock()
    # Each handler is a mock with remote methods
    handler1 = MagicMock()
    handler2 = MagicMock()
    worker.actor_handlers = [handler1, handler2]
    return worker, [handler1, handler2]


def make_worker_without_actor_handlers():
    """Create a mock actor_worker that does NOT have actor_handlers."""
    worker = MagicMock()
    # Remove actor_handlers attribute to simulate the else branch
    del worker.actor_handlers
    # Provide _workers attribute as fallback
    worker._workers = [MagicMock(), MagicMock()]
    return worker


def make_dispatch_actor():
    return MagicMock()


def make_actor(dispatch_actor, actor_worker):
    """Instantiate a WeightUpdateActor with given dependencies."""
    from aura.controllers.train_controller.train_weight_updater import WeightUpdateActor

    return WeightUpdateActor(dispatch_actor, actor_worker)


# ---------------------------------------------------------------------------
# Tests for _ExportTracker
# ---------------------------------------------------------------------------
class TestExportTracker:
    def test_create_tracker(self, fake_updater_env):
        """_ExportTracker can be created and has default seen=0."""
        mod = fake_updater_env["mod"]
        tracker = mod._ExportTracker(iteration=5, start_ts=0.0, expected=10)
        assert tracker.iteration == 5
        assert tracker.start_ts == 0.0
        assert tracker.expected == 10
        assert tracker.seen == 0


# ---------------------------------------------------------------------------
# Tests for __init__
# ---------------------------------------------------------------------------
class TestInit:
    def test_init_with_actor_handlers(self, fake_updater_env):
        """When actor_worker has actor_handlers, it is assigned to self.actor_handlers."""
        worker, _ = make_worker_with_actor_handlers()
        actor = make_actor(MagicMock(), worker)
        assert actor.actor_handlers is worker.actor_handlers
        assert actor.current_training_iter == 1
        assert actor.update_finished is False
        assert actor._exports == {}
        assert actor.weight_export_events == []
        assert actor.export_durations == []
        assert actor.finish_delays == []

    def test_init_without_actor_handlers(self, fake_updater_env):
        """When actor_worker has no actor_handlers, _workers is used instead."""
        worker = make_worker_without_actor_handlers()
        actor = make_actor(MagicMock(), worker)
        assert actor.actor_handlers is worker._workers


# ---------------------------------------------------------------------------
# Tests for _update_weights_async
# ---------------------------------------------------------------------------
class TestUpdateWeightsAsync:
    def test_with_actor_handlers(self, fake_updater_env):
        """Branch with actor_handlers calls prepare_infer_params_to_cpu on each handler."""
        worker, handlers = make_worker_with_actor_handlers()
        dispatch_actor = make_dispatch_actor()
        actor = make_actor(dispatch_actor, worker)

        # Call async update
        actor._update_weights_async("dir1", iteration=3)

        # Check that tracker was created
        assert "dir1" in actor._exports
        tracker = actor._exports["dir1"]
        assert tracker.iteration == 3
        assert tracker.expected == len(handlers)

        # Each handler should have been called with .prepare_infer_params_to_cpu.remote
        for h in handlers:
            h.prepare_infer_params_to_cpu.remote.assert_called_once_with("dir1")

        # ray.get should have been called for each handler
        assert fake_updater_env["fake_ray"].get.call_count == len(handlers)
        # logger should have been used
        fake_updater_env["mock_logger"].info.assert_called()

    def test_without_actor_handlers(self, fake_updater_env):
        """Branch without actor_handlers calls prepare_infer_params_to_cpu directly on worker."""
        worker = make_worker_without_actor_handlers()
        worker.prepare_infer_params_to_cpu = MagicMock()
        dispatch_actor = make_dispatch_actor()
        actor = make_actor(dispatch_actor, worker)

        actor._update_weights_async("dir2", iteration=5)
        worker.prepare_infer_params_to_cpu.assert_called_once_with("dir2")
        # ray.get should not be called
        fake_updater_env["fake_ray"].get.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for weight_saved
# ---------------------------------------------------------------------------
class TestWeightSaved:
    def test_increments_and_triggers_finalise(self, fake_updater_env):
        """After expected number of saves, finalise export is triggered."""
        worker, _ = make_worker_with_actor_handlers()
        dispatch_actor = make_dispatch_actor()
        actor = make_actor(dispatch_actor, worker)

        # Prepare a tracker with expected=2
        actor._exports["dir"] = fake_updater_env["mod"]._ExportTracker(
            iteration=1, start_ts=900.0, expected=2
        )

        # First save
        actor.weight_saved("dir")
        assert actor.update_finished is False
        assert actor._exports["dir"].seen == 1

        # Second save -> triggers finalise
        actor.weight_saved("dir")
        assert actor.update_finished is True
        # Tracker should have been removed by _update_metrics
        assert "dir" not in actor._exports
        # dispatch_actor.notify_weights_update.remote should have been called
        dispatch_actor.notify_weights_update.remote.assert_called_once_with("dir")

    def test_missing_tracker_returns_early(self, fake_updater_env):
        """If tracker does not exist, method returns immediately."""
        worker, _ = make_worker_with_actor_handlers()
        actor = make_actor(make_dispatch_actor(), worker)
        # No tracker for "unknown_dir" -> should not raise
        actor.weight_saved("unknown_dir")
        # Nothing should change
        assert actor.update_finished is False


# ---------------------------------------------------------------------------
# Tests for _update_metrics
# ---------------------------------------------------------------------------
class TestUpdateMetrics:
    def test_records_metrics_and_removes_tracker(self, fake_updater_env):
        """_update_metrics populates lists and logs event, then pops tracker."""
        worker, _ = make_worker_with_actor_handlers()
        dispatch_actor = make_dispatch_actor()
        actor = make_actor(dispatch_actor, worker)
        actor.current_training_iter = 10

        # Add a tracker that will be popped
        tracker = fake_updater_env["mod"]._ExportTracker(
            iteration=5, start_ts=990.0, expected=2
        )
        actor._exports["dir"] = tracker

        # Mock time.time to return a known end time
        fake_updater_env["fake_time"].time.return_value = 1000.0

        actor._update_metrics("dir", 990.0, 5)

        # Check duration
        assert actor.export_durations == [10.0]
        # finish_delay = max(0, current_training_iter - iteration) = max(0, 10-5) = 5
        assert actor.finish_delays == [5]
        # Event log
        assert len(actor.weight_export_events) == 1
        event = actor.weight_export_events[0]
        assert event["weight_save_dir"] == "dir"
        assert event["duration"] == 10.0
        assert event["status"] == "ok"
        assert event["iteration"] == 5
        assert event["finish_delay_iters"] == 5

        # Tracker should be removed
        assert "dir" not in actor._exports

    def test_finish_delay_clamped_to_zero(self, fake_updater_env):
        """finish_delay should be max(0, ...)."""
        worker, _ = make_worker_with_actor_handlers()
        actor = make_actor(make_dispatch_actor(), worker)
        actor.current_training_iter = 2  # less than iteration 5 -> finish_delay = max(0, 2-5)=0

        tracker = fake_updater_env["mod"]._ExportTracker(
            iteration=5, start_ts=0.0, expected=1
        )
        actor._exports["d"] = tracker

        actor._update_metrics("d", 0.0, 5)
        assert actor.finish_delays == [0]


# ---------------------------------------------------------------------------
# Tests for _finalise_export
# ---------------------------------------------------------------------------
class TestFinaliseExport:
    def test_calls_notify_and_update_metrics(self, fake_updater_env):
        """_finalise_export notifies dispatch and then updates metrics."""
        worker, _ = make_worker_with_actor_handlers()
        dispatch_actor = make_dispatch_actor()
        actor = make_actor(dispatch_actor, worker)

        # Prevent _update_metrics from interfering by checking it's called later
        with patch.object(actor, "_update_metrics") as mock_metrics:
            actor._finalise_export("dir", 800.0, 2)

        dispatch_actor.notify_weights_update.remote.assert_called_once_with("dir")
        mock_metrics.assert_called_once_with("dir", 800.0, 2)


# ---------------------------------------------------------------------------
# Tests for update_weights_to_file
# ---------------------------------------------------------------------------
class TestUpdateWeightsToFile:
    def test_delegates_to_async(self, fake_updater_env):
        """update_weights_to_file simply calls _update_weights_async."""
        worker, _ = make_worker_with_actor_handlers()
        actor = make_actor(make_dispatch_actor(), worker)
        with patch.object(actor, "_update_weights_async") as mock_async:
            actor.update_weights_to_file("wdir", 7)
            mock_async.assert_called_once_with(weight_save_dir="wdir", iteration=7)


# ---------------------------------------------------------------------------
# Tests for update_weights_finished
# ---------------------------------------------------------------------------
class TestUpdateWeightsFinished:
    def test_returns_true_and_resets(self, fake_updater_env):
        """When update_finished is True, returns True and resets to False."""
        worker, _ = make_worker_with_actor_handlers()
        actor = make_actor(make_dispatch_actor(), worker)
        actor.update_finished = True
        assert actor.update_weights_finished() is True
        assert actor.update_finished is False

    def test_returns_false_when_not_finished(self, fake_updater_env):
        """When update_finished is False, returns False."""
        worker, _ = make_worker_with_actor_handlers()
        actor = make_actor(make_dispatch_actor(), worker)
        assert actor.update_weights_finished() is False


# ---------------------------------------------------------------------------
# Tests for init_done
# ---------------------------------------------------------------------------
class TestInitDone:
    def test_is_noop(self, fake_updater_env):
        """init_done does nothing."""
        worker, _ = make_worker_with_actor_handlers()
        actor = make_actor(make_dispatch_actor(), worker)
        actor.init_done()  # no exception
