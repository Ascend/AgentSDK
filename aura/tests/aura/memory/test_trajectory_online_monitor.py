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
from unittest.mock import MagicMock, patch, call
import pytest

import numpy as np
import zlib
import math
from scipy.stats import wasserstein_distance
from collections import Counter


# ---------------------------------------------------------------------------
# Fixture: fake module tree for trajectory_online_monitor
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_monitor_env(monkeypatch):
    """Construct isolated fake modules and return the module under test."""

    # ---- fake datetime ----
    fake_datetime = types.ModuleType("datetime")
    class fake_dt_class:
        @staticmethod
        def now():
            m = MagicMock()
            m.strftime.return_value = "2026-01-01-12-00-00"
            return m
    fake_datetime.datetime = fake_dt_class

    # ---- fake multiprocessing ----
    fake_mp = types.ModuleType("multiprocessing")
    fake_mp.Queue = MagicMock
    fake_mp.Process = MagicMock

    # ---- aura packages to locate the real file ----
    import os as _os
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_memory = types.ModuleType("aura.memory")
    fake_aura_memory.__path__ = [_os.path.join(base, "memory")]
    fake_aura_memory_trajmon = types.ModuleType("aura.memory.trajectory_monitor")
    fake_aura_memory_trajmon.__path__ = [_os.path.join(base, "memory/trajectory_monitor")]

    fakes = {
        "datetime": fake_datetime,
        "multiprocessing": fake_mp,
        "aura": fake_aura,
        "aura.memory": fake_aura_memory,
        "aura.memory.trajectory_monitor": fake_aura_memory_trajmon,
    }

    monkeypatch.setenv("TRAJECTORY_MONITOR_MODE", "NONE")

    target = "aura.memory.trajectory_monitor.trajectory_online_monitor"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes), patch("builtins.open") as mock_open:
        import aura.memory.trajectory_monitor.trajectory_online_monitor as mod
        yield {
            "mod": mod,
            "mp": fake_mp,
            "mock_open": mock_open,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_config(metrics=None, remove_tool_response=False, bs=10):
    return {
        "metrics": metrics or {},
        "remove_tool_response": remove_tool_response,
        "bs": bs,
    }


# ---------------------------------------------------------------------------
# Tests for metric functions
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_distinct_n(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        assert mod.get_distinct_n([1,2,3,1,2], n=2) > 0
        assert mod.get_distinct_n([1], n=2) == 0.0

    def test_compression_ratio(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        ratio = mod.get_compression_ratio([1,2,3])
        assert isinstance(ratio, float) and ratio > 0
        assert mod.get_compression_ratio([]) == 1.0

    def test_token_entropy(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        assert mod.get_token_entropy([1,1,1]) == 0.0
        assert mod.get_token_entropy([1,2]) > 0

    def test_vocab_gini(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        assert mod.get_vocab_gini([1,1,1]) == 0.0
        gini = mod.get_vocab_gini([1,2])
        assert gini >= 0

    def test_intra_kld(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        kld = mod.get_intra_kld([1,2,3,4])
        assert kld >= 0
        kld2 = mod.get_intra_kld([1,1])
        assert kld2 > -1e-12

    def test_LZ_complexity(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        assert 0 <= mod.get_LZ_complexity([1,2,3,4]) <= 1
        assert mod.get_LZ_complexity([]) == 0.0


# ---------------------------------------------------------------------------
# Tests for utility functions
# ---------------------------------------------------------------------------
class TestUtilityFunctions:
    def test_compute_metric_wrapper_with_n(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        res = mod.compute_metric_wrapper([1,2,1], mod.get_distinct_n, n=2)
        assert isinstance(res, float)

    def test_compute_metric_wrapper_without_n(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        res = mod.compute_metric_wrapper([1,2], mod.get_token_entropy)
        assert isinstance(res, float)

    def test_moving_window_global(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        vals = mod.moving_window_compute([1,2,3], mod.get_token_entropy)
        assert len(vals) == 1

    def test_moving_window_windowed(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        vals = mod.moving_window_compute([1,2,3,4,5], mod.get_token_entropy,
                                         window_size=3, stride=2)
        assert len(vals) > 1


# ---------------------------------------------------------------------------
# Tests for remove_segments
# ---------------------------------------------------------------------------
class TestRemoveSegments:
    def test_removes_segments(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        data = [1, 151665, 2, 151666, 3]
        result = mod.remove_segments(data)
        assert result == [1, 3]

    def test_no_segments(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        data = [1, 2, 3]
        assert mod.remove_segments(data) == [1, 2, 3]

    def test_start_only(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        data = [151665, 2, 3]
        assert mod.remove_segments(data) == []


# ---------------------------------------------------------------------------
# Tests for TrajectoryOnlineMonitor
# ---------------------------------------------------------------------------
class TestTrajectoryOnlineMonitor:
    @pytest.fixture
    def monitor(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        with patch("os.path.exists", return_value=True), \
             patch("json.load", return_value=make_config(
                 metrics={"distinct_n": {"type": "distinct_n", "thresholds": {"min": 0.1}}}
             )):
            return mod.TrajectoryOnlineMonitor()

    def test_step_no_metrics(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        with patch("os.path.exists", return_value=True), \
             patch("json.load", return_value=make_config()):
            mon = mod.TrajectoryOnlineMonitor()
        mon.step([[1,2,3]])

    def test_step_with_metric(self, monitor):
        monitor.step([[1,2,3]])
        assert len(monitor.results["distinct_n"]) == 1

    def test_step_remove_tool_response(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        config = make_config(
            metrics={"distinct_n": {"type": "distinct_n"}},
            remove_tool_response=True
        )
        with patch("os.path.exists", return_value=True), \
             patch("json.load", return_value=config):
            mon = mod.TrajectoryOnlineMonitor()
        mon.step([[151665, 2, 151666, 3]])
        assert len(mon.results["distinct_n"]) == 1

    def test_initialization(self, monitor):
        monitor.step([[1,2,3]], iteration_id=0)
        assert monitor.initialized is True

    def test_outlier_detection(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        config = make_config(
            metrics={"token_entropy": {"type": "token_entropy",
                                       "thresholds": {"min": 0.1}}}
        )
        with patch("os.path.exists", return_value=True), \
             patch("json.load", return_value=config):
            mon = mod.TrajectoryOnlineMonitor()
        mon.step([[1,1,1,1]], iteration_id=0)
        assert len(mon.results["token_entropy"]) > 0
        assert len(mon.outliers) == 0


# ---------------------------------------------------------------------------
# Tests for _MonitorLogicHandler
# ---------------------------------------------------------------------------
class TestMonitorLogicHandler:
    @pytest.fixture
    def handler(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        with patch("os.path.exists", return_value=True), \
             patch("json.load", return_value=make_config(
                 metrics={"distinct_n": {"type": "distinct_n", "thresholds": {"min": 0.1}}}
             )):
            return mod._MonitorLogicHandler()

    def test_process_batch_no_metrics(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        with patch("os.path.exists", return_value=True), \
             patch("json.load", return_value=make_config()):
            h = mod._MonitorLogicHandler()
        h.process_batch([[1,2,3]])

    def test_process_batch_initialization(self, handler):
        handler.process_batch([[1,2,3]], iteration_id=0)
        assert handler.initialized is True

    def test_log_function(self, handler, fake_monitor_env):
        handler._log("test message")
        fake_monitor_env["mock_open"].assert_called()

    def test_outlier_detection(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        config = make_config(
            metrics={"token_entropy": {"type": "token_entropy",
                                       "thresholds": {"min": 0.1}}}
        )
        with patch("os.path.exists", return_value=True), \
             patch("json.load", return_value=config):
            h = mod._MonitorLogicHandler()
        h.process_batch([[1,1,1,1]], iteration_id=0)
        assert len(h.results["token_entropy"]) > 0
        assert len(h.outliers) == 0

    def test_log_exception_swallowed(self, handler, fake_monitor_env):
        fake_monitor_env["mock_open"].side_effect = IOError("disk full")
        handler._log("should not crash")


# ---------------------------------------------------------------------------
# Tests for windowed metrics and outlier detection
# ---------------------------------------------------------------------------
class TestWindowedAndOutlier:
    def test_windowed_metric_and_reference(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        config = {
            "metrics": {
                "window_entropy": {
                    "type": "token_entropy",
                    "window_size": 3,
                    "stride": 2,
                    "wd": 0.5,
                }
            },
            "bs": 10,
        }
        with patch("os.path.exists", return_value=True), \
             patch("json.load", return_value=config):
            mon = mod.TrajectoryOnlineMonitor()
        tokens = [1,2,3,4,5,6]
        mon.step([tokens], iteration_id=0)
        assert mon.initialized
        assert 'window_entropy' in mon.reference_samples
        assert 'window_entropy_wd' in mon.results
        assert len(mon.results['window_entropy']) > 0

    def test_logic_handler_windowed(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        config = {
            "metrics": {
                "window_entropy": {
                    "type": "token_entropy",
                    "window_size": 3,
                    "stride": 2,
                    "wd": 0.5,
                }
            },
            "bs": 10,
        }
        with patch("os.path.exists", return_value=True), \
             patch("json.load", return_value=config):
            h = mod._MonitorLogicHandler()
        tokens = [1,2,3,4,5,6]
        h.process_batch([tokens], iteration_id=0)
        assert h.initialized
        assert 'window_entropy' in h.reference_samples

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_detect_and_report_triggers_outliers(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        h = mod._MonitorLogicHandler()
        h.config = {
            "metrics": {
                "window_entropy": {
                    "type": "token_entropy",
                    "window_size": 2,
                    "stride": 1,
                    "wd": 0.1,
                }
            },
            "bs": 10,
        }
        h.reference_samples = {"window_entropy": np.array([0.1, 0.2, 0.3])}
        h.initialized = True
        h.results = {
            "window_entropy": [0.5, 0.8, 1.2],
            "window_entropy_wd": [],
        }

        batch_results = {"window_entropy": [np.array([0.0, 0.5, 0.8])]}
        batch_scalar = {"window_entropy": [0.0]}
        h._detect_and_report(batch_scalar, batch_results, iteration_id=0)

        assert len(h.outliers) > 0
        assert len(h.results["window_entropy_wd"]) == 1


# ---------------------------------------------------------------------------
# Tests for background entry point of AsyncMonitor
# ---------------------------------------------------------------------------
class TestAsyncBackground:
    def test_background_entry_point(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        mock_queue = MagicMock()
        mock_queue.get.side_effect = [([1,2,3], 0), KeyboardInterrupt]
        with patch.object(mod._MonitorLogicHandler, 'process_batch') as mock_process:
            mod.AsyncMonitor._background_entry_point(mock_queue, "fake_path")
            mock_process.assert_called_once()

    def test_background_exception_writes_crash_log(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        mock_queue = MagicMock()
        mock_queue.get.side_effect = RuntimeError("worker crash")
        with patch("builtins.open") as crash_open:
            mod.AsyncMonitor._background_entry_point(mock_queue, "fake_path")
            # The open mock will be called twice: once for the log file, once for crash log
            crash_open.assert_any_call("monitor_crash.log", "w")


# ---------------------------------------------------------------------------
# Tests for monitor classes
# ---------------------------------------------------------------------------
class TestMonitors:
    def test_synchronous_monitor(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        with patch.object(mod, "SynchronousMonitor", wraps=mod.SynchronousMonitor):
            m = mod.SynchronousMonitor()
            m.step([[1,2,3]])

    def test_async_monitor(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        m = mod.AsyncMonitor()
        m.queue.put_nowait = MagicMock()
        m.step([[1,2,3]])
        m.queue.put_nowait.assert_called()

    def test_async_monitor_queue_error(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        m = mod.AsyncMonitor()
        m.queue.put_nowait = MagicMock(side_effect=Exception("full"))
        m.step([[1,2,3]])  # should print error, not crash

    def test_dummy_monitor(self, fake_monitor_env):
        mod = fake_monitor_env["mod"]
        m = mod.DummyMonitor()
        m.step([[1,2,3]])

    def test_get_monitor_instance_none(self, fake_monitor_env, monkeypatch):
        monkeypatch.setenv("TRAJECTORY_MONITOR_MODE", "NONE")
        mod = fake_monitor_env["mod"]
        inst = mod._get_monitor_instance()
        assert isinstance(inst, mod.DummyMonitor)

    def test_get_monitor_instance_async(self, fake_monitor_env, monkeypatch):
        monkeypatch.setenv("TRAJECTORY_MONITOR_MODE", "ASYNC")
        mod = fake_monitor_env["mod"]
        inst = mod._get_monitor_instance()
        assert isinstance(inst, mod.AsyncMonitor)

    def test_get_monitor_instance_sync(self, fake_monitor_env, monkeypatch):
        monkeypatch.setenv("TRAJECTORY_MONITOR_MODE", "SYNC")
        mod = fake_monitor_env["mod"]
        inst = mod._get_monitor_instance()
        assert isinstance(inst, mod.SynchronousMonitor)
