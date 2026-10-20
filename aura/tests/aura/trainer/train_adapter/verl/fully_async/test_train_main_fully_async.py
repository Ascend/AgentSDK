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

import sys
import unittest
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure aura package is importable when tests run from repo root
AURA_SRC = str(Path(__file__).resolve().parents[6])
if AURA_SRC not in sys.path:
    sys.path.insert(0, AURA_SRC)


# Build module mocks. The module under test imports ray, verl.trainer.main_ppo,
# aura.base.log.loggers and aura.trainer.train_adapter.verl.full_async.train_main.
# The latter re-imports ray and verl, so it must be stubbed before its real import.
mock_ray = MagicMock()
mock_ray_util_scheduling_strategies = MagicMock()
mock_ray.util.scheduling_strategies = mock_ray_util_scheduling_strategies


def _identity_remote(*dargs, **dkwargs):
    """``ray.remote`` decorator stub. Supports both ``@ray.remote`` (no parens)
    and ``@ray.remote(num_cpus=...)`` usage, preserving ``__wrapped__``."""
    # ``@ray.remote`` without parentheses: dargs[0] is the function itself
    if dargs and callable(dargs[0]) and not dkwargs:
        fn = dargs[0]
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        wrapper.__wrapped__ = fn
        return wrapper
    # ``@ray.remote(num_cpus=...)`` with parentheses
    def wrap(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        wrapper.__wrapped__ = fn
        return wrapper
    return wrap


mock_ray.remote = MagicMock(side_effect=_identity_remote)

mock_omegaconf = MagicMock()
mock_verl = MagicMock()
mock_verl.trainer.main_ppo.run_ppo = MagicMock()
mock_loggers = MagicMock()
mock_loggers.return_value.get_logger.return_value = MagicMock()

# Stub aura.trainer.train_adapter.verl.full_async.train_main to avoid pulling
# the full FullyAsyncTaskRunner dependency graph.
mock_full_async_train_main = MagicMock()
mock_full_async_train_main.FullyAsyncTaskRunner = MagicMock(name="FullyAsyncTaskRunner")

with patch.dict('sys.modules', {
    'ray': mock_ray,
    'ray.util': mock_ray.util,
    'ray.util.scheduling_strategies': mock_ray_util_scheduling_strategies,
    'omegaconf': mock_omegaconf,
    'verl': mock_verl,
    'verl.trainer': mock_verl.trainer,
    'verl.trainer.main_ppo': mock_verl.trainer.main_ppo,
    'aura.base.log.loggers': mock_loggers,
    'aura.trainer.train_adapter.verl.full_async.train_main': mock_full_async_train_main,
}):
    from aura.trainer.train_adapter.verl.fully_async import train_main as module_under_test
    from aura.trainer.train_adapter.verl.fully_async.train_main import (
        start_train,
        _require_async_config,
        _compute_sample_bounds,
        _inject_max_required_samples,
        _launch_rollout_monitor,
    )


def _make_async_conf(staleness=0.5, trigger_step=2, require_batches=4):
    """Build a tiny attribute bag mimicking the ``async_training`` config."""
    return MagicMock(
        staleness_threshold=staleness,
        trigger_parameter_sync_step=trigger_step,
        require_batches=require_batches,
    )


def _make_train_config(ppo_mini_batch_size=8, async_conf=None):
    """Build a tiny attribute bag mimicking the verl ``train_config``."""
    cfg = MagicMock()
    cfg.actor_rollout_ref.actor.ppo_mini_batch_size = ppo_mini_batch_size
    cfg.async_training = async_conf if async_conf is not None else _make_async_conf()
    return cfg


class TestRequireAsyncConfig(unittest.TestCase):
    """``_require_async_config`` resolves the async_training config and validates the missing case."""

    def test_returns_async_conf_when_present(self):
        cfg = _make_train_config()
        self.assertIs(_require_async_config(cfg), cfg.async_training)

    def test_raises_when_attribute_missing(self):
        cfg = MagicMock(spec=[])  # No async_training attribute
        with self.assertRaises(RuntimeError) as ctx:
            _require_async_config(cfg)
        self.assertIn("async_training", str(ctx.exception))

    def test_raises_when_async_conf_is_none(self):
        cfg = MagicMock()
        cfg.async_training = None
        with self.assertRaises(RuntimeError):
            _require_async_config(cfg)

    def test_raises_when_async_conf_is_empty(self):
        cfg = MagicMock()
        cfg.async_training = {}  # Empty dict is falsy
        with self.assertRaises(RuntimeError):
            _require_async_config(cfg)


class TestComputeSampleBounds(unittest.TestCase):
    """``_compute_sample_bounds`` computes required/max_required samples from async params."""

    def test_basic_formula(self):
        cfg = _make_train_config(ppo_mini_batch_size=8, async_conf=_make_async_conf(0.5, 2, 4))
        required, maximum = _compute_sample_bounds(cfg, cfg.async_training)
        # required = 8 * 4 = 32; max = 32 * (1 + 0.5) * 2 = 96
        self.assertEqual(required, 32)
        self.assertEqual(maximum, 96)

    def test_zero_staleness(self):
        cfg = _make_train_config(ppo_mini_batch_size=4, async_conf=_make_async_conf(0.0, 1, 1))
        required, maximum = _compute_sample_bounds(cfg, cfg.async_training)
        self.assertEqual(required, 4)
        self.assertEqual(maximum, 4)

    def test_large_trigger_step(self):
        cfg = _make_train_config(ppo_mini_batch_size=2, async_conf=_make_async_conf(0.0, 10, 1))
        required, maximum = _compute_sample_bounds(cfg, cfg.async_training)
        self.assertEqual(required, 2)
        self.assertEqual(maximum, 20)

    def test_uses_defaults_when_attrs_missing(self):
        # getattr fallbacks: staleness=0, trigger=1, batches=1, ppo_mini=1
        async_conf = MagicMock(spec=[])  # No attributes
        cfg = MagicMock()
        cfg.actor_rollout_ref.actor.ppo_mini_batch_size = 1
        required, maximum = _compute_sample_bounds(cfg, async_conf)
        self.assertEqual(required, 1)
        self.assertEqual(maximum, 1)

    def test_max_required_is_int(self):
        cfg = _make_train_config(ppo_mini_batch_size=3, async_conf=_make_async_conf(0.1, 2, 1))
        _, maximum = _compute_sample_bounds(cfg, cfg.async_training)
        # 3 * (1 + 0.1) * 2 = 6.6 -> int -> 6
        self.assertIsInstance(maximum, int)
        self.assertEqual(maximum, 6)


class TestInjectMaxRequiredSamples(unittest.TestCase):
    """``_inject_max_required_samples`` writes into a dict or OmegaConf config."""

    def test_writes_into_dict(self):
        rollout_config = {"existing": 1}
        _inject_max_required_samples(rollout_config, 42)
        self.assertEqual(rollout_config["max_required_samples"], 42)
        self.assertEqual(rollout_config["existing"], 1)

    def test_noop_when_none(self):
        # Should not raise
        _inject_max_required_samples(None, 42)

    def test_writes_into_omegaconf(self):
        rollout_config = MagicMock()
        # Source uses ``from omegaconf import OmegaConf`` inside the function,
        # so patch the real ``omegaconf.OmegaConf.update`` attribute.
        with patch("omegaconf.OmegaConf.update") as mock_update:
            _inject_max_required_samples(rollout_config, 100)
            mock_update.assert_called_once_with(rollout_config, "max_required_samples", 100)

    def test_swallows_omegaconf_failure(self):
        rollout_config = MagicMock()
        with patch.object(mock_omegaconf, "OmegaConf") as omega:
            omega.update.side_effect = RuntimeError("boom")
            # Should not raise
            _inject_max_required_samples(rollout_config, 100)


class TestLaunchRolloutMonitor(unittest.TestCase):
    """``_launch_rollout_monitor`` starts a daemon thread that exits the process on rollout failure."""

    def test_starts_daemon_thread(self):
        ref = MagicMock(name="ref")
        with patch.object(mock_ray, "get", return_value=None) as mock_get:
            # Thread will block on ray.get call - use a side effect that returns immediately
            thread_started = threading.Event()

            def fake_get(r):
                thread_started.set()
                return None

            mock_get.side_effect = fake_get
            _launch_rollout_monitor(ref)
            # Wait up to 1 second for the thread to start
            self.assertTrue(thread_started.wait(timeout=1.0))
            mock_get.assert_called_once_with(ref)

    def test_calls_os_exit_on_failure(self):
        ref = MagicMock(name="ref")
        exit_called = threading.Event()
        captured = {}

        def fake_get(r):
            raise RuntimeError("rollout crashed")

        def fake_exit(code):
            captured["code"] = code
            exit_called.set()
            # Raise to actually stop the thread
            raise SystemExit(code)

        with patch.object(mock_ray, "get", side_effect=fake_get), \
             patch("os._exit", side_effect=fake_exit):
            _launch_rollout_monitor(ref)
            self.assertTrue(exit_called.wait(timeout=2.0))
            self.assertEqual(captured["code"], 1)


class TestStartTrainIntegration(unittest.TestCase):
    """``start_train`` invokes each helper end-to-end and launches run_ppo."""

    def setUp(self):
        # Reset call counts (keep return_value/side_effect so module-bound
        # references like ``run_ppo`` still point to the same mock instance).
        mock_verl.reset_mock()
        mock_ray.reset_mock()

    def _build_full_call_args(self, train_config=None, rollout_config=None):
        train_config = train_config or _make_train_config()
        rollout_config = rollout_config if rollout_config is not None else {}
        return ("local", train_config, rollout_config, MagicMock(), MagicMock())

    def test_raises_when_async_training_missing(self):
        cfg = MagicMock(spec=[])
        args = self._build_full_call_args(train_config=cfg)
        # Mock the rollout/sample_queue modules in sys.modules so start_train's
        # ``from ... import ...`` does not trigger real torch import.
        mock_rollout_module = MagicMock()
        mock_sq_module = MagicMock()
        with patch.dict("sys.modules", {
            "aura.trainer.rollout.rollout_service": mock_rollout_module,
            "aura.controllers.rollout_controller.sample_queue": mock_sq_module,
        }):
            with self.assertRaises(RuntimeError):
                start_train.__wrapped__(*args)

    def test_full_pipeline_invokes_run_ppo(self):
        args = self._build_full_call_args()
        mock_rollout_module = MagicMock()
        mock_sq_module = MagicMock()
        with patch.dict("sys.modules", {
            "aura.trainer.rollout.rollout_service": mock_rollout_module,
            "aura.controllers.rollout_controller.sample_queue": mock_sq_module,
        }):
            mock_rollout_module.start_fully_async_rollout.options.return_value.remote.return_value = MagicMock(name="ref")
            start_train.__wrapped__(*args)
            mock_sq_module.create_sample_queue.assert_called_once()
            mock_rollout_module.start_fully_async_rollout.options.assert_called_once()
            mock_verl.trainer.main_ppo.run_ppo.assert_called_once()

    def test_full_pipeline_injects_max_required_into_dict(self):
        rollout_config = {}
        args = self._build_full_call_args(rollout_config=rollout_config)
        mock_rollout_module = MagicMock()
        mock_sq_module = MagicMock()
        with patch.dict("sys.modules", {
            "aura.trainer.rollout.rollout_service": mock_rollout_module,
            "aura.controllers.rollout_controller.sample_queue": mock_sq_module,
        }):
            mock_rollout_module.start_fully_async_rollout.options.return_value.remote.return_value = MagicMock(name="ref")
            start_train.__wrapped__(*args)
            self.assertIn("max_required_samples", rollout_config)
            self.assertGreater(rollout_config["max_required_samples"], 0)


if __name__ == '__main__':
    unittest.main()
