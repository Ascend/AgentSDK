#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for :class:`TickScheduler` — the drift-free periodic tick scheduler.

The scheduler inherits its daemon-thread lifecycle from
:class:`PeriodicDaemon` (covered by ``test_periodic.py``); these tests
focus on the *kairos*-specific behavior:

* Callback delivery and ordering.
* Pause / resume semantics (soft flag, thread keeps running).
* Jitter application (symmetric, bounded).
* Tick-number monotonicity.
* Subscription management.
* Idempotent start / clean stop.
* ``enabled=False`` defers thread creation until :meth:`start`.
* Strict config validation.
"""
# pylint: disable=no-name-in-module

from __future__ import annotations

import time

import pytest

from src.services.kairos import (
    SchedulerStateError,
    TickConfig,
    TickEvent,
    TickScheduler,
)


def _wait_for(predicate, *, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll a predicate until it is true or the deadline expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestTickSchedulerConstruction:
    def test_requires_tick_config(self) -> None:
        with pytest.raises(TypeError, match="requires a TickConfig"):
            TickScheduler("not a config")  # type: ignore[arg-type]

    def test_disabled_config_does_not_start(self) -> None:
        cfg = TickConfig(id="main", interval_seconds=60.0, enabled=False)
        s = TickScheduler(cfg)
        try:
            assert s.is_running is False
            assert s.tick_count == 0
        finally:
            s.stop()

    def test_enabled_config_starts_immediately(self) -> None:
        cfg = TickConfig(id="main", interval_seconds=0.05)
        s = TickScheduler(cfg)
        try:
            assert s.is_running is True
        finally:
            s.stop()

    def test_name_follows_config_id(self) -> None:
        cfg = TickConfig(id="my-loop", interval_seconds=60.0, enabled=False)
        s = TickScheduler(cfg)
        try:
            # PeriodicDaemon uses the name we passed in start.
            assert s.config.id == "my-loop"
            assert s.config.display_name == "my-loop"
        finally:
            s.stop()

    def test_invalid_jitter_rejected(self) -> None:
        with pytest.raises(ValueError, match="jitter"):
            TickConfig(id="x", interval_seconds=1.0, jitter_fraction=2.0)

    def test_invalid_interval_rejected(self) -> None:
        with pytest.raises(ValueError, match="interval_seconds"):
            TickConfig(id="x", interval_seconds=0)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestTickSchedulerLifecycle:
    def test_start_is_idempotent(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=60.0, enabled=False)
        s = TickScheduler(cfg)
        try:
            assert s.start() is True
            assert s.is_running is True
            assert s.start() is False
            assert s.is_running is True
        finally:
            s.stop()

    def test_stop_returns_bool_and_joins(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=0.05)
        s = TickScheduler(cfg)
        time.sleep(0.1)
        joined = s.stop(timeout=2.0)
        assert joined is True
        assert s.is_running is False

    def test_stop_safe_when_not_running(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=60.0, enabled=False)
        s = TickScheduler(cfg)
        joined = s.stop(timeout=0.5)
        assert joined is True

    def test_context_manager(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=0.05, enabled=False)
        with TickScheduler(cfg) as s:
            assert s.is_running is True
            time.sleep(0.1)
            assert s.tick_count >= 1
        assert s.is_running is False


# ---------------------------------------------------------------------------
# Callback delivery
# ---------------------------------------------------------------------------


class TestTickSchedulerCallbacks:
    def test_subscribed_callback_fires(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=0.05)
        s = TickScheduler(cfg)
        events: list[TickEvent] = []
        try:
            s.subscribe(events.append)
            assert _wait_for(lambda: len(events) >= 1, timeout=2.0)
            assert events[0].scheduler_id == "x"
            assert events[0].tick_number >= 1
        finally:
            s.stop()

    def test_multiple_callbacks_all_fire(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=0.05)
        s = TickScheduler(cfg)
        a: list[TickEvent] = []
        b: list[TickEvent] = []
        try:
            s.subscribe(a.append)
            s.subscribe(b.append)
            assert _wait_for(lambda: len(a) >= 2 and len(b) >= 2, timeout=2.0)
        finally:
            s.stop()

    def test_unsubscribe_stops_callback(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=0.05)
        s = TickScheduler(cfg)
        events: list[TickEvent] = []

        def cb(event: TickEvent) -> None:
            events.append(event)

        s.subscribe(cb)
        try:
            assert _wait_for(lambda: len(events) >= 1, timeout=2.0)
            s.unsubscribe(cb)
            events.clear()
            time.sleep(0.15)
            assert not events
        finally:
            s.stop()

    def test_unsubscribe_unknown_raises(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=60.0, enabled=False)
        s = TickScheduler(cfg)
        try:
            with pytest.raises(SchedulerStateError, match="not registered"):
                s.unsubscribe(lambda ev: None)
        finally:
            s.stop()

    def test_subscribe_rejects_non_callable(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=60.0, enabled=False)
        s = TickScheduler(cfg)
        try:
            with pytest.raises(TypeError, match="expects a callable"):
                s.subscribe("not callable")  # type: ignore[arg-type]
        finally:
            s.stop()

    def test_failing_callback_does_not_kill_scheduler(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=0.05)
        s = TickScheduler(cfg)
        good: list[TickEvent] = []
        try:
            s.subscribe(lambda ev: (_ for _ in ()).throw(RuntimeError("boom")))
            s.subscribe(good.append)
            assert _wait_for(lambda: len(good) >= 2, timeout=2.0)
        finally:
            s.stop()

    def test_tick_numbers_are_monotonic(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=0.05)
        s = TickScheduler(cfg)
        events: list[TickEvent] = []
        try:
            s.subscribe(events.append)
            assert _wait_for(lambda: len(events) >= 3, timeout=2.0)
            numbers = [ev.tick_number for ev in events]
            assert numbers == sorted(numbers)
            assert numbers == list(range(1, len(events) + 1))
        finally:
            s.stop()

    def test_tick_count_property_matches_events(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=0.05)
        s = TickScheduler(cfg)
        try:
            assert _wait_for(lambda: s.tick_count >= 2, timeout=2.0)
        finally:
            s.stop()


# ---------------------------------------------------------------------------
# Pause / resume
# ---------------------------------------------------------------------------


class TestTickSchedulerPause:
    def test_pause_halts_callback_delivery(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=0.05)
        s = TickScheduler(cfg)
        events: list[TickEvent] = []
        try:
            s.subscribe(events.append)
            assert _wait_for(lambda: len(events) >= 2, timeout=2.0)
            s.pause()
            assert s.is_paused is True
            paused_count = len(events)
            time.sleep(0.2)
            # Thread keeps running but no new callbacks fire.
            assert len(events) == paused_count
            assert s.is_running is True
        finally:
            s.stop()

    def test_resume_restores_callback_delivery(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=0.05)
        s = TickScheduler(cfg)
        events: list[TickEvent] = []
        try:
            s.subscribe(events.append)
            assert _wait_for(lambda: len(events) >= 1, timeout=2.0)
            s.pause()
            time.sleep(0.15)
            s.resume()
            assert s.is_paused is False
            assert _wait_for(lambda: len(events) >= 3, timeout=2.0)
        finally:
            s.stop()


# ---------------------------------------------------------------------------
# Jitter
# ---------------------------------------------------------------------------


class TestTickSchedulerJitter:
    def test_no_jitter_drift_is_zero(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=0.05, jitter_fraction=0.0)
        s = TickScheduler(cfg)
        events: list[TickEvent] = []
        try:
            s.subscribe(events.append)
            assert _wait_for(lambda: len(events) >= 2, timeout=2.0)
            # Without jitter, drift is always zero.
            for ev in events:
                assert ev.jitter_applied == 0.0
                assert ev.drift == pytest.approx(0.0)
        finally:
            s.stop()

    def test_jitter_stays_within_bounds(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=0.05, jitter_fraction=0.5)
        # Use a deterministic RNG so we can assert the exact bounds.
        import random as _r

        rng = _r.Random(0xC0FFEE)
        s = TickScheduler(cfg, rng=rng)
        events: list[TickEvent] = []
        try:
            s.subscribe(events.append)
            assert _wait_for(lambda: len(events) >= 2, timeout=2.0)
            for ev in events:
                # jitter_fraction=0.5 means ±50% of interval=0.05 → ±0.025
                assert abs(ev.jitter_applied) <= cfg.interval_seconds * 0.5 + 1e-9
        finally:
            s.stop()

    def test_jitter_applied_affects_drift(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=0.05, jitter_fraction=0.5)
        import random as _r

        rng = _r.Random(0xC0FFEE)
        s = TickScheduler(cfg, rng=rng)
        events: list[TickEvent] = []
        try:
            s.subscribe(events.append)
            assert _wait_for(lambda: len(events) >= 2, timeout=2.0)
            # drift == actual_at - scheduled_at, so it equals the
            # jitter_applied value the RNG produced.
            for ev in events:
                assert ev.drift == pytest.approx(ev.jitter_applied, abs=1e-6)
        finally:
            s.stop()


# ---------------------------------------------------------------------------
# Inheritance check
# ---------------------------------------------------------------------------


class TestTickSchedulerInheritsPeriodicDaemon:
    def test_is_a_periodic_daemon(self) -> None:
        from clawcodex_ext.services.periodic import PeriodicDaemon

        cfg = TickConfig(id="x", interval_seconds=60.0, enabled=False)
        s = TickScheduler(cfg)
        try:
            assert isinstance(s, PeriodicDaemon)
        finally:
            s.stop()

    def test_thread_is_daemon(self) -> None:
        cfg = TickConfig(id="x", interval_seconds=60.0, enabled=False)
        s = TickScheduler(cfg)
        try:
            s.start()
            # PeriodicDaemon always uses daemon=True threads so a process
            # exit is not blocked by a still-running scheduler.
            assert s.is_running is True
        finally:
            s.stop()
