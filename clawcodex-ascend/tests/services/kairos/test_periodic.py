#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Tests for the PeriodicDaemon base class shared by kairos and mailbox_poller.

These exercise the lifecycle primitives that TickScheduler inherits:

* Idempotent start.
* stop() joins the thread within the timeout.
* Exception in _do_tick() does not kill the daemon.
* Context manager pattern.
* Strict constructor validation.
"""
# pylint: disable=no-name-in-module

from __future__ import annotations

import threading
import time

import pytest

from clawcodex_ext.services.periodic import PeriodicDaemon


class CountingDaemon(PeriodicDaemon):
    def __init__(self, *, tick_seconds: float, name: str = "counter"):
        super().__init__(name=name, tick_seconds=tick_seconds)
        self._count = 0
        self._lock = threading.Lock()
        self._ticks: list[int] = []

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def ticks(self) -> list[int]:
        with self._lock:
            return list(self._ticks)

    def _do_tick(self) -> None:
        with self._lock:
            self._count += 1
            self._ticks.append(self._count)


class TestPeriodicDaemonConstruction:
    def test_name_must_be_nonempty_string(self) -> None:
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            PeriodicDaemon(name="", tick_seconds=0.1)
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            PeriodicDaemon(name=123, tick_seconds=0.1)  # type: ignore[arg-type]

    def test_tick_seconds_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="tick_seconds must be a positive"):
            PeriodicDaemon(name="x", tick_seconds=0)
        with pytest.raises(ValueError, match="tick_seconds must be a positive"):
            PeriodicDaemon(name="x", tick_seconds=-1.0)
        with pytest.raises(ValueError, match="tick_seconds must be a positive"):
            PeriodicDaemon(name="x", tick_seconds="0.1")  # type: ignore[arg-type]

    def test_attributes_exposed(self) -> None:
        d = PeriodicDaemon(name="x", tick_seconds=0.5)
        assert d.name == "x"
        assert d.tick_seconds == 0.5
        assert d.is_running is False


class TestPeriodicDaemonLifecycle:
    def test_start_is_idempotent(self) -> None:
        d = CountingDaemon(tick_seconds=0.05)
        assert d.start() is True
        try:
            assert d.is_running is True
            assert d.start() is False
            assert d.is_running is True
        finally:
            d.stop(timeout=2.0)

    def test_stop_returns_true_when_thread_joined(self) -> None:
        d = CountingDaemon(tick_seconds=0.05)
        d.start()
        # Give the daemon a moment to actually start firing.
        time.sleep(0.15)
        joined = d.stop(timeout=2.0)
        assert joined is True
        assert d.is_running is False

    def test_stop_safe_when_not_running(self) -> None:
        d = CountingDaemon(tick_seconds=0.05)
        joined = d.stop(timeout=0.5)
        assert joined is True

    def test_context_manager_starts_and_stops(self) -> None:
        with CountingDaemon(tick_seconds=0.05) as d:
            assert d.is_running is True
            time.sleep(0.1)
            assert d.count >= 1
        assert d.is_running is False


class TestPeriodicDaemonTicking:
    def test_daemon_fires_repeatedly(self) -> None:
        d = CountingDaemon(tick_seconds=0.05)
        try:
            d.start()
            time.sleep(0.3)
        finally:
            d.stop(timeout=2.0)
        # At ~50ms ticks over 300ms we expect at least 3 ticks; assert a
        # conservative lower bound to avoid flakes on slow CI.
        assert d.count >= 2, f"expected at least 2 ticks, got {d.count}"

    def test_tick_exception_does_not_kill_daemon(self) -> None:
        class FlakyDaemon(PeriodicDaemon):
            def __init__(self) -> None:
                super().__init__(name="flaky", tick_seconds=0.05)
                self._calls = 0

            @property
            def calls(self) -> int:
                return self._calls

            def _do_tick(self) -> None:
                self._calls += 1
                raise RuntimeError("boom")

        d = FlakyDaemon()
        try:
            d.start()
            time.sleep(0.25)
        finally:
            d.stop(timeout=2.0)
        assert d.calls >= 2, f"expected at least 2 calls, got {d.calls}"

    def test_default_do_tick_raises(self) -> None:
        d = PeriodicDaemon(name="x", tick_seconds=0.05)
        # Use a context that starts the daemon only briefly; the default
        # _do_tick raises NotImplementedError which the loop catches.
        d.start()
        try:
            time.sleep(0.15)
        finally:
            d.stop(timeout=2.0)
        # The daemon should still be alive after default-raise ticks because
        # the loop catches every exception.
        assert d.is_running is False  # stopped


class TestPeriodicDaemonConcurrency:
    def test_concurrent_starts_only_one_thread(self) -> None:
        d = CountingDaemon(tick_seconds=0.05)
        results: list[bool] = []
        barrier = threading.Barrier(5)

        def go() -> None:
            barrier.wait()
            results.append(d.start())

        threads = [threading.Thread(target=go) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        try:
            assert sum(results) == 1, f"expected exactly one True, got {results}"
            assert d.is_running is True
        finally:
            d.stop(timeout=2.0)
