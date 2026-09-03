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

"""Tests for cacheWarning capacity limit."""

from __future__ import annotations


from src.utils.cache_warning import (
    CacheWarning,
    CacheWarningState,
    MAX_SOURCE_ENTRIES,
)


class TestCacheWarningState:
    """Tests for CacheWarningState dataclass."""

    def test_default_values(self):
        state = CacheWarningState()
        assert state.warned is False
        assert state.count == 0

    def test_custom_values(self):
        state = CacheWarningState(warned=True, count=42)
        assert state.warned is True
        assert state.count == 42


class TestCacheWarning:
    """Tests for CacheWarning class with LRU eviction."""

    def test_init_empty(self):
        cache = CacheWarning()
        assert len(cache.cache_warning_state_by_source) == 0

    def test_update_and_get(self):
        cache = CacheWarning()
        state = CacheWarningState(warned=True, count=1)
        cache.update("source_a", state)

        result = cache.get("source_a")
        assert result is not None
        assert result.warned is True
        assert result.count == 1

    def test_get_missing_returns_none(self):
        cache = CacheWarning()
        result = cache.get("nonexistent")
        assert result is None

    def test_update_overwrites_existing(self):
        cache = CacheWarning()
        cache.update("source_a", CacheWarningState(warned=False, count=1))
        cache.update("source_a", CacheWarningState(warned=True, count=2))

        result = cache.get("source_a")
        assert result is not None
        assert result.warned is True
        assert result.count == 2

    def test_eviction_at_capacity(self):
        cache = CacheWarning()

        # Fill to exactly MAX_SOURCE_ENTRIES
        for i in range(MAX_SOURCE_ENTRIES):
            cache.update(f"source_{i}", CacheWarningState(warned=False, count=i))

        assert len(cache.cache_warning_state_by_source) == MAX_SOURCE_ENTRIES
        assert cache.get("source_0") is not None  # Oldest still present

        # Adding one more should evict the oldest (source_0)
        cache.update("source_extra", CacheWarningState(warned=False, count=100))

        assert len(cache.cache_warning_state_by_source) == MAX_SOURCE_ENTRIES
        assert cache.get("source_0") is None  # Evicted
        assert cache.get("source_extra") is not None  # New entry present

    def test_boundary_exactly_at_capacity_no_eviction(self):
        """At exactly the configured capacity, no eviction should occur."""
        cache = CacheWarning()

        for i in range(MAX_SOURCE_ENTRIES):
            cache.update(f"source_{i}", CacheWarningState(warned=False, count=i))

        assert len(cache.cache_warning_state_by_source) == MAX_SOURCE_ENTRIES
        assert cache.get("source_0") is not None
        assert cache.get(f"source_{MAX_SOURCE_ENTRIES - 1}") is not None

    def test_boundary_capacity_plus_one_triggers_eviction(self):
        """The first entry beyond capacity should evict the oldest."""
        cache = CacheWarning()

        for i in range(MAX_SOURCE_ENTRIES):
            cache.update(f"source_{i}", CacheWarningState(warned=False, count=i))

        extra_source = f"source_{MAX_SOURCE_ENTRIES}"
        cache.update(
            extra_source,
            CacheWarningState(warned=False, count=MAX_SOURCE_ENTRIES),
        )

        assert len(cache.cache_warning_state_by_source) == MAX_SOURCE_ENTRIES
        assert cache.get("source_0") is None  # First entry evicted
        assert cache.get("source_1") is not None  # Second entry still present
        assert cache.get(extra_source) is not None  # New entry present

    def test_reset_for_test(self):
        cache = CacheWarning()
        cache.update("source_a", CacheWarningState(warned=True, count=1))
        cache.update("source_b", CacheWarningState(warned=True, count=2))

        assert len(cache.cache_warning_state_by_source) == 2

        cache.reset_for_test()

        assert len(cache.cache_warning_state_by_source) == 0
        assert cache.get("source_a") is None
        assert cache.get("source_b") is None

    def test_multiple_evictions(self):
        """Verify eviction happens correctly over multiple cycles."""
        cache = CacheWarning()

        overflow = 50
        for i in range(MAX_SOURCE_ENTRIES + overflow):
            cache.update(f"source_{i}", CacheWarningState(warned=False, count=i))

        # Should have exactly MAX_SOURCE_ENTRIES entries
        assert len(cache.cache_warning_state_by_source) == MAX_SOURCE_ENTRIES

        for i in range(overflow):
            assert cache.get(f"source_{i}") is None

        for i in range(overflow, MAX_SOURCE_ENTRIES + overflow):
            assert cache.get(f"source_{i}") is not None

    def test_fifo_order(self):
        """Verify entries are evicted in insertion order (FIFO)."""
        cache = CacheWarning()

        cache.update("first", CacheWarningState(warned=False, count=1))
        cache.update("second", CacheWarningState(warned=False, count=2))
        cache.update("third", CacheWarningState(warned=False, count=3))

        # Add entries to trigger eviction
        for i in range(MAX_SOURCE_ENTRIES):
            cache.update(f"extra_{i}", CacheWarningState(warned=False, count=i))

        # First three entries should be evicted in order
        assert cache.get("first") is None
        assert cache.get("second") is None
        assert cache.get("third") is None

        # Most recent entries should still be present
        assert cache.get(f"extra_{MAX_SOURCE_ENTRIES - 1}") is not None


class TestMaxSourceEntries:
    """Tests for MAX_SOURCE_ENTRIES constant."""

    def test_max_source_entries_value(self):
        """Keep normal usage unbounded while capping daemon memory growth."""
        assert MAX_SOURCE_ENTRIES == 10_000
