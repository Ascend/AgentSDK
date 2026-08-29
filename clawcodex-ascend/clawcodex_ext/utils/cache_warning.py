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

"""Cache warning module with LRU eviction for memory safety.

This module provides a CacheWarning class that maintains a bounded cache of
cache warning states by source, with automatic eviction of the oldest entry
when the cache reaches capacity.

Designed to prevent memory leaks in long-running daemon/swarm sessions where
querySource might be typed as `any`, producing many unique source values.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

# Maximum number of source entries before oldest eviction.
# Set to a large value (10 000) so normal usage is unaffected while
# still bounding memory in long-running daemon/swarm scenarios.
MAX_SOURCE_ENTRIES = 10_000


@dataclass
class CacheWarningState:
    """State for a cache warning entry.

    Attributes:
        warned: Whether a warning has been issued for this source.
        count: Number of times this source has been seen.
    """

    warned: bool = False
    count: int = 0


class CacheWarning:
    """Cache warning manager with LRU-style eviction.

    Maintains an OrderedDict of cache warning states keyed by source string.
    When capacity is reached during an *insert* of a new source, the oldest
    entry (FIFO) is evicted before inserting the new entry.  Updating an
    already-existing source does NOT trigger eviction.

    Note:
        This class is not thread-safe. Callers must ensure serialized access.
    """

    def __init__(self) -> None:
        self.cache_warning_state_by_source: OrderedDict[str, CacheWarningState] = OrderedDict()

    def update(self, source: str, state: CacheWarningState) -> None:
        """Update or insert a cache warning state for a source.

        If *source* already exists, the state is replaced in-place without
        affecting the eviction order.  If *source* is new and the cache has
        reached capacity, the oldest entry is evicted first.

        Args:
            source: The source identifier for this warning state.
            state: The cache warning state to store.
        """
        # Update existing — no eviction, preserve insertion order.
        if source in self.cache_warning_state_by_source:
            self.cache_warning_state_by_source[source] = state
            return

        # Insert — evict oldest if at capacity.
        if len(self.cache_warning_state_by_source) >= MAX_SOURCE_ENTRIES:
            self.cache_warning_state_by_source.popitem(last=False)
        self.cache_warning_state_by_source[source] = state

    def get(self, source: str) -> Optional[CacheWarningState]:
        """Retrieve the cache warning state for a source.

        Args:
            source: The source identifier to look up.

        Returns:
            The CacheWarningState if found, None otherwise.
        """
        return self.cache_warning_state_by_source.get(source)

    def reset_for_test(self) -> None:
        """Clear all entries from the cache.

        Intended for test isolation. Not for use in production code.
        """
        self.cache_warning_state_by_source.clear()
