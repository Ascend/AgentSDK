#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""Source registry for the Agent Dashboard.

A :class:`DashboardSourceRegistry` maps ``source_name`` strings to
:class:`DashboardSource` instances. The :class:`DashboardStore` reads
from the registry to know which subsystems to poll.

There is one process-wide registry (lazily created) so subsystems can
register themselves at import time without the store having to
exist yet. Tests can build their own private registry to avoid
contaminating the global one.
"""

from __future__ import annotations

import threading
from typing import Iterator, Optional

from extensions.capabilities.dashboard_entry import (
    DashboardSource,
    normalize_source_name,
)

__all__ = [
    "DashboardSourceRegistry",
    "get_default_registry",
    "register_dashboard_source",
    "unregister_dashboard_source",
]


class DashboardSourceRegistry:
    """Thread-safe mapping of source_name -> DashboardSource.

    The registry intentionally exposes only a tiny surface
    (register/unregister/get/iter) so callers don't accidentally
    treat it as a generic dict. We need a custom class (not just a
    ``dict`` alias) so we can normalize keys at insertion time and
    guard against double-registration.
    """

    def __init__(self) -> None:
        self._sources: dict[str, DashboardSource] = {}
        self._lock = threading.RLock()

    def register(self, source: DashboardSource) -> None:
        """Register or replace a source.

        Duplicate ``source_name`` registrations replace the previous
        source — this matches Python's import system behavior and
        keeps hot-reload scenarios (dev-mode plugin reload) simple.
        """
        name = normalize_source_name(source.source_name)
        if not name:
            raise ValueError("DashboardSource.source_name must be non-empty")
        with self._lock:
            self._sources[name] = source

    def unregister(self, source_name: str) -> bool:
        """Remove a source by name. Returns True if it existed."""
        name = normalize_source_name(source_name)
        with self._lock:
            return self._sources.pop(name, None) is not None

    def get(self, source_name: str) -> Optional[DashboardSource]:
        name = normalize_source_name(source_name)
        with self._lock:
            return self._sources.get(name)

    def has(self, source_name: str) -> bool:
        name = normalize_source_name(source_name)
        with self._lock:
            return name in self._sources

    def names(self) -> list[str]:
        """Return a sorted snapshot of registered source names."""
        with self._lock:
            return sorted(self._sources.keys())

    def __iter__(self) -> Iterator[DashboardSource]:
        with self._lock:
            # Snapshot under the lock so callers can iterate without
            # holding it (sources may re-enter the registry on
            # register-time).
            return iter(list(self._sources.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._sources)

    def clear(self) -> None:
        """Drop all sources. Test-only convenience."""
        with self._lock:
            self._sources.clear()


_DEFAULT_REGISTRY: Optional[DashboardSourceRegistry] = None
_DEFAULT_REGISTRY_LOCK = threading.Lock()


def get_default_registry() -> DashboardSourceRegistry:
    """Return the process-wide :class:`DashboardSourceRegistry`."""
    global _DEFAULT_REGISTRY
    with _DEFAULT_REGISTRY_LOCK:
        if _DEFAULT_REGISTRY is None:
            _DEFAULT_REGISTRY = DashboardSourceRegistry()
        return _DEFAULT_REGISTRY


def register_dashboard_source(source: DashboardSource) -> None:
    """Register a source on the default registry. Module-level helper."""
    get_default_registry().register(source)


def unregister_dashboard_source(name: str) -> bool:
    """Unregister a source from the default registry. Returns bool."""
    return get_default_registry().unregister(name)


def reset_default_registry() -> None:
    """Drop the cached default registry. Test-only helper."""
    global _DEFAULT_REGISTRY
    with _DEFAULT_REGISTRY_LOCK:
        _DEFAULT_REGISTRY = None
