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

"""DashboardStore — the cross-system aggregator.

The store is the single point of truth for "what's happening across
all agent-loop subsystems right now". It pulls from every registered
:class:`DashboardSource`, deduplicates by ``DashboardEntry.id``,
applies the per-source TTL cache, and serves the merged snapshot to
any consumer (TUI, Visualizer, model tools).

Architecture choices, per the plan §2.3 / §6:

  * **Per-source TTL cache** — each source declares its own
    ``cache_ttl_ms``. We keep the most recent successful snapshot
    per source and only re-pull when the TTL elapses. This keeps the
    common case (10×/sec) cheap.
  * **Sink fan-out** — consumers that want push updates (Visualizer
    WebSocket, TUI live-mode) call :meth:`subscribe`. Every
    :meth:`snapshot` call that produces a *new* merged set fires all
    sinks. The merge uses :func:`_merge_snapshots` so we only
    notify when something actually changed (cheap dict-keyed diff).
  * **NDJSON archive** — every successful pull writes a copy to
    ``archive_dir/<source>.ndjson`` so Visualizer file-tail consumers
    and external audit tooling can read history. We use append-mode
    with an internal lock; writes never block the read path.
  * **Read-only by construction** — the store has zero methods that
    mutate a source. All write paths live in their owning
    subsystem's tools (``TaskCreate``, ``Goal("set")``,
    ``/goal`` command, ...).

The store is safe to share across threads; all mutating paths
take a single ``_lock`` (an RLock so sinks can re-enter).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from extensions.capabilities.dashboard_entry import (
    DashboardEntry,
    DashboardSink,
    DashboardSource,
    filter_entries,
    normalize_source_name,
)

from .source_registry import (
    DashboardSourceRegistry,
    get_default_registry,
)

logger = logging.getLogger(__name__)

__all__ = ["DashboardStore", "get_default_store"]


def _default_archive_dir() -> Path:
    """Return the default NDJSON archive directory.

    Defaults to ``~/.clawcodex/dashboard/`` but respects the
    ``CLAWCODEX_DASHBOARD_HOME`` env var so tests can redirect to a
    tmp dir without polluting the user's real home.
    """
    env_dir = os.environ.get("CLAWCODEX_DASHBOARD_HOME")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / ".clawcodex" / "dashboard"


class _SourceCache:
    """Per-source TTL cache.

    ``entries`` is the most-recent successful snapshot. ``pulled_at``
    is the monotonic-clock timestamp (``time.monotonic()``) of when
    we last invoked ``source.pull()``. We compare against
    ``source.cache_ttl_ms`` to decide whether the next
    :meth:`DashboardStore.snapshot` call needs to re-pull.
    """

    __slots__ = ("entries", "pulled_at", "last_error")

    def __init__(self) -> None:
        self.entries: list[DashboardEntry] = []
        self.pulled_at: float = 0.0
        self.last_error: Optional[str] = None

    def is_fresh(self, ttl_ms: int) -> bool:
        if ttl_ms <= 0:
            return False
        if self.pulled_at == 0.0:
            return False
        age_ms = (time.monotonic() - self.pulled_at) * 1000.0
        return age_ms < ttl_ms


class DashboardStore:
    """Aggregate read-only store. See module docstring for design."""

    def __init__(
        self,
        *,
        registry: Optional[DashboardSourceRegistry] = None,
        archive_dir: Optional[Path | str] = None,
        archive_enabled: bool = True,
    ) -> None:
        self._registry = registry if registry is not None else get_default_registry()
        self._archive_dir: Optional[Path] = None
        if archive_enabled:
            base = Path(archive_dir) if archive_dir is not None else _default_archive_dir()
            try:
                base.mkdir(parents=True, exist_ok=True)
                self._archive_dir = base
            except OSError as exc:
                logger.warning(
                    "dashboard NDJSON archive disabled (cannot create %s): %s",
                    base,
                    exc,
                )
                self._archive_dir = None
        self._cache: dict[str, _SourceCache] = {}
        self._sinks: list[DashboardSink] = []
        self._archive_lock = threading.Lock()
        self._lock = threading.RLock()
        self._last_merged_signature: Optional[str] = None

    def register_source(self, source: DashboardSource) -> None:
        """Register a source on the underlying registry."""
        self._registry.register(source)
        name = normalize_source_name(source.source_name)
        with self._lock:
            self._cache.pop(name, None)
            self._last_merged_signature = None

    def unregister_source(self, source_name: str) -> bool:
        """Remove a source by name. Returns True if it existed."""
        name = normalize_source_name(source_name)
        with self._lock:
            removed = self._registry.unregister(name)
            self._cache.pop(name, None)
            self._last_merged_signature = None
            return removed

    @property
    def registry(self) -> DashboardSourceRegistry:
        return self._registry

    def snapshot(self, *, filters: Optional[dict[str, Any]] = None) -> list[DashboardEntry]:
        """Return the merged, filtered snapshot.

        Honors per-source TTL: if a source was pulled recently we
        reuse the cached entries; otherwise we re-invoke
        ``source.pull()``. The merged list is de-duplicated by
        ``DashboardEntry.id`` (first occurrence wins; we treat the
        registry order as authoritative — first registered, first
        seen).

        Fires :meth:`subscribe`d sinks when the merged signature
        changes. The filter dict is forwarded to ``filter_entries``
        for cross-source filtering (keys: ``source``, ``status``,
        ``entry_id``).
        """
        merged = self._compute_snapshot()
        filtered = filter_entries(merged, **(filters or {}))
        return filtered

    def get_by_source(self, source: str, **filters: Any) -> list[DashboardEntry]:
        """Convenience: snapshot filtered by source name only."""
        return self.snapshot(filters={"source": source, **filters})

    def get_by_id(self, entry_id: str) -> Optional[DashboardEntry]:
        """Return the entry with the given id, or None."""
        merged = self._compute_snapshot()
        for entry in merged:
            if entry.id == entry_id:
                return entry
        return None

    def source_names(self) -> list[str]:
        """List of registered source names (sorted)."""
        return self._registry.names()

    def subscribe(self, sink: DashboardSink) -> Callable[[], None]:
        """Register a sink; returns an unsubscribe function.

        A sink is ``Callable[[list[DashboardEntry]], None]``. It is
        invoked synchronously from the thread that triggered the
        :meth:`snapshot` recompute. Sinks that need to do heavy
        work (e.g. send a WebSocket frame) should schedule it
        onto a worker thread to avoid blocking the caller.
        """
        with self._lock:
            self._sinks.append(sink)

        def _unsub() -> None:
            with self._lock:
                try:
                    self._sinks.remove(sink)
                except ValueError:
                    pass

        return _unsub

    def _compute_snapshot(self) -> list[DashboardEntry]:
        """Pull-from-cache-or-source, merge, archive, fan-out."""
        now = time.monotonic()
        with self._lock:
            for source in self._registry:
                self._refresh_source(source, now)
            merged = self._merge_caches()
            signature = self._signature(merged)
            if signature != self._last_merged_signature:
                self._last_merged_signature = signature
                sinks = list(self._sinks)
            else:
                sinks = []

        if sinks:
            for sink in sinks:
                try:
                    sink(merged)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("dashboard sink raised; dropping")
        return merged

    def _refresh_source(self, source: DashboardSource, now: float) -> None:
        """Pull from ``source`` if its cache is stale or broken."""
        name = normalize_source_name(source.source_name)
        cache = self._cache.get(name)
        if cache is None:
            cache = _SourceCache()
            self._cache[name] = cache
        ttl_ms = getattr(source, "cache_ttl_ms", 5000)
        ttl_ms = int(ttl_ms) if ttl_ms is not None else 5000
        if cache.is_fresh(ttl_ms):
            return
        try:
            entries = source.pull() or []
        except Exception as exc:
            cache.last_error = str(exc)
            logger.warning("dashboard source %s.pull() failed: %s", name, exc, exc_info=False)
            return
        if not all(isinstance(e, DashboardEntry) for e in entries):
            bad = [type(e).__name__ for e in entries if not isinstance(e, DashboardEntry)]
            cache.last_error = f"source {name} returned non-DashboardEntry: {bad}"
            logger.warning("dashboard source %s returned invalid entries: %s", name, bad)
            return
        cache.entries = list(entries)
        cache.pulled_at = time.monotonic()
        cache.last_error = None
        self._archive_source(name, entries)

    def _merge_caches(self) -> list[DashboardEntry]:
        """De-duplicate by ``id`` (first occurrence wins).

        We walk the registry in order so a higher-priority source
        can shadow a lower-priority one with the same id. This is
        useful for tests that want to inject a synthetic entry on
        top of a real one.
        """
        seen: dict[str, DashboardEntry] = {}
        for source in self._registry:
            cache = self._cache.get(normalize_source_name(source.source_name))
            if cache is None:
                continue
            for entry in cache.entries:
                if entry.id in seen:
                    continue
                seen[entry.id] = entry
        return list(seen.values())

    @staticmethod
    def _signature(entries: Iterable[DashboardEntry]) -> str:
        """Cheap content signature used to suppress sink spam.

        We don't want to JSON-dump the full snapshot per call (the
        merge cost would dominate on a 1000-entry dashboard). A
        sorted ``id|status|updated_at_ms`` triple is plenty to
        detect "anything changed" without doing field-by-field
        diffs.
        """
        parts: list[str] = []
        for e in entries:
            parts.append(f"{e.id}|{e.status}|{e.updated_at_ms}")
        return "\n".join(sorted(parts))

    def _archive_source(self, name: str, entries: list[DashboardEntry]) -> None:
        """Append entries to ``archive_dir/<name>.ndjson``.

        Best-effort: a write failure must not break the snapshot
        path. We lock around the whole write batch so lines never
        interleave on disk.
        """
        if self._archive_dir is None:
            return
        if not entries:
            return
        path = self._archive_dir / f"{name}.ndjson"
        ts = int(time.time() * 1000)
        try:
            with self._archive_lock:
                with path.open("a", encoding="utf-8") as fh:
                    for entry in entries:
                        payload = entry.to_dict()
                        payload["_archived_at_ms"] = ts
                        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                        fh.write("\n")
        except OSError as exc:
            logger.warning("dashboard archive write failed for %s: %s", path, exc)

    def _clear_caches(self) -> None:
        """Drop all per-source caches. Test-only."""
        with self._lock:
            self._cache.clear()
            self._last_merged_signature = None


_DEFAULT_STORE: Optional[DashboardStore] = None
_DEFAULT_STORE_LOCK = threading.Lock()


def get_default_store() -> DashboardStore:
    """Return the process-wide :class:`DashboardStore`.

    Construction is lazy and idempotent: the first call wires the
    store against the default source registry and the default
    archive directory. Subsequent calls return the same instance.
    """
    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = DashboardStore()
        return _DEFAULT_STORE


def reset_default_store() -> None:
    """Drop the cached default store. Test-only helper."""
    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        _DEFAULT_STORE = None
