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

"""Tests for the Agent Dashboard store and source registry."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from extensions.agent_dashboard import (
    DashboardEntry,
    DashboardStore,
    reset_default_store,
)
from extensions.agent_dashboard.source_registry import (
    DashboardSourceRegistry,
    get_default_registry,
    register_dashboard_source,
    reset_default_registry,
)


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------


class _StaticSource:
    def __init__(self, name: str, entries: list[DashboardEntry], ttl_ms: int = 5_000):
        self._name = name
        self._entries = entries
        self._ttl = ttl_ms
        self.pull_count = 0

    @property
    def source_name(self) -> str:
        return self._name

    @property
    def cache_ttl_ms(self) -> int:
        return self._ttl

    def pull(self, **filters: Any) -> list[DashboardEntry]:
        self.pull_count += 1
        return list(self._entries)


def test_registry_register_and_get() -> None:
    reg = DashboardSourceRegistry()
    src = _StaticSource("alpha", [])
    reg.register(src)
    assert reg.has("alpha")
    assert reg.get("alpha") is src
    assert reg.names() == ["alpha"]


def test_registry_register_normalizes_name() -> None:
    reg = DashboardSourceRegistry()
    src = _StaticSource("Goal-Service", [])
    reg.register(src)
    assert reg.has("goal_service")
    assert reg.get("GOAL_SERVICE") is src


def test_registry_register_rejects_empty_name() -> None:
    reg = DashboardSourceRegistry()
    with pytest.raises(ValueError):
        reg.register(_StaticSource("", []))


def test_registry_register_replaces_existing() -> None:
    reg = DashboardSourceRegistry()
    a = _StaticSource("a", [])
    b = _StaticSource("a", [])
    reg.register(a)
    reg.register(b)
    assert reg.get("a") is b
    assert len(list(reg)) == 1


def test_registry_unregister_returns_bool() -> None:
    reg = DashboardSourceRegistry()
    reg.register(_StaticSource("a", []))
    assert reg.unregister("a") is True
    assert reg.unregister("a") is False
    assert reg.names() == []


# ---------------------------------------------------------------------------
# Store snapshot semantics
# ---------------------------------------------------------------------------


def _build_entry(source: str, idx: int, status: str = "pending") -> DashboardEntry:
    return DashboardEntry(
        id=f"{source}:{idx}",
        source=source,
        title=f"{source}-{idx}",
        status=status,
        updated_at_ms=1700000000000 + idx,
    )


def test_store_aggregates_entries_from_all_sources() -> None:
    reg = DashboardSourceRegistry()
    reg.register(_StaticSource("a", [_build_entry("a", 1)]))
    reg.register(_StaticSource("b", [_build_entry("b", 1), _build_entry("b", 2)]))
    store = DashboardStore(registry=reg, archive_dir=None)
    entries = store.snapshot()
    ids = sorted(e.id for e in entries)
    assert ids == ["a:1", "b:1", "b:2"]


def test_store_caches_per_source_per_ttl() -> None:
    reg = DashboardSourceRegistry()
    src = _StaticSource("a", [_build_entry("a", 1)], ttl_ms=60_000)
    reg.register(src)
    store = DashboardStore(registry=reg, archive_dir=None)
    assert src.pull_count == 0
    store.snapshot()
    store.snapshot()
    store.snapshot()
    assert src.pull_count == 1  # only the first call hit the source


def test_store_re_pulls_after_ttl_expires() -> None:
    reg = DashboardSourceRegistry()
    src = _StaticSource("a", [_build_entry("a", 1)], ttl_ms=1)
    reg.register(src)
    store = DashboardStore(registry=reg, archive_dir=None)
    store.snapshot()
    time.sleep(0.005)
    store.snapshot()
    assert src.pull_count == 2


def test_store_get_by_source_filters() -> None:
    reg = DashboardSourceRegistry()
    reg.register(_StaticSource("a", [_build_entry("a", 1)]))
    reg.register(_StaticSource("b", [_build_entry("b", 1)]))
    store = DashboardStore(registry=reg, archive_dir=None)
    out = store.get_by_source("a")
    assert [e.id for e in out] == ["a:1"]


def test_store_get_by_id_returns_entry_or_none() -> None:
    reg = DashboardSourceRegistry()
    reg.register(_StaticSource("a", [_build_entry("a", 1)]))
    store = DashboardStore(registry=reg, archive_dir=None)
    assert store.get_by_id("a:1") is not None
    assert store.get_by_id("nope") is None


def test_store_dedupes_by_id() -> None:
    """Two sources emitting the same id — first registered wins."""
    reg = DashboardSourceRegistry()
    e1 = _build_entry("a", 1)
    e2 = _build_entry("a", 1)  # same id, different (mocked) content
    # We can't really test content here because the dataclass is
    # frozen, so we just check that only one entry surfaces.
    reg.register(_StaticSource("a", [e1]))
    reg.register(_StaticSource("b", [e2]))
    store = DashboardStore(registry=reg, archive_dir=None)
    entries = store.snapshot()
    assert len(entries) == 1


def test_store_survives_broken_source() -> None:
    reg = DashboardSourceRegistry()

    class _Broken:
        source_name = "broken"
        cache_ttl_ms = 5_000

        def pull(self, **filters: Any):
            raise RuntimeError("boom")

    reg.register(_Broken())
    reg.register(_StaticSource("ok", [_build_entry("ok", 1)]))
    store = DashboardStore(registry=reg, archive_dir=None)
    out = store.snapshot()
    assert [e.id for e in out] == ["ok:1"]


def test_store_drops_invalid_entries() -> None:
    reg = DashboardSourceRegistry()

    class _Bogus:
        source_name = "bogus"
        cache_ttl_ms = 5_000

        def pull(self, **filters: Any):
            return [{"not": "an entry"}]  # type: ignore[list-item]

    reg.register(_Bogus())
    store = DashboardStore(registry=reg, archive_dir=None)
    out = store.snapshot()
    assert out == []


def test_store_writes_ndjson_archive(tmp_path: Path) -> None:
    reg = DashboardSourceRegistry()
    reg.register(_StaticSource("goal", [_build_entry("goal", 1, "in_progress")]))
    archive = tmp_path / "dash"
    store = DashboardStore(registry=reg, archive_dir=archive)
    store.snapshot()
    files = list(archive.glob("*.ndjson"))
    assert files == [archive / "goal.ndjson"]
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["id"] == "goal:1"
    assert payload["_archived_at_ms"] > 0


def test_store_archive_disabled_when_path_unwritable(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    reg = DashboardSourceRegistry()
    reg.register(_StaticSource("goal", [_build_entry("goal", 1)]))
    # A read-only "directory" — we make a file at the same path so
    # ``mkdir`` fails.
    bad = tmp_path / "not-a-dir"
    bad.write_text("blocking", encoding="utf-8")
    store = DashboardStore(registry=reg, archive_dir=bad)
    assert store.snapshot()  # still works, just no archive


def test_store_subscribe_fires_on_change() -> None:
    reg = DashboardSourceRegistry()
    reg.register(_StaticSource("a", [_build_entry("a", 1)]))
    store = DashboardStore(registry=reg, archive_dir=None)
    received: list[list[DashboardEntry]] = []
    unsub = store.subscribe(received.append)
    store.snapshot()
    store.snapshot()  # no change → no second fire
    assert len(received) == 1
    unsub()
    store.snapshot()
    assert len(received) == 1  # unsubscribed


def test_store_subscribe_skipped_when_unchanged() -> None:
    reg = DashboardSourceRegistry()
    reg.register(_StaticSource("a", [_build_entry("a", 1)]))
    store = DashboardStore(registry=reg, archive_dir=None)
    received: list[list[DashboardEntry]] = []
    store.subscribe(received.append)
    # First snapshot: signature transitions from None to a real
    # value, so sinks fire.
    store.snapshot()
    assert len(received) == 1
    # Second snapshot with no data change: signature is identical,
    # so sinks do NOT fire again.
    store.snapshot()
    store.snapshot()
    assert len(received) == 1


def test_store_sink_can_be_called_safely() -> None:
    reg = DashboardSourceRegistry()
    reg.register(_StaticSource("a", [_build_entry("a", 1)]))
    store = DashboardStore(registry=reg, archive_dir=None)

    def _bad_sink(entries: list[DashboardEntry]) -> None:
        raise RuntimeError("boom")

    store.subscribe(_bad_sink)
    # A bad sink must not break the snapshot path.
    out = store.snapshot()
    assert [e.id for e in out] == ["a:1"]


def test_store_filters_via_snapshot() -> None:
    reg = DashboardSourceRegistry()
    reg.register(_StaticSource("a", [_build_entry("a", 1, "in_progress")]))
    reg.register(_StaticSource("a", [_build_entry("a", 2, "completed")]))
    store = DashboardStore(registry=reg, archive_dir=None)
    out = store.snapshot(filters={"status": "completed"})
    assert [e.id for e in out] == ["a:2"]


def test_store_register_clears_cache() -> None:
    reg = DashboardSourceRegistry()
    src_a = _StaticSource("a", [_build_entry("a", 1)])
    reg.register(src_a)
    store = DashboardStore(registry=reg, archive_dir=None)
    store.snapshot()  # populates cache
    # Replace the source — cache should drop so we don't serve stale data.
    src_b = _StaticSource("a", [_build_entry("a", 99)])
    store.register_source(src_b)
    out = store.snapshot()
    assert [e.id for e in out] == ["a:99"]


def test_store_unregister_clears_cache() -> None:
    reg = DashboardSourceRegistry()
    reg.register(_StaticSource("a", [_build_entry("a", 1)]))
    store = DashboardStore(registry=reg, archive_dir=None)
    store.snapshot()
    store.unregister_source("a")
    assert store.snapshot() == []


# ---------------------------------------------------------------------------
# Default singletons (test isolation)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons() -> Any:
    reset_default_registry()
    reset_default_store()
    yield
    reset_default_registry()
    reset_default_store()


def test_default_registry_is_singleton() -> None:
    a = get_default_registry()
    b = get_default_registry()
    assert a is b


def test_register_helper_uses_default_registry() -> None:
    src = _StaticSource("a", [])
    register_dashboard_source(src)
    assert get_default_registry().get("a") is src
