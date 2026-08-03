"""F-22-H-1: mtime-based incremental read cache for durable cron tasks.

Pins the contract documented in
``docs/feature_plan/05-cron-system/f-22-cron-execution.md`` §Phase H:

- TTL window prevents repeat reads within the same scheduler tick.
- mtime unchanged → skip re-read.
- mtime changed → trigger full re-read.
- File missing → fall back to ``read_cron_tasks`` (returns empty list).
- Per-workspace path isolation in the module-level cache dict.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from clawcodex_ext.cron_system import tasks as tasks_module
from clawcodex_ext.cron_system.tasks import (
    _MtimeCache,
    read_cron_tasks,
    read_cron_tasks_cached,
    write_cron_tasks,
    add_cron_task,
)
from clawcodex_ext.cron_system.models import CronTask


@pytest.fixture(autouse=True)
def _clear_global_mtime_cache():
    """Reset the module-level cache between tests to avoid cross-test pollution."""
    _MtimeCache.clear()
    yield
    _MtimeCache.clear()


@pytest.fixture
def _short_ttl(monkeypatch):
    """Force the TTL window to zero so mtime checks take effect immediately.

    The production TTL is 1.0s (aligned with the scheduler's 1-second tick),
    which is too slow for pytest. For unit tests that need to observe the
    ``mtime changed → re-read`` branch without sleeping, we drop the TTL to
    zero so the only gating signal is the file's mtime.
    """
    monkeypatch.setattr(tasks_module, "_MTIME_CACHE_TTL", 0.0)
    yield


def _write_task_payload(workspace: Path, task_id: str, prompt: str = "ping") -> None:
    """Write a single canonical task payload bypassing ``add_cron_task``."""
    workspace.mkdir(parents=True, exist_ok=True)
    cron_path = workspace / ".clawcodex" / "cron" / "scheduled_tasks.json"
    cron_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "tasks": [
            {
                "id": task_id,
                "cron": "*/5 * * * *",
                "prompt": prompt,
                "recurring": True,
                "durable": True,
                "created_at": 1_000,
                "updated_at": 1_000,
            }
        ],
    }
    cron_path.write_text(json.dumps(payload), encoding="utf-8")


def test_first_read_populates_cache(tmp_path):
    """Cache miss path: full read populates ``_MtimeCache``."""
    _write_task_payload(tmp_path, "t-001")
    assert read_cron_tasks_cached(tmp_path) == list(read_cron_tasks(tmp_path))
    # Cache entry should be present after the read.
    cached_path = tmp_path / ".clawcodex" / "cron" / "scheduled_tasks.json"
    assert cached_path in _MtimeCache


def test_ttl_window_skips_underlying_read(tmp_path):
    """Within TTL + same mtime → ``read_cron_tasks`` is not called again."""
    _write_task_payload(tmp_path, "t-ttl")
    read_cron_tasks_cached(tmp_path)  # warm cache
    with patch.object(tasks_module, "read_cron_tasks", wraps=read_cron_tasks) as spy:
        # Second call inside TTL should hit cache without invoking the full reader.
        result = read_cron_tasks_cached(tmp_path)
    assert spy.call_count == 0
    assert len(result) == 1
    assert result[0].id == "t-ttl"


def test_mtime_change_triggers_reread(tmp_path, _short_ttl):
    """``os.utime`` mtime bump → next call must invoke full reader.

    TTL is forced to zero so the mtime check is the sole gating signal.
    """
    cron_path = tmp_path / ".clawcodex" / "cron" / "scheduled_tasks.json"
    _write_task_payload(tmp_path, "t-old")
    read_cron_tasks_cached(tmp_path)
    # Bump mtime and rewrite payload so content differs.
    time.sleep(0.01)  # ensure mtime resolution gap on slow filesystems
    _write_task_payload(tmp_path, "t-new")
    os.utime(cron_path, (time.time(), time.time()))
    with patch.object(tasks_module, "read_cron_tasks", wraps=read_cron_tasks) as spy:
        result = read_cron_tasks_cached(tmp_path)
    assert spy.call_count == 1
    assert [t.id for t in result] == ["t-new"]


def test_ttl_expiry_with_unchanged_mtime_returns_cache(tmp_path):
    """Past TTL but file mtime unchanged → still serve cached, no re-read.

    Documents the mtime-as-truth policy: TTL is a coarse-grained tick
    deduplication, but the file mtime is the authoritative "has the file
    changed?" signal. When the file has not been touched, even a TTL
    expiry must not trigger a full re-read — the cached entry is fresh.
    """
    _write_task_payload(tmp_path, "t-ttl-unchanged")
    read_cron_tasks_cached(tmp_path)
    cron_path = tmp_path / ".clawcodex" / "cron" / "scheduled_tasks.json"
    cached_at, cached_mtime, cached_tasks = _MtimeCache[cron_path]
    # Rewind cached_at by 2 seconds — TTL gate is now wide open.
    _MtimeCache[cron_path] = (cached_at - 2.0, cached_mtime, cached_tasks)
    with patch.object(tasks_module, "read_cron_tasks", wraps=read_cron_tasks) as spy:
        result = read_cron_tasks_cached(tmp_path)
    # TTL expired but mtime unchanged → still cached, no re-read.
    assert spy.call_count == 0
    assert result == cached_tasks


def test_missing_file_returns_empty_list_without_caching(tmp_path):
    """No file → fall back to ``read_cron_tasks`` (returns []) and do not cache the miss.

    A negative result must not poison the cache so the next mtime-valid read
    works.
    """
    # Pre-create the parent so ``stat`` raises only on the missing file itself.
    cron_dir = tmp_path / ".clawcodex" / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    assert read_cron_tasks_cached(tmp_path) == []
    cron_path = cron_dir / "scheduled_tasks.json"
    # A negative entry should not be cached.
    assert cron_path not in _MtimeCache
    # Now write a real file and ensure the next call sees it.
    _write_task_payload(tmp_path, "t-after-miss")
    result = read_cron_tasks_cached(tmp_path)
    assert [t.id for t in result] == ["t-after-miss"]


def test_per_workspace_isolation(tmp_path):
    """Two workspaces get separate cache entries."""
    ws_a = tmp_path / "a"
    ws_b = tmp_path / "b"
    _write_task_payload(ws_a, "t-a")
    _write_task_payload(ws_b, "t-b")
    res_a = read_cron_tasks_cached(ws_a)
    res_b = read_cron_tasks_cached(ws_b)
    assert [t.id for t in res_a] == ["t-a"]
    assert [t.id for t in res_b] == ["t-b"]
    path_a = ws_a / ".clawcodex" / "cron" / "scheduled_tasks.json"
    path_b = ws_b / ".clawcodex" / "cron" / "scheduled_tasks.json"
    assert path_a in _MtimeCache
    assert path_b in _MtimeCache
    assert _MtimeCache[path_a][1] is not _MtimeCache[path_b][1]


def test_add_cron_task_invalidates_cache_via_mtime(tmp_path, _short_ttl):
    """``add_cron_task`` writes via ``os.replace``; next read sees the new task.

    TTL is forced to zero so the cache observes the new mtime on the very
    next read rather than waiting out a 1-second wall clock.
    """
    add_cron_task(tmp_path, cron="*/5 * * * *", prompt="first", created_at=1_000)
    read_cron_tasks_cached(tmp_path)  # warm
    time.sleep(0.01)
    add_cron_task(tmp_path, cron="*/10 * * * *", prompt="second", created_at=2_000)
    result = read_cron_tasks_cached(tmp_path)
    prompts = sorted(t.prompt for t in result)
    assert prompts == ["first", "second"]


def test_write_path_does_not_eagerly_invalidate_cache(tmp_path, _short_ttl):
    """Cache relies on mtime changes; explicit invalidation is not required.

    Documents §3.4 H risk: ``write_cron_tasks()`` does not actively invalidate
    the cache. Subsequent reads detect the mtime change and re-read.
    """
    cron_path = tmp_path / ".clawcodex" / "cron" / "scheduled_tasks.json"
    cron_path.parent.mkdir(parents=True, exist_ok=True)
    # Seed an initial task via write_cron_tasks
    initial = [
        CronTask(
            id="t-init",
            cron="*/5 * * * *",
            prompt="init",
            recurring=True,
            durable=True,
            created_at=1_000,
            updated_at=1_000,
        )
    ]
    write_cron_tasks(tmp_path, initial)
    read_cron_tasks_cached(tmp_path)  # warm
    # Cache shape: (cached_at_monotonic, cached_file_mtime, tasks)
    assert _MtimeCache[cron_path][2][0].id == "t-init"

    time.sleep(0.01)
    # Replace with a different task; do NOT touch the cache directly.
    next_task = [
        CronTask(
            id="t-next",
            cron="*/5 * * * *",
            prompt="next",
            recurring=True,
            durable=True,
            created_at=2_000,
            updated_at=2_000,
        )
    ]
    write_cron_tasks(tmp_path, next_task)
    # The cached entry still references the old list (mtime check is what
    # gates the next read). With TTL forced to zero, the next read sees
    # the new mtime and refreshes.
    refreshed = read_cron_tasks_cached(tmp_path)
    assert refreshed[0].id == "t-next"
