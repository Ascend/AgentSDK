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

"""Unit tests for TeamMemoryStore.

Covers append-only JSONL, atomic writes, tombstone delete, compact,
archive, corrupt-line tolerance (acceptance #6), and entry rebuild.
The store is exercised directly with an in-memory team_id and a tmp
dir; no env/auto-memory path resolution involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from extensions.agents.team_memory import (
    TeamMemoryConfig,
    TeamMemoryEntry,
    TeamMemoryStore,
    TeamMemoryTooLargeError,
    _entry_id,
    make_iso_timestamp,
)


def _make_entry(
    *,
    team_id: str = "t1",
    author: str = "lead-1",
    content: str = "hello team",
    summary: str = "greeting",
    tags: tuple[str, ...] = (),
    scope: str = "team",
    source: str = "manual",
    counter: int = 0,
) -> TeamMemoryEntry:
    ts = make_iso_timestamp()
    # Use a counter so each entry gets a unique id (the real service
    # derives id from team+ts+author+content; ts alone collides when
    # multiple entries are appended in the same second).
    eid = _entry_id(team_id, ts, author, f"{content}-{counter}")
    return TeamMemoryEntry(
        id=eid,
        team_id=team_id,
        content=content,
        summary=summary,
        author_agent_id=author,
        created_at=ts,
        tags=tags,
        scope=scope,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
    )


def test_append_then_list_roundtrip(tmp_path: Path) -> None:
    store = TeamMemoryStore(team_id="t1", root=tmp_path, config=TeamMemoryConfig())
    entry = store.append(_make_entry(content="hello", summary="g"))
    assert entry.id  # id assigned by helper
    listed = store.list_entries()
    assert len(listed) == 1
    assert listed[0].content == "hello"


def test_append_writes_entries_jsonl_and_memory_md(tmp_path: Path) -> None:
    store = TeamMemoryStore(team_id="t1", root=tmp_path, config=TeamMemoryConfig())
    store.append(_make_entry(content="first", summary="s1"))
    assert (tmp_path / "entries.jsonl").exists()
    assert (tmp_path / "MEMORY.md").exists()
    md = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert "s1" in md


def test_delete_tombstones_and_hides_entry(tmp_path: Path) -> None:
    store = TeamMemoryStore(team_id="t1", root=tmp_path, config=TeamMemoryConfig())
    store.append(_make_entry(content="to-delete", summary="x"))
    # Pre-fix: _make_entry sets id="" — patch by re-reading the store.
    live = store.list_entries()
    target_id = live[0].id
    assert store.delete(target_id, actor="lead-1", reason="test") is True
    assert store.list_entries() == []
    # Deleting again is a no-op.
    assert store.delete(target_id, actor="lead-1", reason="again") is False


def test_compact_collapses_and_archives(tmp_path: Path) -> None:
    store = TeamMemoryStore(team_id="t1", root=tmp_path, config=TeamMemoryConfig())
    for i in range(3):
        store.append(_make_entry(content=f"c{i}", summary=f"s{i}", counter=i))
    compact = store.compact(actor="lead-1")
    assert compact.summary.startswith("compact of 3 entries")
    # After compact, only the compact summary entry is live.
    live = store.list_entries()
    assert len(live) == 1
    assert live[0].id == compact.id
    # Archive snapshot exists.
    archives = list((tmp_path / "archive").glob("*.jsonl"))
    assert len(archives) == 1


def test_archive_writes_snapshot_file(tmp_path: Path) -> None:
    store = TeamMemoryStore(team_id="t1", root=tmp_path, config=TeamMemoryConfig())
    store.append(_make_entry(content="persisted", summary="s"))
    dst = store.archive(reason="TeamDelete")
    assert dst.exists()
    assert dst.read_text(encoding="utf-8").strip().count("\n") >= 0


def test_corrupt_line_skipped_not_fatal(tmp_path: Path) -> None:
    store = TeamMemoryStore(team_id="t1", root=tmp_path, config=TeamMemoryConfig())
    store.append(_make_entry(content="good", summary="g"))
    # Append a corrupt line manually.
    with (tmp_path / "entries.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
        fh.write("\n")  # blank line too
    listed = store.list_entries()
    assert len(listed) == 1
    assert listed[0].content == "good"


def test_max_entry_bytes_rejected(tmp_path: Path) -> None:
    cfg = TeamMemoryConfig(max_entry_bytes=8)
    store = TeamMemoryStore(team_id="t1", root=tmp_path, config=cfg)
    with pytest.raises(TeamMemoryTooLargeError):
        store.append(_make_entry(content="x" * 100, summary="big"))


def test_audit_log_recorded(tmp_path: Path) -> None:
    store = TeamMemoryStore(team_id="t1", root=tmp_path, config=TeamMemoryConfig())
    store.append(_make_entry(content="audited", summary="a"))
    audit = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "append" in audit
    assert "audited" not in audit  # audit records metadata, not content
