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

"""Tests for :class:`DailyLogWriter`.

The writer is a thin, thread-safe, append-only file helper. The
canonical daily-log path resolution lives in
:func:`src.memdir.paths.get_auto_mem_daily_log_path` and is exercised
in ``test_daily_log_path_integration`` below.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from src.services.kairos import (
    DailyLogEntry,
    DailyLogWriter,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestDailyLogWriterConstruction:
    def test_accepts_str_and_path(self, tmp_path: Path) -> None:
        DailyLogWriter(tmp_path / "log.md")
        DailyLogWriter(str(tmp_path / "log.md"))
        assert (tmp_path / "log.md").parent.exists()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "log.md"
        DailyLogWriter(deep)
        assert deep.parent.exists()

    def test_path_property(self, tmp_path: Path) -> None:
        target = tmp_path / "log.md"
        w = DailyLogWriter(target)
        assert w.path == target


# ---------------------------------------------------------------------------
# Append
# ---------------------------------------------------------------------------


class TestDailyLogWriterAppend:
    def test_append_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "log.md"
        w = DailyLogWriter(path)
        assert not w.exists()
        n = w.append(DailyLogEntry(timestamp="2026-06-19T10:00:00", body="hi"))
        assert n > 0
        assert w.exists()

    def test_append_writes_markdown(self, tmp_path: Path) -> None:
        path = tmp_path / "log.md"
        w = DailyLogWriter(path)
        w.append(
            DailyLogEntry(
                timestamp="2026-06-19T10:00:00",
                body="first",
                tags=("init",),
            )
        )
        content = w.read()
        assert content.startswith("## 2026-06-19T10:00:00")
        assert "first" in content
        assert "#init" in content

    def test_append_multiple_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "log.md"
        w = DailyLogWriter(path)
        w.append(DailyLogEntry(timestamp="t1", body="a"))
        w.append(DailyLogEntry(timestamp="t2", body="b"))
        w.append(DailyLogEntry(timestamp="t3", body="c"))
        content = w.read()
        assert content.count("## ") == 3

    def test_append_rejects_non_entry(self, tmp_path: Path) -> None:
        w = DailyLogWriter(tmp_path / "log.md")
        with pytest.raises(TypeError, match="DailyLogEntry"):
            w.append("not an entry")  # type: ignore[arg-type]

    def test_concurrent_appends_do_not_interleave(self, tmp_path: Path) -> None:
        path = tmp_path / "log.md"
        w = DailyLogWriter(path)
        barrier = threading.Barrier(8)

        def worker(i: int) -> None:
            barrier.wait()
            w.append(DailyLogEntry(timestamp=f"t{i}", body=f"body-{i:02d}"))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        content = w.read()
        assert content.count("## ") == 8
        # Every body should appear intact (no partial interleaving).
        for i in range(8):
            assert f"body-{i:02d}" in content


# ---------------------------------------------------------------------------
# Read / exists / delete
# ---------------------------------------------------------------------------


class TestDailyLogWriterRead:
    def test_read_empty_when_missing(self, tmp_path: Path) -> None:
        w = DailyLogWriter(tmp_path / "log.md")
        assert w.read() == ""

    def test_exists_reflects_filesystem(self, tmp_path: Path) -> None:
        path = tmp_path / "log.md"
        w = DailyLogWriter(path)
        assert w.exists() is False
        w.append(DailyLogEntry(timestamp="t", body="x"))
        assert w.exists() is True


class TestDailyLogWriterDelete:
    def test_delete_removes_file(self, tmp_path: Path) -> None:
        path = tmp_path / "log.md"
        w = DailyLogWriter(path)
        w.append(DailyLogEntry(timestamp="t", body="x"))
        assert w.exists() is True
        w.delete()
        assert w.exists() is False

    def test_delete_missing_file_is_noop(self, tmp_path: Path) -> None:
        path = tmp_path / "log.md"
        w = DailyLogWriter(path)
        # Must not raise even though the file doesn't exist.
        w.delete()
        assert w.exists() is False


# ---------------------------------------------------------------------------
# Integration with the canonical daily-log path helper
# ---------------------------------------------------------------------------


class TestDailyLogPathIntegration:
    """Demonstrates the refactor's "use memdir.paths for canonical path"
    contract. The kairos service layer no longer ships its own
    default_daily_log_path; callers compose the canonical helper
    directly with the writer.
    """

    def test_compose_with_memdir_helper(self, tmp_path: Path, monkeypatch) -> None:
        from datetime import date as _date

        from src.memdir import paths as memdir_paths

        target_date = _date(2026, 6, 19)
        canonical = memdir_paths.get_auto_mem_daily_log_path(date=target_date)
        # Redirect get_auto_mem_path via env override so we don't write to
        # the user's real auto-memory directory.
        monkeypatch.setenv("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", str(tmp_path))
        # Re-fetch with the override active.
        canonical = memdir_paths.get_auto_mem_daily_log_path(date=target_date)

        w = DailyLogWriter(canonical)
        w.append(
            DailyLogEntry(
                timestamp="2026-06-19T10:00:00",
                body="hello",
            )
        )
        assert w.exists()
        # The canonical path lives under <root>/logs/YYYY/MM/YYYY-MM-DD.md.
        rel = Path(canonical).relative_to(tmp_path)
        assert rel.parts == ("logs", "2026", "06", "2026-06-19.md")
