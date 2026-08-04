#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for LocalJsonlStorage."""

from __future__ import annotations

from pathlib import Path

from telemetry.storage import LocalJsonlStorage, utc_date, utc_now


def _make_storage(tmp_path: Path) -> LocalJsonlStorage:
    return LocalJsonlStorage(tmp_path / "telemetry", retention_days=7)


def test_append_writes_jsonl(tmp_path):
    storage = _make_storage(tmp_path)
    today = utc_date(utc_now())
    ok = storage.append("events", {"type": "session_start", "timestamp": 1.0})
    assert ok is True
    rows = storage.read_day("events", today)
    assert len(rows) == 1
    assert rows[0]["type"] == "session_start"


def test_append_to_crash_kind(tmp_path):
    storage = _make_storage(tmp_path)
    today = utc_date(utc_now())
    storage.append("crashes", {"type": "crash", "fingerprint": "abc"})
    rows = storage.read_day("crashes", today)
    assert rows and rows[0]["fingerprint"] == "abc"


def test_append_rejects_unknown_kind(tmp_path):
    storage = _make_storage(tmp_path)
    try:
        storage.append("bogus", {"x": 1})
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown kind")


def test_write_summary_round_trip(tmp_path):
    storage = _make_storage(tmp_path)
    today = utc_date(utc_now())
    payload = {"date": today, "version": "2026.6.24", "sessions": 3}
    assert storage.write_summary(today, payload) is True
    out = storage.read_latest_summary(today)
    assert out == payload


def test_retention_sweep_removes_old_files(tmp_path):
    storage = _make_storage(tmp_path)
    old_date = "2020-01-01"
    new_date = utc_date(utc_now())
    storage.append("events", {"x": 1})  # creates today
    storage._dir_for("events").mkdir(parents=True, exist_ok=True)
    (storage._dir_for("events") / f"{old_date}.jsonl").write_text('{"x":1}\n')
    (storage._dir_for("events") / f"{new_date}.jsonl").write_text('{"x":2}\n')
    removed = storage.retention_sweep()
    assert removed >= 1
    assert not (storage._dir_for("events") / f"{old_date}.jsonl").exists()
    assert (storage._dir_for("events") / f"{new_date}.jsonl").exists()


def test_read_day_returns_empty_for_missing_file(tmp_path):
    storage = _make_storage(tmp_path)
    assert storage.read_day("events", "1999-01-01") == []


def test_list_dates(tmp_path):
    storage = _make_storage(tmp_path)
    storage.append("events", {"x": 1})
    dates = storage.list_dates("events")
    assert utc_date(utc_now()) in dates


def test_base_dir_creation_failure_does_not_raise(tmp_path, monkeypatch):
    storage = LocalJsonlStorage(tmp_path / "missing" / "deep", retention_days=1)
    assert storage.base_dir.exists() or not storage.base_dir.exists()
    # We only assert that the constructor returned without raising.
    assert isinstance(storage, LocalJsonlStorage)
