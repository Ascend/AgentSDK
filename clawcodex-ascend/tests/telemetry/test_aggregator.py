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

"""Tests for DailyAggregator."""

from __future__ import annotations

import time
from pathlib import Path

from telemetry.aggregator import DailyAggregator
from telemetry.storage import LocalJsonlStorage, utc_date, utc_now


def _storage(tmp_path: Path) -> LocalJsonlStorage:
    return LocalJsonlStorage(tmp_path / "telemetry", retention_days=7)


def _session_start(sid, fields=None):
    return {
        "type": "session_start",
        "timestamp": time.time(),
        "session_id": sid,
        "schema_version": 1,
        "fields": {
            "entrypoint": "cli",
            "client_type": "cli",
            "platform": "Linux",
            "python_version": "3.11.0",
            "provider": "anthropic",
            "model": "claude-3-5-sonnet",
            **(fields or {}),
        },
    }


def _command_run(sid, name="repl", success=True, duration_s=1.0, exit_status=0):
    return {
        "type": "command_run",
        "timestamp": time.time(),
        "session_id": sid,
        "schema_version": 1,
        "fields": {
            "command_name": name,
            "mode": "interactive",
            "success": success,
            "duration_s": duration_s,
            "exit_status": exit_status,
        },
    }


def _crash(sid, fingerprint="abc123", error_class="ValueError", ts=None):
    return {
        "type": "error",
        "timestamp": ts if ts is not None else time.time(),
        "session_id": sid,
        "schema_version": 1,
        "fields": {
            "error_class": error_class,
            "fingerprint": fingerprint,
            "stacktrace": ["line1", "line2"],
        },
    }


def test_aggregate_empty(tmp_path):
    storage = _storage(tmp_path)
    agg = DailyAggregator(storage)
    today = utc_date(utc_now())
    summary = agg.aggregate(today)
    assert summary["sessions"] == 0
    assert summary["commands"] == 0
    assert summary["crashes"]["total"] == 0


def test_aggregate_counts_sessions_and_commands(tmp_path):
    storage = _storage(tmp_path)
    storage.append("events", _session_start("s1"))
    storage.append("events", _session_start("s2"))
    storage.append("events", _command_run("s1", name="repl"))
    storage.append("events", _command_run("s2", name="headless", success=False, exit_status=1))
    agg = DailyAggregator(storage)
    today = utc_date(utc_now())
    summary = agg.aggregate(today)
    assert summary["sessions"] == 2
    assert summary["commands"] == 2
    assert summary["top_commands"][0]["name"] in ("repl", "headless")
    assert summary["command_success"].get("repl", 0) == 1
    assert summary["command_failure"].get("headless", 0) == 1


def test_aggregate_collects_crashes(tmp_path):
    storage = _storage(tmp_path)
    storage.append("crashes", _crash("s1", fingerprint="deadbeef", error_class="RuntimeError"))
    storage.append("crashes", _crash("s1", fingerprint="deadbeef", error_class="RuntimeError"))
    storage.append("crashes", _crash("s2", fingerprint="1234abcd", error_class="ValueError"))
    agg = DailyAggregator(storage)
    today = utc_date(utc_now())
    summary = agg.aggregate(today)
    assert summary["crashes"]["total"] == 3
    top = {row["fingerprint"]: row for row in summary["crashes"]["top"]}
    assert top["deadbeef"]["count"] == 2
    assert top["deadbeef"]["error_class"] == "RuntimeError"


def test_aggregate_persists_to_summary_file(tmp_path):
    storage = _storage(tmp_path)
    storage.append("events", _session_start("s1"))
    agg = DailyAggregator(storage)
    today = utc_date(utc_now())
    summary = agg.aggregate(today)
    assert summary["date"] == today
    assert storage.read_latest_summary(today) == summary


def test_aggregate_today_if_stale_is_idempotent(tmp_path):
    storage = _storage(tmp_path)
    storage.append("events", _session_start("s1"))
    agg = DailyAggregator(storage)
    s1 = agg.aggregate_today_if_stale()
    s2 = agg.aggregate_today_if_stale()
    assert s1["sessions"] == s2["sessions"] == 1
    # Same date is reused; the disk file is unchanged in shape.
    assert s1["date"] == s2["date"]


def test_aggregate_tool_summary(tmp_path):
    storage = _storage(tmp_path)
    storage.append(
        "events",
        {
            "type": "tool_summary",
            "timestamp": time.time(),
            "session_id": "s1",
            "fields": {"tool_name": "bash", "success": True, "duration_s": 0.5},
        },
    )
    agg = DailyAggregator(storage)
    summary = agg.aggregate(utc_date(utc_now()))
    assert summary["tools"]["top"][0]["name"] == "bash"
    assert summary["tools"]["success"]["bash"] == 1


def test_aggregate_handles_payload_without_fields(tmp_path):
    storage = _storage(tmp_path)
    storage.append("events", {"type": "session_start", "timestamp": time.time(), "session_id": "s1"})
    agg = DailyAggregator(storage)
    summary = agg.aggregate(utc_date(utc_now()))
    assert summary["sessions"] == 1


# ---------------------------------------------------------------------------
# F-97-L: mixed v1 / v2 events
# ---------------------------------------------------------------------------


def _session_start_v2(sid):
    """v2-shaped session_start row, mirroring :func:`_session_start`."""
    return {
        "type": "session_start",
        "timestamp": time.time(),
        "session_id": sid,
        "schema_version": 2,
        "fields": {
            "entrypoint": "tui",
            "client_type": "tui",
            "platform": "Linux",
            "python_version": "3.12.0",
            "provider": "anthropic",
            "model": "claude-opus-4-7",
        },
    }


def _crash_v2(sid, fingerprint_hash, error_class="ValueError"):
    """v2-shaped crash row with the structured fingerprint dict form."""
    return {
        "type": "error",
        "timestamp": time.time(),
        "session_id": sid,
        "schema_version": 2,
        "fields": {
            "error_class": error_class,
            "fingerprint": {
                "hash": fingerprint_hash,
                "version": 2,
                "method": "sha1-truncate",
            },
            "stacktrace": ["line1"],
        },
    }


def test_aggregate_handles_v1_and_v2_mixed(tmp_path):
    """F-97-L: events written before and after the v1→v2 cutover must
    produce a single coherent daily summary — sessions counted from
    both shapes, summary stamped at v2.
    """
    storage = _storage(tmp_path)
    # v1 row (legacy binary)
    storage.append("events", _session_start("v1-sess"))
    storage.append("events", _command_run("v1-sess", name="repl"))
    # v2 row (current binary)
    storage.append("events", _session_start_v2("v2-sess"))
    storage.append("events", _command_run("v2-sess", name="tui"))
    agg = DailyAggregator(storage)
    summary = agg.aggregate(utc_date(utc_now()))

    assert summary["schema_version"] == 2
    assert summary["sessions"] == 2
    assert summary["commands"] == 2
    # Both shapes must contribute to the top-commands list. Tied
    # counts preserve insertion order, so don't assert index here.
    names = [entry["name"] for entry in summary["top_commands"]]
    assert set(names) == {"repl", "tui"}


def test_aggregate_crash_summary_groups_v1_v2_same_hash(tmp_path):
    """F-97-L: a v1 crash with fingerprint string 'abc123' and a v2
    crash with fingerprint dict ``{'hash': 'abc123', ...}`` must end
    up in the same crash bucket after the v1→v2 migration.
    """
    storage = _storage(tmp_path)
    storage.append("crashes", _crash("v1-sess", fingerprint="abc123", error_class="E"))
    storage.append("crashes", _crash_v2("v2-sess", "abc123", error_class="E"))
    storage.append("crashes", _crash("v1-sess", fingerprint="different", error_class="E"))
    agg = DailyAggregator(storage)
    summary = agg.aggregate(utc_date(utc_now()))

    crashes = summary["crashes"]
    assert crashes["total"] == 3
    # The shared "abc123" hash from both shapes must collapse to a
    # single bucket with count=2; the distinct hash forms a second
    # bucket.
    by_hash = {entry["fingerprint"]: entry for entry in crashes["top"]}
    assert by_hash["abc123"]["count"] == 2
    assert by_hash["different"]["count"] == 1
    # And the v2-shaped crash and v1-shaped crash must agree on the
    # recorded error_class
    assert by_hash["abc123"]["error_class"] == "E"
