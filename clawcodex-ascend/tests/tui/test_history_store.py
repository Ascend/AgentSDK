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

"""Unit tests for :class:`HistoryStore`."""

from __future__ import annotations


from src.tui.history_store import HistoryStore


def test_append_and_reload_returns_records(tmp_path):
    store = HistoryStore(tmp_path / "history.jsonl")
    store.append("git status")
    store.append("rm -rf build")
    records = store.load()
    assert [r.prompt for r in records] == ["git status", "rm -rf build"]
    assert all(r.timestamp > 0 for r in records)


def test_empty_prompts_are_skipped(tmp_path):
    store = HistoryStore(tmp_path / "history.jsonl")
    store.append("")
    store.append("   ")
    assert store.load() == []


def test_recent_is_reverse_chronological(tmp_path):
    store = HistoryStore(tmp_path / "history.jsonl")
    store.append("first")
    store.append("second")
    store.append("third")
    assert [r.prompt for r in store.recent()] == ["third", "second", "first"]


def test_recent_limit(tmp_path):
    store = HistoryStore(tmp_path / "history.jsonl")
    for i in range(5):
        store.append(f"entry {i}")
    assert [r.prompt for r in store.recent(limit=2)] == ["entry 4", "entry 3"]


def test_rotation_caps_entries(tmp_path):
    store = HistoryStore(tmp_path / "history.jsonl", max_entries=3)
    for i in range(10):
        store.append(f"entry {i}")
    records = store.load()
    assert [r.prompt for r in records] == ["entry 7", "entry 8", "entry 9"]


def test_corrupt_lines_are_skipped(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text(
        '{"prompt": "ok", "ts": 1}\nthis is garbage\n{"prompt": "ok2", "ts": 2}\n',
        encoding="utf-8",
    )
    store = HistoryStore(path)
    assert [r.prompt for r in store.load()] == ["ok", "ok2"]
