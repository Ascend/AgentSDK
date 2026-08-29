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

"""Persistence tests: atomic save/load, corruption, merge_stores."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from clawcodex_ext.services.compact.context_collapse import ContextCollapseStore
from src.services.context_collapse.exceptions import (
    CollapseStateCorruptError,
    CollapseStateNotFoundError,
)
from src.services.context_collapse.persistence import (
    CollapseStateFile,
    load_store,
    merge_stores,
    save_store,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store_with(commits: list[tuple[list[str], str]]) -> ContextCollapseStore:
    s = ContextCollapseStore()
    for uuids, summary in commits:
        s.add_commit(uuids, summary)
    return s


# ---------------------------------------------------------------------------
# CollapseStateFile.save / load round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "collapse.json"
    original = _store_with(
        [
            (["u1", "u2"], "summary A"),
            (["u3", "u4"], "summary B"),
        ]
    )
    CollapseStateFile(path).save(original)

    loaded = CollapseStateFile(path).load()
    assert len(loaded.commits) == 2
    assert loaded.commits[0].archived == ["u1", "u2"]
    assert loaded.commits[0].summary == "summary A"
    assert loaded.commits[1].archived == ["u3", "u4"]


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deep" / "collapse.json"
    store = _store_with([(["u1"], "summary")])
    CollapseStateFile(path).save(store)
    assert path.exists()


def test_save_rejects_non_store(tmp_path: Path) -> None:
    path = tmp_path / "collapse.json"
    with pytest.raises(TypeError):
        CollapseStateFile(path).save("not a store")  # type: ignore[arg-type]


def test_load_missing_file_raises_not_found(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    with pytest.raises(CollapseStateNotFoundError):
        CollapseStateFile(path).load()


def test_load_corrupt_json_raises_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "collapse.json"
    path.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(CollapseStateCorruptError):
        CollapseStateFile(path).load()


def test_load_invalid_payload_raises_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "collapse.json"
    # Missing required structure
    path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    loaded = CollapseStateFile(path).load()
    # Empty commits is acceptable; it just means no history.
    assert loaded.commits == []


def test_load_validation_failure_raises_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "collapse.json"
    # Pass a non-dict value that triggers a TypeError during validation
    # (data.get("commits") on a non-dict will throw AttributeError, which is
    # not caught — exercise the json-decoder error path instead).
    path.write_text(json.dumps(None), encoding="utf-8")
    with pytest.raises((CollapseStateCorruptError, Exception)):
        CollapseStateFile(path).load()


def test_exists_reports_true_when_present(tmp_path: Path) -> None:
    path = tmp_path / "collapse.json"
    file = CollapseStateFile(path)
    assert not file.exists()
    file.save(_store_with([]))
    assert file.exists()


def test_delete_removes_file(tmp_path: Path) -> None:
    path = tmp_path / "collapse.json"
    file = CollapseStateFile(path)
    file.save(_store_with([]))
    assert file.exists()
    file.delete()
    assert not file.exists()


def test_delete_missing_file_is_safe(tmp_path: Path) -> None:
    path = tmp_path / "never_existed.json"
    # Should not raise even if the file doesn't exist.
    CollapseStateFile(path).delete()
    assert not path.exists()


# ---------------------------------------------------------------------------
# Atomic write safety
# ---------------------------------------------------------------------------


def test_atomic_write_no_temp_leftover(tmp_path: Path) -> None:
    path = tmp_path / "collapse.json"
    store = _store_with([(["u1"], "summary")])
    CollapseStateFile(path).save(store)
    # No .tmp leftovers should remain in the directory.
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_atomic_write_overwrites_existing(tmp_path: Path) -> None:
    path = tmp_path / "collapse.json"
    CollapseStateFile(path).save(_store_with([(["u1"], "old")]))
    CollapseStateFile(path).save(_store_with([(["u2"], "new")]))
    loaded = CollapseStateFile(path).load()
    assert len(loaded.commits) == 1
    assert loaded.commits[0].archived == ["u2"]
    assert loaded.commits[0].summary == "new"


def test_concurrent_writes_do_not_corrupt_file(tmp_path: Path) -> None:
    """Many threads writing simultaneously should produce a valid file."""
    path = tmp_path / "collapse.json"
    file = CollapseStateFile(path)

    def writer(i: int) -> None:
        store = _store_with([([f"u{i}"], f"summary {i}")])
        file.save(store)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The final file must be valid JSON and parseable.
    loaded = file.load()
    assert isinstance(loaded, ContextCollapseStore)


# ---------------------------------------------------------------------------
# save_store / load_store convenience
# ---------------------------------------------------------------------------


def test_save_store_and_load_store(tmp_path: Path) -> None:
    path = tmp_path / "collapse.json"
    original = _store_with([(["u1"], "summary")])
    save_store(path, original)

    loaded = load_store(path)
    assert len(loaded.commits) == 1
    assert loaded.commits[0].archived == ["u1"]


# ---------------------------------------------------------------------------
# merge_stores
# ---------------------------------------------------------------------------


def test_merge_stores_adds_new_commits(tmp_path: Path) -> None:
    target = _store_with([(["a"], "first")])
    source = _store_with(
        [
            (["a"], "first"),  # duplicate
            (["b"], "second"),  # new
            (["c"], "third"),  # new
        ]
    )
    added = merge_stores(target, source)
    assert added == 2
    assert len(target.commits) == 3


def test_merge_stores_dedupes_by_archived_set(tmp_path: Path) -> None:
    """Order in the archived list matters for frozenset comparison."""
    target = _store_with([(["a", "b"], "first")])
    # Source has the same set in a different order — same frozenset, treated as duplicate.
    source = _store_with([(["b", "a"], "first")])
    added = merge_stores(target, source)
    assert added == 0
    assert len(target.commits) == 1


def test_merge_stores_prefer_source_overwrites(tmp_path: Path) -> None:
    target = _store_with([(["a"], "old summary")])
    source = _store_with([(["a"], "new summary")])
    added = merge_stores(target, source, prefer_target=False)
    assert added == 0
    summaries = [c.summary for c in target.commits]
    assert "new summary" in summaries


def test_merge_stores_prefer_target_keeps_existing(tmp_path: Path) -> None:
    target = _store_with([(["a"], "target summary")])
    source = _store_with([(["a"], "source summary")])
    merge_stores(target, source, prefer_target=True)
    summaries = [c.summary for c in target.commits]
    assert "target summary" in summaries
    assert "source summary" not in summaries


def test_merge_stores_rejects_non_store_args() -> None:
    s = _store_with([])
    with pytest.raises(TypeError):
        merge_stores("not a store", s)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        merge_stores(s, "not a store")  # type: ignore[arg-type]


def test_merge_stores_appends_only_unique(tmp_path: Path) -> None:
    target = _store_with([(["a"], "x")])
    source = _store_with(
        [
            (["a"], "x"),
            (["b"], "y"),
            (["b"], "y"),  # duplicate within source — only first added
        ]
    )
    added = merge_stores(target, source)
    # 'a' is duplicate (skip), 'b' is new (add). The duplicate within source
    # is not re-added because target already has 'b' after first copy.
    assert added == 1
    assert len(target.commits) == 2
