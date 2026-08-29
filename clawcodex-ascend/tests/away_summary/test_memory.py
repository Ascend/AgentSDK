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

# pylint: disable=no-name-in-module,useless-return

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from clawcodex_ext.away_summary.memory import (
    _format_memory,
    _resolve_load_summary,
    get_session_memory_content,
)


def test_get_session_memory_returns_none_for_missing_session() -> None:
    """No sidecar → no memory block. Recap must not crash on missing data."""
    assert get_session_memory_content(session_id="absent") is None
    assert get_session_memory_content(session_id=None) is None
    assert get_session_memory_content(session_id="") is None


def test_optional_memory_module_can_be_absent(monkeypatch) -> None:
    def missing(_name: str):
        exc = ModuleNotFoundError("No module named 'clawcodex_ext.session_intelligence.index'")
        exc.name = "clawcodex_ext.session_intelligence.index"
        raise exc

    monkeypatch.setattr(
        "clawcodex_ext.away_summary.memory.load_summary",
        None,
    )
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.memory.import_module",
        missing,
    )

    assert _resolve_load_summary() is None


def test_optional_memory_import_reraises_unrelated_missing_dependency(
    monkeypatch,
) -> None:
    def missing_dependency(_name: str):
        exc = ModuleNotFoundError("No module named 'sqlite_driver'")
        exc.name = "sqlite_driver"
        raise exc

    monkeypatch.setattr(
        "clawcodex_ext.away_summary.memory.load_summary",
        None,
    )
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.memory.import_module",
        missing_dependency,
    )

    with pytest.raises(ModuleNotFoundError, match="sqlite_driver"):
        _resolve_load_summary()


def test_get_session_memory_returns_none_when_load_summary_fails(
    monkeypatch,
) -> None:
    """A failing ``load_summary`` yields None gracefully."""
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.memory.load_summary",
        lambda *_args, **_kwargs: None,
    )
    assert get_session_memory_content(session_id="any") is None


def test_get_session_memory_returns_none_when_load_summary_raises(
    monkeypatch,
) -> None:
    """An exception inside ``load_summary`` is swallowed, not propagated."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("sidecar broken")

    monkeypatch.setattr(
        "clawcodex_ext.away_summary.memory.load_summary",
        boom,
    )
    assert get_session_memory_content(session_id="any") is None


def test_get_session_memory_formats_known_fields(monkeypatch) -> None:
    """A populated sidecar projects into a human-readable recap block."""
    summary = {
        "schema_version": 1,
        "session_id": "s1",
        "cwd": "/home/dev/project",
        "title": "Refactor away-summary",
        "goals": ["Align prompt wording with upstream"],
        "completed": ["Migrated test_service.py"],
        "open_threads": ["Add cache reuse for /recap"],
        "next_action_candidates": ["Wire memory injection"],
        "user_preferences": [],
    }

    def fake_load_summary(session_id: str, **_kwargs) -> dict[str, Any]:
        assert session_id == "s1"
        return summary

    monkeypatch.setattr(
        "clawcodex_ext.away_summary.memory.load_summary",
        fake_load_summary,
    )

    text = get_session_memory_content(session_id="s1")
    assert text is not None
    # Title + cwd come first.
    assert "Refactor away-summary" in text
    assert "/home/dev/project" in text
    # All four populated sections appear with their labels.
    assert "Goals:" in text
    assert "Completed:" in text
    assert "Open threads:" in text
    assert "Next candidates:" in text
    # Empty sections are omitted.
    assert "User preferences:" not in text


def test_get_session_memory_truncates_long_sides(monkeypatch) -> None:
    """Long lists are tail-truncated to keep the recap prompt short."""
    summary = {
        "schema_version": 1,
        "session_id": "s2",
        "title": "Long session",
        "goals": [f"goal-{i}" for i in range(20)],
        "completed": [f"done-{i}" for i in range(20)],
        "open_threads": [f"thread-{i}" for i in range(20)],
        "next_action_candidates": [f"next-{i}" for i in range(20)],
    }

    monkeypatch.setattr(
        "clawcodex_ext.away_summary.memory.load_summary",
        lambda *_a, **_kw: summary,
    )

    text = get_session_memory_content(session_id="s2")
    assert text is not None
    # Only the most recent 5 entries per section survive (indices 15..19).
    assert "goal-19" in text
    assert "goal-15" in text
    assert "goal-14" not in text
    assert "done-19" in text and "done-15" in text and "done-14" not in text


def test_get_session_memory_truncates_by_chars(monkeypatch) -> None:
    """The block is hard-capped by ``max_chars`` with an ellipsis tail."""
    long_goal = "x" * 2000
    summary = {
        "schema_version": 1,
        "session_id": "s3",
        "title": "Big session",
        "goals": [long_goal],
    }
    monkeypatch.setattr(
        "clawcodex_ext.away_summary.memory.load_summary",
        lambda *_a, **_kw: summary,
    )

    text = get_session_memory_content(session_id="s3", max_chars=400)
    assert text is not None
    assert text.endswith("…")
    assert len(text) <= 400


def test_format_memory_returns_none_for_empty_dict() -> None:
    """A sidecar with only schema_version fields produces None, not empty string."""
    assert _format_memory({"schema_version": 1, "session_id": "x"}, max_chars=4000) is None
    assert _format_memory({"title": "x"}, max_chars=0) is None


def test_format_memory_handles_non_string_entries() -> None:
    """Sidecar lists sometimes contain dicts/None — they must not crash."""
    text = _format_memory(
        {
            "title": 12345,  # coerced to str
            "goals": ["real-goal", None, "", 0, False, {"nested": "ignore"}],
            "completed": [],
        },
        max_chars=4000,
    )
    assert text is not None
    assert "12345" in text
    assert "real-goal" in text
    assert "- 0" in text
    assert "- False" in text


def test_get_session_memory_passes_sessions_dir_through(monkeypatch) -> None:
    """The helper forwards a custom sessions_dir when provided."""

    seen: dict[str, Any] = {}

    def fake_load_summary(session_id: str, *, sessions_dir=None):
        seen["session_id"] = session_id
        seen["sessions_dir"] = sessions_dir
        return None

    monkeypatch.setattr(
        "clawcodex_ext.away_summary.memory.load_summary",
        fake_load_summary,
    )

    # 1) Default — no sessions_dir override.
    get_session_memory_content(session_id="s1")
    assert seen["session_id"] == "s1"
    assert seen["sessions_dir"] is None

    # 2) Custom sessions_dir flows through unchanged.
    custom = Path("/tmp/custom-sessions")
    get_session_memory_content(session_id="s2", sessions_dir=custom)
    assert seen["session_id"] == "s2"
    assert seen["sessions_dir"] == custom
