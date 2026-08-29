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

# pylint: disable=no-name-in-module,protected-access

"""Spec-6 goal objective materialization tests."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from stat import S_IMODE
from unittest.mock import MagicMock, patch

import pytest
from clawcodex_ext.goal import files as goal_files
from clawcodex_ext.goal import model as goal_model
from clawcodex_ext.goal.command import _set_goal_condition
from clawcodex_ext.goal.files import (
    GOAL_FILE_NAME,
    MAX_THREAD_GOAL_OBJECTIVE_CHARS,
    materialize_goal_objective,
    objective_text_for_edit,
    remove_materialized_goal_objective,
)
from clawcodex_ext.goal.store import GoalStore


def test_long_goal_objective_is_materialized_under_codex_attachments(
    tmp_path: Path,
) -> None:
    objective = "ship it\n" + ("x" * MAX_THREAD_GOAL_OBJECTIVE_CHARS)

    materialized = materialize_goal_objective(objective, codex_home=tmp_path)

    goal_file = materialized.attachment_dir / GOAL_FILE_NAME
    assert goal_file.read_text() == objective
    assert objective_text_for_edit(materialized.objective, codex_home=tmp_path) == objective


def test_short_goal_objective_is_not_materialized(tmp_path: Path) -> None:
    materialized = materialize_goal_objective("short objective", codex_home=tmp_path)

    assert materialized.objective == "short objective"
    assert materialized.attachment_dir is None
    assert not (tmp_path / "attachments").exists()


def test_materialized_goal_objective_can_be_removed_safely(tmp_path: Path) -> None:
    materialized = materialize_goal_objective("x" * 4001, codex_home=tmp_path)

    assert remove_materialized_goal_objective(
        materialized.objective,
        codex_home=tmp_path,
    )
    assert materialized.attachment_dir is not None
    assert not materialized.attachment_dir.exists()


def test_materialized_goal_cleanup_reports_file_removal_when_rmdir_fails(
    tmp_path: Path,
) -> None:
    materialized = materialize_goal_objective("x" * 4001, codex_home=tmp_path)
    assert materialized.attachment_dir is not None
    goal_file = materialized.attachment_dir / GOAL_FILE_NAME

    with patch.object(Path, "rmdir", side_effect=OSError("directory busy")):
        assert remove_materialized_goal_objective(
            materialized.objective,
            codex_home=tmp_path,
        )

    assert not goal_file.exists()
    assert materialized.attachment_dir.exists()


def test_materialized_goal_cleanup_refuses_symlink(tmp_path: Path) -> None:
    materialized = materialize_goal_objective("x" * 4001, codex_home=tmp_path)
    assert materialized.attachment_dir is not None
    goal_file = materialized.attachment_dir / GOAL_FILE_NAME
    target = tmp_path / "outside.md"
    target.write_text("keep", encoding="utf-8")
    goal_file.unlink()
    goal_file.symlink_to(target)

    assert not remove_materialized_goal_objective(
        materialized.objective,
        codex_home=tmp_path,
    )
    assert target.read_text(encoding="utf-8") == "keep"


def test_objective_text_for_edit_only_reads_valid_goal_objective_references(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "elsewhere" / GOAL_FILE_NAME
    outside.parent.mkdir()
    outside.write_text("secret")
    reference = f"Read the Codex goal objective file at {outside} before continuing."

    assert objective_text_for_edit(reference, codex_home=tmp_path) == reference


def test_goal_objective_uses_configured_clawcodex_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    monkeypatch.setenv("CLAWCODEX_HOME", str(configured))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fallback")

    result = materialize_goal_objective("x" * (MAX_THREAD_GOAL_OBJECTIVE_CHARS + 1))

    assert result.attachment_dir.parent == configured / "attachments"
    monkeypatch.delenv("CLAWCODEX_HOME")
    fallback = materialize_goal_objective("x" * (MAX_THREAD_GOAL_OBJECTIVE_CHARS + 1))
    assert fallback.attachment_dir.parent == tmp_path / "fallback" / ".clawcodex" / "attachments"


def test_materialization_failures_remove_the_uuid_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(goal_files, "MAX_THREAD_GOAL_OBJECTIVE_CHARS", 1)
    with pytest.raises(ValueError, match="reference is too long"):
        materialize_goal_objective("xx", codex_home=tmp_path)
    assert not any((tmp_path / "attachments").iterdir())

    monkeypatch.setattr(goal_files, "MAX_THREAD_GOAL_OBJECTIVE_CHARS", 4000)
    with patch.object(Path, "write_text", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            materialize_goal_objective("x" * 4001, codex_home=tmp_path)
    assert not any((tmp_path / "attachments").iterdir())


def test_directory_permission_failure_removes_the_uuid_directory(tmp_path: Path) -> None:
    with patch.object(Path, "chmod", side_effect=OSError("chmod failed")):
        with pytest.raises(OSError, match="chmod failed"):
            materialize_goal_objective("x" * 4001, codex_home=tmp_path)

    assert not any((tmp_path / "attachments").iterdir())


def test_goal_command_returns_validation_outcome_for_oversized_condition() -> None:
    api = MagicMock()

    outcome = _set_goal_condition(
        api,
        "thread-1",
        "x" * (MAX_THREAD_GOAL_OBJECTIVE_CHARS + 1),
        MagicMock(),
    )

    assert outcome.message == (f"Goal condition must be {MAX_THREAD_GOAL_OBJECTIVE_CHARS:,} characters or fewer.")
    assert outcome.display == "system"
    api.thread_goal_replace.assert_not_called()


def test_goal_files_and_database_are_private(tmp_path: Path) -> None:
    previous_umask = os.umask(0)
    try:
        result = materialize_goal_objective("x" * 4001, codex_home=tmp_path)
        store = GoalStore(tmp_path / "goals.sqlite")
    finally:
        os.umask(previous_umask)
    store.close()

    assert S_IMODE(result.attachment_dir.stat().st_mode) == 0o700
    assert S_IMODE((result.attachment_dir / GOAL_FILE_NAME).stat().st_mode) == 0o600
    assert S_IMODE((tmp_path / "goals.sqlite").stat().st_mode) == 0o600


def test_goal_store_supports_scoped_cleanup_and_locks_reads(tmp_path: Path) -> None:
    with GoalStore(tmp_path / "goals.sqlite") as store:
        lock = MagicMock()
        store._lock = lock
        assert store.get_thread_goal("missing") is None
        lock.__enter__.assert_called_once_with()

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        store.get_thread_goal("missing")


def test_datetime_utc_normalization_rejects_underflow_semantically() -> None:
    value = datetime.min.replace(tzinfo=timezone(timedelta(hours=1)))
    with pytest.raises(ValueError, match="represented in UTC"):
        goal_model._normalize_datetime(value)
