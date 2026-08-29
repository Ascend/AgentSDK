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
#

"""Tests for the ``/dream`` slash skill — dream skill.

Covers subcommand dispatch: ``run`` / ``once`` / ``status`` / ``help``,
plus registration as a :class:`LocalCommand` in the global command
registry. Hermetic via autouse fixture: pins
``project_transcript_dir`` to a tmp path so manual_dream's session
scan never reaches the real ``~/.clawcodex/sessions/`` directory.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from extensions.skills_ext.bundled.dream import (
    _dream_call,
    _dream_help,
    _dream_status,
    register_dream_skill,
)
from clawcodex_ext.task_registry import RuntimeTaskRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reset state between tests.

    * Clear the dream service closure (so ``manual_dream`` creates
      a fresh one) and any prior registry state.
    * Pin ``project_transcript_dir`` (in BOTH ``service.py`` and
      ``lock.py``) to ``tmp_path / "sessions"`` to keep the session
      scan hermetic.
    * Pin auto-memory dir to ``tmp_path`` for the consolidation lock.
    * Wipe the lock file between tests.
    """
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    monkeypatch.setenv("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", str(tmp_path))
    monkeypatch.delenv("CLAWCODEX_DISABLE_AUTO_DREAM", raising=False)
    monkeypatch.delenv("CLAWCODEX_KAIROS", raising=False)

    from clawcodex_ext.dreaming.config import (
        DEFAULT_DREAM_CONFIG,
        set_dream_config,
    )
    from clawcodex_ext.dreaming.lock import LOCK_FILE_NAME
    from clawcodex_ext.dreaming.runner import set_dream_runner_factory

    set_dream_config(DEFAULT_DREAM_CONFIG)
    set_dream_runner_factory(None)

    monkeypatch.setattr(
        "clawcodex_ext.dreaming.service.project_transcript_dir",
        lambda *_a, **_kw: str(sessions_dir),
    )
    monkeypatch.setattr(
        "clawcodex_ext.dreaming.lock.project_transcript_dir",
        lambda *_a, **_kw: str(sessions_dir),
    )

    # Reset dream service closure (no leftover state from prior tests).
    from clawcodex_ext.dreaming import service as _service

    _service._runner = None

    lock_path = tmp_path / LOCK_FILE_NAME
    if lock_path.exists():
        lock_path.unlink()

    yield  # type: ignore[misc]

    set_dream_runner_factory(None)
    _service._runner = None


def _init_dream() -> RuntimeTaskRegistry:
    """Initialize the auto-dream service with permissive gates for testing."""
    from clawcodex_ext.dreaming import init_auto_dream
    from clawcodex_ext.dreaming.config import DreamConfig

    reg = RuntimeTaskRegistry()
    init_auto_dream(
        config=DreamConfig(min_hours=0.0, min_sessions=0),
        registry=reg,
    )
    return reg


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_dream_skill_registers_local_command() -> None:
    """``register_dream_skill`` adds a LocalCommand named ``dream`` to the
    global command registry.
    """
    from clawcodex_ext.command_system import get_command_registry
    from clawcodex_ext.command_system.types import CommandType

    # Clear first so the test is order-independent.
    get_command_registry().clear()
    register_dream_skill()

    cmd = get_command_registry().get("dream")
    assert cmd is not None, "dream command should be registered"
    assert cmd.command_type == CommandType.LOCAL, f"expected LOCAL, got {cmd.command_type}"
    assert cmd.loaded_from == "bundled"
    assert "/dream" in cmd.description or "dream" in cmd.description.lower()


def test_register_dream_skill_is_idempotent() -> None:
    """Re-registering replaces the prior entry (no duplicate / no error)."""
    from clawcodex_ext.command_system import get_command_registry

    get_command_registry().clear()
    register_dream_skill()
    register_dream_skill()
    # Only one entry remains.
    assert len(get_command_registry().list_commands(include_hidden=True)) == 1


# ---------------------------------------------------------------------------
# Subcommand dispatch — pure (no side effects)
# ---------------------------------------------------------------------------


def test_dream_no_args_shows_help() -> None:
    """``/dream`` with no args shows the usage help."""
    result = _dream_call("", context=None)
    assert result.type == "text"
    assert "Usage:" in result.value
    assert "run" in result.value
    assert "status" in result.value


def test_dream_help_subcommand() -> None:
    """``/dream help`` shows the same usage."""
    result = _dream_call("help", context=None)
    assert "Usage:" in result.value
    assert "Subcommands:" in result.value


def test_dream_unknown_subcommand_warns() -> None:
    """Unknown subcommand shows help with a warning line."""
    result = _dream_call("frobnicate", context=None)
    assert "Unknown subcommand" in result.value
    assert "frobnicate" in result.value
    # The usage block still appears below.
    assert "Usage:" in result.value


def test_dream_help_pure() -> None:
    """``_dream_help`` helper is pure — no env / fs access."""
    result = _dream_help()
    assert result.type == "text"
    assert "run" in result.value and "status" in result.value


# ---------------------------------------------------------------------------
# status subcommand
# ---------------------------------------------------------------------------


def test_dream_status_no_tasks() -> None:
    """``/dream status`` with no in-flight tasks shows a clean message."""
    reg = _init_dream()  # init the service so execute path is wired
    assert reg.by_type("dream") == []
    result = _dream_status()
    assert "No dream tasks in flight" in result.value


def test_dream_status_with_in_flight_task() -> None:
    """``/dream status`` shows the in-flight dream task with its state."""
    from src.tasks.dream import register_dream_task

    # Wire the closure so /dream status reads from *this* registry.
    reg = _init_dream()
    task_id = register_dream_task(
        sessions_reviewing=3,
        prior_mtime=int(time.time() * 1000),
        registry=reg,
    )

    result = _dream_status()
    assert "In-flight dream tasks (1)" in result.value
    assert task_id in result.value
    assert "status=" in result.value
    assert "phase=starting" in result.value
    assert "sessions=3" in result.value


# ---------------------------------------------------------------------------
# run subcommand (side effect: triggers manual_dream which fires force=True)
# ---------------------------------------------------------------------------


def test_dream_run_registers_completed_task() -> None:
    """``/dream run`` triggers manual_dream → registers a dream task → completes."""
    reg = _init_dream()
    result = _dream_call("run", context=None)
    # Success message.
    assert "Dream consolidation triggered" in result.value
    # The manual_dream path ran a force=True pass.
    tasks = reg.by_type("dream")
    assert len(tasks) == 1
    assert tasks[0].status == "completed"


def test_dream_once_alias_runs_the_same_path() -> None:
    """``/dream once`` is an alias for ``run``."""
    reg = _init_dream()
    result = _dream_call("once", context=None)
    assert "Dream consolidation triggered" in result.value
    assert len(reg.by_type("dream")) == 1


def test_dream_run_returns_text_type() -> None:
    """``/dream run`` always returns a text result (engine converts to system)."""
    _init_dream()
    result = _dream_call("run", context=None)
    assert result.type == "text"
    assert result.value  # non-empty
