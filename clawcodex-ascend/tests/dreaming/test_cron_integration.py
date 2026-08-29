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

"""Tests for ``clawcodex_ext.dreaming.cron_integration``.

Covers the permanent-cron installer and the local fire handler that
intercepts the dream task before it reaches the model outbox.

Hermetic via autouse fixture: pins ``project_transcript_dir`` to a
tmp path so any gate-chain call in the fire handler never reads the
real ``~/.clawcodex/sessions/`` directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from clawcodex_ext.dreaming import (
    DREAM_DEFAULT_CRON,
    DREAM_PERMANENT_PROMPT,
    DREAM_PERMANENT_TASK_ID,
    init_auto_dream,
    install_and_wire_dream,
    install_dream_permanent_cron_task,
    wire_dream_fire_handler,
)
from clawcodex_ext.dreaming.config import DreamConfig
from clawcodex_ext.dreaming.lock import LOCK_FILE_NAME
from clawcodex_ext.task_registry import RuntimeTaskRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reset state between tests."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    monkeypatch.setenv("CLAUDE_COWORK_MEMORY_PATH_OVERRIDE", str(tmp_path))
    monkeypatch.delenv("CLAWCODEX_DISABLE_AUTO_DREAM", raising=False)
    monkeypatch.delenv("CLAWCODEX_KAIROS", raising=False)

    from clawcodex_ext.dreaming import service as _service
    from clawcodex_ext.dreaming.config import DEFAULT_DREAM_CONFIG, set_dream_config
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

    _service._runner = None

    lock_path = tmp_path / LOCK_FILE_NAME
    if lock_path.exists():
        lock_path.unlink()

    yield  # type: ignore[misc]

    set_dream_runner_factory(None)
    _service._runner = None


# ---------------------------------------------------------------------------
# Stubs / fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeRun:
    """Minimal CronRun stand-in for the wire-handler tests."""

    id: str = "run-1"


@dataclass
class _FakeScheduler:
    """Minimal CronScheduler stand-in — only the fields the wire
    handler touches.
    """

    on_fire_task: Any = None
    on_fire: Any = None
    on_missed: Any = None


# ---------------------------------------------------------------------------
# install_dream_permanent_cron_task
# ---------------------------------------------------------------------------


def test_install_creates_dream_task_with_defaults(tmp_path: Path) -> None:
    """First install creates a permanent task with the well-known id
    and the default 3 AM cron expression.
    """
    task, created = install_dream_permanent_cron_task(tmp_path)
    assert created is True
    assert task.id == DREAM_PERMANENT_TASK_ID
    assert task.cron == DREAM_DEFAULT_CRON
    assert task.permanent is True
    assert task.expires_at is None  # permanent never expires
    assert task.recurring is True
    assert task.prompt.casefold() == DREAM_PERMANENT_PROMPT.casefold()


def test_install_is_idempotent(tmp_path: Path) -> None:
    """Re-installing the same spec is a no-op (created=False)."""
    task1, created1 = install_dream_permanent_cron_task(tmp_path)
    task2, created2 = install_dream_permanent_cron_task(tmp_path)
    assert created1 is True
    assert created2 is False
    assert task1.id == task2.id


def test_install_respects_custom_cron(tmp_path: Path) -> None:
    """The caller can override the cron expression."""
    task, created = install_dream_permanent_cron_task(tmp_path, cron_expr="0 4 * * 0")
    assert created is True
    assert task.cron == "0 4 * * 0"


def test_install_respects_custom_task_id(tmp_path: Path) -> None:
    """The caller can override the task id (e.g. for tests)."""
    task, created = install_dream_permanent_cron_task(tmp_path, task_id="dream-custom")
    assert created is True
    assert task.id == "dream-custom"


def test_install_rejects_overwrite_of_other_permanent(tmp_path: Path) -> None:
    """A different prompt under the same cron is rejected (PermissionError)."""
    install_dream_permanent_cron_task(tmp_path)
    with pytest.raises(PermissionError):
        install_dream_permanent_cron_task(tmp_path, prompt="something completely different")


# ---------------------------------------------------------------------------
# wire_dream_fire_handler
# ---------------------------------------------------------------------------


def test_wire_intercepts_dream_task_and_calls_execute(monkeypatch, tmp_path: Path) -> None:
    """A fired dream task triggers ``execute_auto_dream`` (force=False)
    via the wire handler, not the original on_fire_task.
    """
    # Init the service so execute_auto_dream is wired.
    reg = RuntimeTaskRegistry()
    init_auto_dream(
        config=DreamConfig(min_hours=0.0, min_sessions=0),
        registry=reg,
    )

    fired_original: list[Any] = []
    scheduler = _FakeScheduler(on_fire_task=lambda t, r: fired_original.append(t))
    wire_dream_fire_handler(scheduler, registry=reg)

    dream_task, _ = install_dream_permanent_cron_task(tmp_path)
    # Drive the wire handler directly with a fake run.
    scheduler.on_fire_task(dream_task, _FakeRun(id="run-x"))

    # The dream task was handled locally — execute_auto_dream
    # registered a completed task on the reg.
    assert reg.by_type("dream"), "wire handler should have fired execute_auto_dream"
    assert all(t.status == "completed" for t in reg.by_type("dream"))
    # The original handler was NOT called for the dream task.
    assert fired_original == []


def test_wire_passes_through_non_dream_tasks(tmp_path: Path) -> None:
    """Non-dream tasks are forwarded to the original on_fire_task
    unchanged.
    """
    fired_original: list[Any] = []
    scheduler = _FakeScheduler(on_fire_task=lambda t, r: fired_original.append((t, r)))
    wire_dream_fire_handler(scheduler)

    other_task = type("T", (), {"id": "morning-checkin", "prompt": "x"})()
    scheduler.on_fire_task(other_task, _FakeRun(id="run-y"))

    assert len(fired_original) == 1
    t, r = fired_original[0]
    assert t.id == "morning-checkin"
    assert r.id == "run-y"


def test_wire_works_without_original_handler() -> None:
    """If the scheduler had no original on_fire_task, the wire handler
    still intercepts dream and silently drops non-dream.
    """
    scheduler = _FakeScheduler(on_fire_task=None)
    wire_dream_fire_handler(scheduler)

    dream_task = type("T", (), {"id": DREAM_PERMANENT_TASK_ID, "prompt": "x"})()
    # Should not raise.
    scheduler.on_fire_task(dream_task, _FakeRun())
    # Non-dream — also no raise.
    other_task = type("T", (), {"id": "other", "prompt": "y"})()
    scheduler.on_fire_task(other_task, _FakeRun())


def test_wire_uses_custom_task_id() -> None:
    """``task_id`` override lets the wire handler match a different
    installed id (e.g. for tests).
    """
    fired_original: list[Any] = []
    scheduler = _FakeScheduler(on_fire_task=lambda t, r: fired_original.append(t))
    wire_dream_fire_handler(scheduler, task_id="dream-custom")

    # A task with id="dream-custom" should be intercepted (treated as dream).
    custom = type("T", (), {"id": "dream-custom", "prompt": "x"})()
    scheduler.on_fire_task(custom, _FakeRun())
    assert fired_original == [], "dream-custom should be intercepted"

    # A task with the well-known id="dream" should NOT be intercepted
    # (the custom task_id override changes the match).
    standard = type("T", (), {"id": "dream", "prompt": "x"})()
    scheduler.on_fire_task(standard, _FakeRun())
    assert len(fired_original) == 1
    assert fired_original[0].id == "dream"


# ---------------------------------------------------------------------------
# install_and_wire_dream
# ---------------------------------------------------------------------------


def test_install_and_wire_end_to_end(tmp_path: Path) -> None:
    """Convenience wrapper installs the task AND wires the handler in
    one call; firing the task afterwards runs the local handler.
    """
    reg = RuntimeTaskRegistry()
    init_auto_dream(
        config=DreamConfig(min_hours=0.0, min_sessions=0),
        registry=reg,
    )

    fired_original: list[Any] = []
    scheduler = _FakeScheduler(on_fire_task=lambda t, r: fired_original.append(t))
    task, created = install_and_wire_dream(tmp_path, scheduler, registry=reg)
    assert created is True
    assert task.id == DREAM_PERMANENT_TASK_ID

    # The wire handler is in place — fire the task.
    scheduler.on_fire_task(task, _FakeRun(id="run-end-to-end"))
    assert reg.by_type("dream"), "dream task should have fired"
    assert fired_original == []


def test_install_and_wire_is_idempotent(tmp_path: Path) -> None:
    """Re-running install_and_wire_dream with the same spec leaves the
    task list with exactly one permanent dream task.
    """
    scheduler = _FakeScheduler()
    task1, created1 = install_and_wire_dream(tmp_path, scheduler)
    task2, created2 = install_and_wire_dream(tmp_path, scheduler)
    assert created1 is True
    assert created2 is False
    assert task1.id == task2.id
