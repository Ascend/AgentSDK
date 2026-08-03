#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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

"""``dream`` task state machine — F-100.

Mirrors the chapter-10 pattern used by ``local_agent`` / ``in_process_teammate``:
a typed ``DreamTaskState`` dataclass (subclass of ``TaskStateBase``) plus
the lifecycle helpers that ``clawcodex_ext.dreaming.service`` calls into.

Reference (upstream): ``typescript/src/tasks/DreamTask/DreamTask.ts``.

State shape
-----------
In addition to the base fields, the state carries:

* ``phase`` — ``"starting"`` initially, flips to ``"updating"`` the
  first time the watcher observes an Edit / Write tool_use touching a
  path. Mirrors upstream ``DreamPhase``.
* ``sessions_reviewing`` — count of session transcripts scanned.
* ``files_touched`` — deduplicated list of file paths observed in
  Edit / Write tool_use blocks. Best-effort (bash-mediated writes
  won't show up; treat as "at least these were touched", not
  "only these").
* ``turns`` — recent assistant turns (capped) with text + tool-use
  count, used for live display.
* ``abort_event`` — ``asyncio.Event`` for cooperative cancellation.
* ``prior_mtime`` — pre-acquire lock mtime, stashed so ``kill`` can
  rewind it the same way the fork-failure path does.

Hard contract
-------------
Every mutation runs through ``RuntimeTaskRegistry.update`` so the A6 /
C5 rule (mutator must be sync, never ``await`` under the lock) holds
end-to-end. Async work happens outside the lock, exactly like
``kill_async_agent`` does in ``local_agent.py``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TYPE_CHECKING

# pylint: disable=no-name-in-module  # tasks_core: pending patch migration
from clawcodex_ext.tasks_core import (
    TaskStateBase,
    create_task_state_base,
    generate_task_id,
    is_terminal_task_status,
)

# TaskStateBase is supplied by the upstream runtime at integration time;
# Pylint cannot infer the inherited dataclass fields in this standalone tree.
# pylint: disable=unexpected-keyword-arg

if TYPE_CHECKING:
    from src.task_registry import RuntimeTaskRegistry

# Max recent turns to keep in live state (mirrors upstream
# ``DreamTask.ts::MAX_TURNS = 30``). Old turns are dropped in FIFO
# order; we don't summarize.
MAX_DREAM_TURNS = 30

DreamPhase = Literal["starting", "updating"]


@dataclass(kw_only=True)
class DreamTaskState(TaskStateBase):
    """Runtime state for a background ``dream`` task.

    See module docstring for the field-by-field contract.
    """

    type: Literal["dream"] = "dream"  # type: ignore[assignment]
    phase: DreamPhase = "starting"
    sessions_reviewing: int = 0
    files_touched: list[str] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    abort_event: asyncio.Event | None = field(default=None, repr=False, compare=False)
    # Stashed pre-acquire lock mtime — kill rewind uses it.
    prior_mtime: int = 0


def is_dream_task(state: Any) -> bool:
    """Type guard. Tolerant of ``None`` and arbitrary objects."""
    return isinstance(state, DreamTaskState)


def is_dream_task_terminal(state: Any) -> bool:
    """Terminal-state predicate scoped to dream tasks."""
    return is_dream_task(state) and is_terminal_task_status(state.status)


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------


def register_dream_task(
    *,
    sessions_reviewing: int,
    prior_mtime: int,
    output_file: str = "",
    registry: "RuntimeTaskRegistry",
) -> str:
    """Register a brand-new background dream on the runtime registry.

    Returns the freshly-generated task id. Idempotent: re-registering
    with the same id replaces the prior entry (the manual ``/dream``
    path can call this more than once safely).

    The output_file is reserved for chapter-fidelity with
    ``LocalAgentTaskState``; dream tasks do not currently stream output
    to disk (their transcripts are written by the underlying agent
    runner), so callers usually pass ``""``.
    """
    task_id = generate_task_id("dream")
    state = DreamTaskState(
        id=task_id,
        type="dream",
        status="running",
        description="dreaming",
        start_time=time.time(),
        output_file=output_file,
        phase="starting",
        sessions_reviewing=sessions_reviewing,
        files_touched=[],
        turns=[],
        prior_mtime=prior_mtime,
    )
    registry.upsert(state)
    return task_id


def add_dream_turn(
    task_id: str,
    *,
    text: str,
    tool_use_count: int,
    touched_paths: list[str] | None = None,
    registry: "RuntimeTaskRegistry",
) -> None:
    """Append a single assistant turn to the live state.

    Mirrors upstream ``addDreamTurn``. Skips the update entirely if
    the turn is empty AND nothing new was touched (pure no-op — keeps
    re-renders down).

    Idempotent on the touched-paths set: a path seen before is not
    re-appended, but the new path is still included in the dedup
    decision so the turn is recorded.
    """
    new_touched = touched_paths or []
    applied = False

    def _mutate(prev: TaskStateBase) -> TaskStateBase:
        nonlocal applied
        if not isinstance(prev, DreamTaskState):
            return prev
        if prev.status != "running":
            return prev
        seen = set(prev.files_touched)
        fresh = [p for p in new_touched if not (p in seen or seen.add(p))]
        if not text and tool_use_count == 0 and not fresh:
            return prev
        applied = True
        new_phase: DreamPhase = "updating" if fresh else prev.phase
        new_files = list(prev.files_touched) + fresh
        new_turns = (prev.turns + [{"text": text, "tool_use_count": tool_use_count}])[-MAX_DREAM_TURNS:]
        return replace(
            prev,
            phase=new_phase,
            files_touched=new_files,
            turns=new_turns,
        )

    registry.update(task_id, _mutate)


def _terminal_replace(
    prev: DreamTaskState,
    *,
    status: str,
) -> DreamTaskState:
    """Compose the standard terminal-state mutation: status flip +
    end_time stamp + abort_event drop. Pure helper.
    """
    moment = time.time()
    return replace(
        prev,
        status=status,  # type: ignore[arg-type]
        end_time=moment,
        abort_event=None,
    )


def complete_dream_task(
    task_id: str,
    registry: "RuntimeTaskRegistry",
) -> None:
    """Flip status to ``completed``, stamp end_time, drop abort_event.

    Mirrors upstream ``completeDreamTask`` — notified defaults to True
    (dream has no model-facing notification path; the inline
    ``appendSystemMessage`` completion note is the user surface).
    """

    def _complete(prev: TaskStateBase) -> TaskStateBase:
        if not isinstance(prev, DreamTaskState):
            return prev
        if is_terminal_task_status(prev.status):
            return prev
        return _terminal_replace(prev, status="completed")

    registry.update(task_id, _complete)


def fail_dream_task(
    task_id: str,
    registry: "RuntimeTaskRegistry",
) -> None:
    """Flip status to ``failed``, stamp end_time, drop abort_event."""

    def _fail(prev: TaskStateBase) -> TaskStateBase:
        if not isinstance(prev, DreamTaskState):
            return prev
        if is_terminal_task_status(prev.status):
            return prev
        return _terminal_replace(prev, status="failed")

    registry.update(task_id, _fail)


def rollback_dream_lock_after_kill(
    task_id: str,
    registry: "RuntimeTaskRegistry",
) -> int | None:
    """Extract the stashed prior mtime from a dream task and clear the
    abort_event. Returns the prior mtime (or ``None`` if the task was
    already terminal / not found). Caller is responsible for actually
    rewinding the lock file via ``clawcodex_ext.dreaming.lock`` —
    keeping the I/O out of this module avoids a circular import.

    Mirrors upstream ``DreamTask.kill``'s pattern of stashing
    ``priorMtime`` on the state so the rewind can be replayed after
    the registry mutator returns.
    """
    captured: int | None = None

    def _mutate(prev: TaskStateBase) -> TaskStateBase:
        nonlocal captured
        if not isinstance(prev, DreamTaskState):
            return prev
        if is_terminal_task_status(prev.status):
            return prev
        captured = prev.prior_mtime
        return _terminal_replace(prev, status="killed")

    registry.update(task_id, _mutate)
    return captured


# ---------------------------------------------------------------------------
# Task adapter — polymorphic kill dispatch
# ---------------------------------------------------------------------------


class DreamTask:
    """``Task`` adapter for ``dream`` entries.

    The minimal ``kill`` body delegates to
    :func:`rollback_dream_lock_after_kill` so the registry update +
    abort signal + prior-mtime capture stay in one place. The actual
    lock-file rewind is performed by the service module's kill path
    (after the registry mutator returns) to keep I/O out of this
    module.
    """

    name: str = "DreamTask"
    type: Literal["dream"] = "dream"

    async def kill(self, task_id: str, registry: "RuntimeTaskRegistry") -> None:
        rollback_dream_lock_after_kill(task_id, registry)


__all__ = [
    "DreamTask",
    "DreamTaskState",
    "DreamPhase",
    "MAX_DREAM_TURNS",
    "is_dream_task",
    "is_dream_task_terminal",
    "register_dream_task",
    "add_dream_turn",
    "complete_dream_task",
    "fail_dream_task",
    "rollback_dream_lock_after_kill",
]


# ``create_task_state_base`` re-exported for tests / downstream helpers
# that want the chapter-10 factory but the dream-typed state.
__all__ += ["create_task_state_base"]
