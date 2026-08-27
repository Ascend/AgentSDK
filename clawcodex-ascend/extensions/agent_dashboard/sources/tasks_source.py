#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""TasksDashboardSource — bridges ``ToolContext.tasks`` to the dashboard.

The plan §3.2 says this source reads from
``ToolContext.tasks`` and emits one entry per task. The actual
shape (ch17 / tasks_v2) is::

    context.tasks[task_id] = {
        "id": "...",
        "subject": "...",
        "description": "...",
        "activeForm": "...",
        "status": "pending" | "in_progress" | "completed" | ...,
        "owner": str | None,
        "blocks": [...],
        "blockedBy": [...],
        "metadata": {...},
        "output": "...",
    }

We treat the dict as duck-typed: any object with the relevant
keys works, so tests can pass ``SimpleNamespace`` instances and
the conversion is decoupled from the actual ``TaskCreate`` /
``TaskUpdate`` tool implementations.

Design notes:

  * ``source_name = "task"``.
  * ``cache_ttl_ms = 1_000`` — tasks mutate at task-update
    granularity (sub-second in long runs), so we re-pull more
    eagerly than the goal source.
  * No write-back: the source never mutates the dict it reads.
  * ``parent_id`` is left ``None`` for now; the plan
    explicitly defers cross-task parent linking to a later
    feature.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from extensions.capabilities.dashboard_entry import (
    DASHBOARD_STATUS_BLOCKED,
    DASHBOARD_STATUS_COMPLETED,
    DASHBOARD_STATUS_FAILED,
    DASHBOARD_STATUS_IN_PROGRESS,
    DASHBOARD_STATUS_PENDING,
    DashboardEntry,
)

logger = logging.getLogger(__name__)

__all__ = ["TasksDashboardSource"]


# Task status values emitted by ``TaskCreate`` / ``TaskUpdate`` in
# ``clawcodex_ext.tool_system.tools.tasks_v2``. We map them onto
# the dashboard's 5-state vocabulary.
_TASK_STATUS_MAP: dict[str, str] = {
    "pending": DASHBOARD_STATUS_PENDING,
    "in_progress": DASHBOARD_STATUS_IN_PROGRESS,
    "completed": DASHBOARD_STATUS_COMPLETED,
    "done": DASHBOARD_STATUS_COMPLETED,
    "failed": DASHBOARD_STATUS_FAILED,
    "blocked": DASHBOARD_STATUS_BLOCKED,
    "cancelled": DASHBOARD_STATUS_FAILED,
}


def _coerce_tasks_dict(provider: Any) -> dict[str, dict[str, Any]]:
    """Read a tasks dict from a ToolContext-like object.

    Returns an empty dict on any error (missing attribute, None
    value, wrong type). The source is read-only so callers
    should never see exceptions propagating out of
    :meth:`pull`.
    """
    if provider is None:
        return {}
    tasks_attr = getattr(provider, "tasks", None)
    if tasks_attr is None:
        # Allow passing the dict directly.
        if isinstance(provider, dict):
            tasks_attr = provider
        else:
            return {}
    if not isinstance(tasks_attr, dict):
        return {}
    return dict(tasks_attr)


class TasksDashboardSource:
    """Read-only :class:`DashboardSource` backed by ``ToolContext.tasks``.

    Parameters
    ----------
    tool_context_provider:
        Zero-arg callable returning the live ``ToolContext`` (or
        its tasks dict). We accept a callable rather than the
        context itself so the source can be constructed at
        import time, before the REPL/TUI is up. The default
        returns ``None`` so a misconfigured deployment doesn't
        crash — pull() simply yields zero entries.
    cache_ttl_ms:
        1 second by default; tasks mutate at task-update
        granularity.
    """

    source_name = "task"

    def __init__(
        self,
        tool_context_provider: Optional[Callable[[], Any]] = None,
        *,
        cache_ttl_ms: int = 1_000,
    ) -> None:
        self._provider = tool_context_provider or (lambda: None)
        self._cache_ttl_ms = int(cache_ttl_ms)

    @property
    def cache_ttl_ms(self) -> int:
        return self._cache_ttl_ms

    def pull(self, **filters: Any) -> list[DashboardEntry]:
        """Return one :class:`DashboardEntry` per task."""
        try:
            ctx = self._provider()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("tasks_source provider raised: %s", exc)
            return []
        tasks = _coerce_tasks_dict(ctx)
        entries: list[DashboardEntry] = []
        for tid, task in tasks.items():
            entry = self._to_entry(str(tid), task)
            if entry is not None:
                entries.append(entry)
        return entries

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_entry(task_id: str, task: Any) -> Optional[DashboardEntry]:
        if task is None:
            return None
        if not isinstance(task, dict):
            # Be lenient: try to read dict-like attributes.
            task = {
                "id": getattr(task, "id", task_id),
                "subject": getattr(task, "subject", ""),
                "description": getattr(task, "description", ""),
                "status": getattr(task, "status", "pending"),
                "owner": getattr(task, "owner", None),
                "updated_at_ms": getattr(task, "updated_at_ms", 0),
                "tags": getattr(task, "tags", []) or [],
            }
        subject = str(task.get("subject") or task.get("activeForm") or "").strip()
        if not subject:
            subject = task_id
        description = str(task.get("description") or "")
        raw_status = str(task.get("status") or "pending")
        status = _TASK_STATUS_MAP.get(raw_status.lower(), DASHBOARD_STATUS_PENDING)
        owner = task.get("owner")
        if owner is not None:
            owner = str(owner)
        updated_at = int(task.get("updated_at_ms") or 0)
        if updated_at <= 0:
            updated_at = int(time.time() * 1000)
        tags_raw = task.get("tags") or []
        tags: list[str] = [str(t) for t in tags_raw] if tags_raw else []
        if blocked := task.get("blockedBy"):
            tags.append("blocked_by:" + ",".join(str(b) for b in blocked))
        return DashboardEntry(
            id=f"task:{task_id}",
            source="task",
            title=subject,
            status=status,
            detail=description,
            source_session_id=None,
            progress_pct=None,
            parent_id=None,
            order=int(task.get("order", 0) or 0),
            tags=tags,
            owner=owner or None,
            updated_at_ms=updated_at,
        )
