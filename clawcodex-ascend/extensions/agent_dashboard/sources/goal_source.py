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

"""GoalDashboardSource — bridges :class:`GoalService` to the dashboard.

The plan §3.1 says this source reads from a
``GoalStateRegistry`` and emits one entry per session with an active
``GoalState``. The actual goal implementation uses a SQLite
``GoalStore`` keyed by thread_id, accessed through
``GoalService``. There is no in-memory registry of *all* live
goals because most processes only have one thread; we therefore
accept a ``thread_ids_provider`` callable that yields the threads
the source should query.

Design notes:

  * ``source_name = "goal"`` — the canonical id consumed by
    :class:`DashboardSourceRegistry` and the TUI/Visualizer.
  * ``cache_ttl_ms = 5_000`` — goals change at most once per turn
    boundary, so a 5-second cache is plenty.
  * The source is purely read-only: it never calls
    ``service.create_goal`` / ``service.update_goal``. Any change
    flows through the existing ``/goal`` command and the
    ``Goal*`` model tools.
  * ``progress_pct`` is computed as ``tokens_used / token_budget``
    when a budget exists, otherwise ``None`` (the renderer falls
    back to "no budget" formatting).
  * Mapping from ``ThreadGoalStatus`` -> dashboard status is
    stable; see ``_GOAL_STATUS_MAP``.

A ``None`` thread id is filtered out so callers can pass a wide
provider (e.g. ``lambda: get_active_thread_ids()``) without
needing to guard against optionals in their callback.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterable, Optional

from extensions.capabilities.dashboard_entry import (
    DASHBOARD_STATUS_BLOCKED,
    DASHBOARD_STATUS_COMPLETED,
    DASHBOARD_STATUS_IN_PROGRESS,
    DASHBOARD_STATUS_PENDING,
    DashboardEntry,
)

logger = logging.getLogger(__name__)

__all__ = ["GoalDashboardSource"]


# ``ThreadGoalStatus`` -> dashboard status. Mapped by value (the
# upstream enum) so we don't need a hard import here; the import
# below is intentionally lazy so this module stays import-safe
# even when the goal package is unavailable (e.g. spec
# boundary tests).
_GOAL_STATUS_MAP: dict[str, str] = {
    "active": DASHBOARD_STATUS_IN_PROGRESS,
    "paused": DASHBOARD_STATUS_PENDING,
    "blocked": DASHBOARD_STATUS_BLOCKED,
    "usage_limited": DASHBOARD_STATUS_BLOCKED,
    "budget_limited": DASHBOARD_STATUS_BLOCKED,
    "complete": DASHBOARD_STATUS_COMPLETED,
}


def _to_ms(value: Any) -> int:
    """Coerce a datetime / int / float into a millisecond timestamp."""
    if value is None:
        return 0
    # datetime-like objects expose ``timestamp()`` returning seconds.
    ts_fn = getattr(value, "timestamp", None)
    if callable(ts_fn):
        try:
            return int(ts_fn() * 1000)
        except Exception:  # pragma: no cover - defensive
            return 0
    try:
        return int(float(value) * 1000) if float(value) < 1e12 else int(float(value))
    except (TypeError, ValueError):
        return 0


class GoalDashboardSource:
    """Read-only :class:`DashboardSource` backed by :class:`GoalService`.

    Parameters
    ----------
    service:
        A :class:`clawcodex_ext.goal.service.GoalService` (or any
        duck-typed equivalent with ``get_goal(thread_id)`` and
        ``_ensure_enabled``). Tests can pass a simple object.
    thread_ids_provider:
        Zero-arg callable returning the iterable of thread ids to
        query on each :meth:`pull`. The default returns an empty
        list so a misconfigured deployment doesn't accidentally
        surface the *current* process's goal as if it were the
        only goal in the world — callers must opt in to a real
        provider.
    cache_ttl_ms:
        See :class:`DashboardSource`. 5 s default.
    """

    source_name = "goal"

    def __init__(
        self,
        service: Any,
        thread_ids_provider: Optional[Callable[[], Iterable[str]]] = None,
        *,
        cache_ttl_ms: int = 5_000,
    ) -> None:
        self._service = service
        self._thread_ids_provider = thread_ids_provider or (lambda: ())
        self._cache_ttl_ms = int(cache_ttl_ms)

    @property
    def cache_ttl_ms(self) -> int:
        return self._cache_ttl_ms

    def pull(self, **filters: Any) -> list[DashboardEntry]:
        """Return a dashboard entry for every active goal thread."""
        try:
            thread_ids = list(self._thread_ids_provider() or ())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("goal_source thread_ids_provider failed: %s", exc)
            return []
        entries: list[DashboardEntry] = []
        seen: set[str] = set()
        for raw_id in thread_ids:
            if raw_id is None:
                continue
            thread_id = str(raw_id).strip()
            if not thread_id or thread_id in seen:
                continue
            seen.add(thread_id)
            goal = self._safe_get_goal(thread_id)
            if goal is None:
                continue
            entries.append(self._to_entry(goal))
        return entries

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _safe_get_goal(self, thread_id: str) -> Any:
        """Fetch a goal; never raise (we drop the entry on error)."""
        get_goal = getattr(self._service, "get_goal", None)
        if not callable(get_goal):
            return None
        try:
            return get_goal(thread_id)
        except Exception as exc:
            logger.debug("goal_source get_goal(%s) failed: %s", thread_id, exc)
            return None

    def _to_entry(self, goal: Any) -> DashboardEntry:
        """Convert a ``ThreadGoal``-shaped object into a DashboardEntry.

        We don't import ``ThreadGoal`` directly — fields are
        duck-typed, so tests can pass a ``SimpleNamespace`` with
        the same attributes and the conversion Just Works.
        """
        objective = str(getattr(goal, "objective", "") or "").strip() or "(no objective)"
        thread_id = str(getattr(goal, "thread_id", "") or "")
        goal_id = str(getattr(goal, "goal_id", "") or "")
        status_obj = getattr(goal, "status", None)
        # ``status`` may be an enum with ``.value`` or a raw string.
        status_value = getattr(status_obj, "value", None) or str(status_obj or "")
        dashboard_status = _GOAL_STATUS_MAP.get(status_value, DASHBOARD_STATUS_PENDING)
        tokens_used = int(getattr(goal, "tokens_used", 0) or 0)
        token_budget = getattr(goal, "token_budget", None)
        progress_pct: Optional[float] = None
        if token_budget:
            try:
                budget_int = int(token_budget)
                if budget_int > 0:
                    progress_pct = tokens_used / budget_int
            except (TypeError, ValueError):
                progress_pct = None
        updated_at = _to_ms(getattr(goal, "updated_at", None))
        if updated_at <= 0:
            updated_at = int(time.time() * 1000)
        detail = self._format_detail(goal)
        return DashboardEntry(
            id=f"goal:{thread_id}" if thread_id else f"goal:{goal_id}",
            source="goal",
            title=objective,
            status=dashboard_status,
            detail=detail,
            source_session_id=thread_id or None,
            progress_pct=progress_pct,
            tags=["goal"],
            owner=None,
            updated_at_ms=updated_at,
        )

    @staticmethod
    def _format_detail(goal: Any) -> str:
        """Build the human-readable detail line.

        Mirrors the format used by ``/goal status`` in
        ``clawcodex_ext.goal.command._format_goal_summary`` so
        users see consistent text between the dedicated command
        and the dashboard.
        """
        tokens_used = int(getattr(goal, "tokens_used", 0) or 0)
        time_used = int(getattr(goal, "time_used_seconds", 0) or 0)
        budget = getattr(goal, "token_budget", None)
        if budget is not None:
            try:
                budget_int = int(budget)
            except (TypeError, ValueError):
                budget_int = None
        else:
            budget_int = None
        if budget_int is not None and budget_int > 0:
            return f"{tokens_used} / {budget_int} tokens ({time_used}s)"
        return f"{tokens_used} tokens ({time_used}s)"
