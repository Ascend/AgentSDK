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

"""DashboardEntry/Source/Sink protocols copied here for package self-containment.

Originally this Protocol module lived at
``extensions/capabilities/dashboard_entry.py``. The visualizer package
keeps a local copy so it can run without importing the upstream
``extensions.capabilities`` package.

This is a *local copy*, not a re-export — the upstream module remains
the authoritative reference for ``extensions.agent_dashboard`` and the
orchestrator. Review this local copy against the upstream module every
six months, and propagate signature or semantic changes to both modules
in the same commit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol, runtime_checkable


# Status values that ``DashboardEntry.status`` is allowed to take. We do
# not enum-ify this so individual subsystems can opt in to richer
# vocabularies (e.g. ThreadGoalStatus values) without forcing the
# dashboard to enumerate them — the snapshot layer just stringifies.
DASHBOARD_STATUS_PENDING = "pending"
DASHBOARD_STATUS_IN_PROGRESS = "in_progress"
DASHBOARD_STATUS_COMPLETED = "completed"
DASHBOARD_STATUS_FAILED = "failed"
DASHBOARD_STATUS_BLOCKED = "blocked"
DASHBOARD_STATUSES: frozenset[str] = frozenset(
    {
        DASHBOARD_STATUS_PENDING,
        DASHBOARD_STATUS_IN_PROGRESS,
        DASHBOARD_STATUS_COMPLETED,
        DASHBOARD_STATUS_FAILED,
        DASHBOARD_STATUS_BLOCKED,
    }
)


@dataclass(frozen=True)
class DashboardEntry:
    """A single line on the cross-system dashboard.

    All fields are positional-friendly (id, source, title, status come
    first) so renderers can construct entries with the minimum context
    they already have. The trailing fields (progress_pct, parent_id,
    tags, owner, updated_at_ms, source_session_id) default to
    subsystem-neutral values.

    ``id`` MUST be globally unique across all sources — the dashboard
    uses it as the primary key for ``get_by_id`` lookups and the model
    tool ``DashboardGet``. We recommend ``f"{source}:{native_id}"`` as
    the canonical encoding (e.g. ``"goal:thread-1234"``).
    """

    id: str
    source: str
    title: str
    status: str = DASHBOARD_STATUS_PENDING
    detail: str = ""
    # ``source_session_id`` is the *upstream* session/thread identifier,
    # e.g. ThreadGoal.thread_id. ``id`` is the dashboard's own
    # ``f"{source}:{...}"`` synthetic identifier; consumers should use
    # ``source_session_id`` for cross-referencing the originating
    # subsystem, not ``id``.
    source_session_id: Optional[str] = None
    progress_pct: Optional[float] = None
    parent_id: Optional[str] = None
    order: int = 0
    tags: list[str] = field(default_factory=list)
    owner: Optional[str] = None
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        # ``frozen=True`` dataclasses need ``object.__setattr__`` to
        # normalize the tags tuple, but we want ``list`` for callers
        # that mutate the field. We keep the original list reference
        # by validating in-place instead of replacing.
        if self.tags is None:
            # Defensive: a caller passing ``tags=None`` shouldn't crash
            # older code paths that expect an iterable.
            object.__setattr__(self, "tags", [])
        if self.progress_pct is not None:
            # Normalize out-of-range progress values; store keeps None
            # for "unknown" but a 0..1 float for known percentages.
            try:
                pct = float(self.progress_pct)
            except (TypeError, ValueError):
                object.__setattr__(self, "progress_pct", None)
                return
            # ``math.isfinite`` rejects NaN / +-inf from buggy sources
            # before they propagate through min/max (which return NaN
            # for one of their args, breaking the clamp).
            if not math.isfinite(pct):
                object.__setattr__(self, "progress_pct", None)
                return
            pct = max(0.0, min(1.0, pct))
            object.__setattr__(self, "progress_pct", pct)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the entry.

        Used by the model tools (``DashboardList``, ``DashboardGet``)
        and the NDJSON archiver; we keep the dataclass as the
        in-memory shape and serialize on demand.
        """
        return asdict(self)


@runtime_checkable
class DashboardSource(Protocol):
    """Contract every dashboard data source must satisfy.

    Implementations live in ``extensions/agent_dashboard/sources/`` and
    pull live state from their owning subsystem (goal store, task
    registry, orchestrator status, SOP pipeline, etc.).

    The contract is intentionally minimal:

      * ``source_name`` — unique slug used as the ``source`` field on
        every emitted entry. Must match ``^[a-z][a-z0-9_]*$``.
      * ``pull`` — return a fresh snapshot of all current entries.
        ``filters`` is an optional source-specific bag (e.g. session
        id, status). DashboardStore never enforces keys here; it just
        forwards whatever the consumer asked for.
      * ``cache_ttl_ms`` — how long the snapshot returned by ``pull``
        stays valid. DashboardStore caches per-source and re-pulls
        only when the TTL elapses. High-frequency sources
        (tasks) should return ~1000ms; low-frequency ones (SOP stages)
        can return 30_000ms+.
    """

    @property
    def source_name(self) -> str:
        """Globally-unique identifier (e.g. ``"goal"`` / ``"task"``)."""

    def pull(self, **filters: Any) -> list[DashboardEntry]:
        """Return the current snapshot. May raise on transient errors."""

    @property
    def cache_ttl_ms(self) -> int:
        """Cache lifetime in milliseconds. Default = 5_000 (5s)."""


# A sink is anything that consumes a freshly-computed dashboard
# snapshot. Used by the WebSocket live tail and any other push
# consumer. We type as ``Callable[[list[DashboardEntry]], None]`` so
# both sync and ``async def`` functions are accepted (caller decides
# how to schedule them).
DashboardSink = Callable[[list[DashboardEntry]], None]


def normalize_source_name(name: str) -> str:
    """Coerce ``name`` to a valid source slug.

    Used by the source-registry so accidentally-passed ``"Goals"`` /
    ``" task "`` still register against the canonical
    ``"goal"`` / ``"task"`` keys.
    """
    cleaned = (name or "").strip().lower().replace("-", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def filter_entries(
    entries: Iterable[DashboardEntry],
    *,
    source: Optional[str] = None,
    status: Optional[str] = None,
    entry_id: Optional[str] = None,
) -> list[DashboardEntry]:
    """Apply common cross-source filters to a list of entries.

    Used by ``DashboardStore.get_by_source`` and by the model tool
    ``DashboardList`` to avoid duplicating the same filter logic
    per-source. All filters are AND-composed and ``None`` disables
    a filter.
    """
    out: list[DashboardEntry] = []
    src_norm = normalize_source_name(source) if source else None
    for entry in entries:
        if src_norm is not None and entry.source != src_norm:
            continue
        if status is not None and entry.status != status:
            continue
        if entry_id is not None and entry.id != entry_id:
            continue
        out.append(entry)
    return out


__all__ = [
    "DASHBOARD_STATUS_BLOCKED",
    "DASHBOARD_STATUS_COMPLETED",
    "DASHBOARD_STATUS_FAILED",
    "DASHBOARD_STATUS_IN_PROGRESS",
    "DASHBOARD_STATUS_PENDING",
    "DASHBOARD_STATUSES",
    "DashboardEntry",
    "DashboardSink",
    "DashboardSource",
    "filter_entries",
    "normalize_source_name",
]
