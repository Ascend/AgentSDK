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

"""Agent Dashboard — unified entry & source contracts.

Cross-subsystem contract for TUI ``/dashboard``, Visualizer Web tab, and
Agent ``DashboardList``/``DashboardGet`` tools; implementation lives in
``extensions/agent_dashboard/``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol, runtime_checkable


# Status values ``DashboardEntry.status`` may take; not enum-ified so
# subsystems keep richer vocabularies.
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

    Positional-friendly leading fields (id, source, title, status); trailing
    fields default to subsystem-neutral values. ``id`` MUST be globally
    unique across sources — canonical encoding ``f"{source}:{native_id}"``.
    """

    id: str
    source: str
    title: str
    status: str = DASHBOARD_STATUS_PENDING
    detail: str = ""
    # Upstream session/thread identifier (e.g. ThreadGoal.thread_id); ``id``
    # is the dashboard's own synthetic ``f"{source}:..."`` key.
    source_session_id: Optional[str] = None
    progress_pct: Optional[float] = None
    parent_id: Optional[str] = None
    order: int = 0
    tags: list[str] = field(default_factory=list)
    owner: Optional[str] = None
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        # Normalize tags (frozen dataclass -> in-place) and clamp progress
        # to 0..1, rejecting NaN / +-inf and non-numeric values.
        if self.tags is None:
            object.__setattr__(self, "tags", [])
        if self.progress_pct is not None:
            try:
                pct = float(self.progress_pct)
            except (TypeError, ValueError):
                pct = 0.0
            import math

            if not math.isfinite(pct):
                pct = 0.0
            pct = max(0.0, min(1.0, pct))
            object.__setattr__(self, "progress_pct", pct)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view for model tools and the NDJSON archiver."""
        return asdict(self)


@runtime_checkable
class DashboardSource(Protocol):
    """Contract every dashboard data source must satisfy.

    Implementations live in ``extensions/agent_dashboard/sources/``.
    Minimal surface: ``source_name`` (unique ``^[a-z][a-z0-9_]*$`` slug),
    ``pull`` (fresh snapshot; may raise — DashboardStore converts to an
    empty snapshot and logs), ``cache_ttl_ms`` (per-source cache lifetime).
    """

    @property
    def source_name(self) -> str: ...

    def pull(self, **filters: Any) -> list[DashboardEntry]: ...  # pragma: no cover

    @property
    def cache_ttl_ms(self) -> int: ...


# Anything consuming a freshly-computed snapshot (WebSocket live tail,
# other push consumers); typed as Callable so sync and async def both fit.
DashboardSink = Callable[[list[DashboardEntry]], None]


def normalize_source_name(name: str) -> str:
    """Coerce ``name`` to a canonical source slug (e.g. ``"Goals"`` -> ``"goal"``)."""
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
    """AND-composed cross-source filters; ``None`` disables a filter."""
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
    "DASHBOARD_STATUS_IN_PROGRESS",
    "DASHBOARD_STATUS_PENDING",
    "DASHBOARD_STATUSES",
    "DashboardEntry",
    "DashboardSink",
    "DashboardSource",
    "filter_entries",
    "normalize_source_name",
]
