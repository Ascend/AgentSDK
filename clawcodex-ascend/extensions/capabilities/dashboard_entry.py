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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
#
"""Agent Dashboard — unified entry & source contracts.

Cross-subsystem contract for TUI ``/dashboard``, Visualizer Web tab, and
Agent ``DashboardList``/``DashboardGet`` tools; implementation lives in
``extensions/agent_dashboard/``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


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
    ``progress_pct`` is normalized on construction: non-numeric / NaN / ±inf
    values are reset to ``0.0`` (with a warning) and out-of-range values are
    clamped to ``[0, 1]``.
    """

    id: str
    source: str
    title: str
    status: str = DASHBOARD_STATUS_PENDING
    detail: str = ""
    # Upstream session/thread identifier (e.g. ThreadGoal.thread_id); ``id``
    # is the dashboard's own synthetic ``f"{source}:..."`` key.
    source_session_id: str | None = None
    progress_pct: float | None = None
    parent_id: str | None = None
    order: int = 0
    tags: list[str] = field(default_factory=list)
    owner: str | None = None
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
                logger.warning(
                    "DashboardEntry progress_pct %r is not numeric; reset to 0.0",
                    self.progress_pct,
                )
                pct = 0.0
            import math

            if not math.isfinite(pct):
                logger.warning(
                    "DashboardEntry progress_pct %r is not finite; reset to 0.0",
                    self.progress_pct,
                )
                pct = 0.0
            elif not 0.0 <= pct <= 1.0:
                logger.warning(
                    "DashboardEntry progress_pct %r out of range; clamping to [0, 1]",
                    self.progress_pct,
                )
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

    def pull(self, **filters: Any) -> list[DashboardEntry]: ...

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
    source: str | None = None,
    status: str | None = None,
    entry_id: str | None = None,
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
