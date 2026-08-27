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

"""Agent Dashboard — cross-system read-only aggregator.

The agent_dashboard package provides a unified, read-only view of
progress data from all agent-loop subsystems (goal, task,
orchestrator, SOP). It is the data layer only — no rendering happens
here. Consumers (TUI ``/dashboard`` command, Visualizer Web tab,
model tools ``DashboardList``/``DashboardGet``) read from the
``DashboardStore`` singleton and decide how to display the data.

Public surface:
  * :class:`DashboardStore` — the aggregate store.
  * :class:`DashboardSourceRegistry` — the source registration
    mechanism (re-exported from ``source_registry`` for convenience).
  * Sources: :class:`GoalDashboardSource`,
    :class:`TasksDashboardSource`.

The store is intentionally kept process-singleton: there is one
in-flight "view of the world" at any given moment per Python
process. Tests can instantiate their own store; production code
should use :func:`get_default_store` so the Visualizer WebSocket
and TUI see the same data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from extensions.capabilities.dashboard_entry import (
    DASHBOARD_STATUSES,
    DashboardEntry,
    DashboardSink,
    DashboardSource,
    filter_entries,
    normalize_source_name,
)

from .source_registry import (
    DashboardSourceRegistry,
    get_default_registry,
    register_dashboard_source,
    unregister_dashboard_source,
)
from .store import (
    DashboardStore,
    get_default_store,
    reset_default_store,
)

__all__ = [
    "DASHBOARD_STATUSES",
    "DashboardEntry",
    "DashboardSink",
    "DashboardSource",
    "DashboardSourceRegistry",
    "DashboardStore",
    "filter_entries",
    "get_default_registry",
    "get_default_store",
    "normalize_source_name",
    "register_dashboard_source",
    "reset_default_store",
    "unregister_dashboard_source",
]


def dashboard_archive_dir(*, home: Optional[Path] = None) -> Path:
    """Return the NDJSON archive directory (``~/.clawcodex/dashboard/``).

    NDJSON files live under this directory, one per source
    (``goal.ndjson`` / ``task.ndjson`` / ...). The path is overridable
    via the ``CLAWCODEX_DASHBOARD_HOME`` env var so tests can
    redirect to a tmp dir.
    """
    env_dir = os.environ.get("CLAWCODEX_DASHBOARD_HOME")
    if env_dir:
        return Path(env_dir).expanduser()
    if home is not None:
        return home / ".clawcodex" / "dashboard"
    return Path.home() / ".clawcodex" / "dashboard"


get_or_create_default_store = get_default_store
