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

"""DashboardList — agent-side multi-entry reader for the dashboard.

Read-only counterpart to :class:`TaskList`. Returns a list of
:class:`DashboardEntry` records aggregated across all registered
sources (or a single source if the agent asks for one).

Wire-up:

  * ``input_schema`` accepts an optional ``source`` filter
    (``"goal" | "task" | "orchestrator" | "sop" | "all"``) and
    an optional ``status`` filter (any dashboard status string).
  * ``is_read_only = True``.
  * ``is_concurrency_safe = True`` — the read path is
    read-only-after-cache-fill; the underlying store is
    thread-safe.
  * ``max_result_size_chars = 50_000`` — large enough for a few
    hundred entries. The store caps via per-source TTL so the
    caller doesn't have to.
  * Output is a plain dict ``{"entries": [...], "count": N}``
    so the model can iterate.

The tool resolves the dashboard store via
:func:`extensions.agent_dashboard.get_default_store` by default
but accepts a per-context override through ``context.dashboard_store``
(used by tests and the visualizer).
"""

from __future__ import annotations

import logging
from typing import Any

from extensions.agent_dashboard import get_default_store
from extensions.capabilities.dashboard_entry import (
    DashboardEntry,
    normalize_source_name,
)

from clawcodex_ext.tool_system.build_tool import Tool, build_tool  # pylint: disable=import-error
from clawcodex_ext.tool_system.context import ToolContext  # pylint: disable=import-error
from clawcodex_ext.tool_system.errors import ToolInputError  # pylint: disable=import-error
from clawcodex_ext.tool_system.protocol import ToolResult  # pylint: disable=import-error

logger = logging.getLogger(__name__)

__all__ = ["DashboardListTool"]


def _resolve_store(context: ToolContext) -> Any:
    custom = getattr(context, "dashboard_store", None)
    if custom is not None:
        return custom
    return get_default_store()


def _dashboard_list_call(
    tool_input: dict[str, Any],
    context: ToolContext,
) -> ToolResult:
    source = tool_input.get("source") or "all"
    if not isinstance(source, str):
        raise ToolInputError("source must be a string when provided")
    source_norm = normalize_source_name(source)
    if source_norm not in {"goal", "task", "orchestrator", "sop", "all"}:
        raise ToolInputError("source must be one of: goal|task|orchestrator|sop|all")
    status = tool_input.get("status")
    if status is not None and not isinstance(status, str):
        raise ToolInputError("status must be a string when provided")
    filters: dict[str, Any] = {}
    if source_norm != "all":
        filters["source"] = source_norm
    if status:
        filters["status"] = status
    store = _resolve_store(context)
    try:
        entries = store.snapshot(filters=filters)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("DashboardList: store.snapshot() failed: %s", exc)
        return ToolResult(
            name="DashboardList",
            output={"entries": [], "count": 0, "error": str(exc)},
        )
    serialized = [e.to_dict() for e in entries if isinstance(e, DashboardEntry)]
    return ToolResult(
        name="DashboardList",
        output={"entries": serialized, "count": len(serialized)},
    )


DashboardListTool: Tool = build_tool(
    name="DashboardList",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source": {
                "type": "string",
                "enum": ["goal", "task", "orchestrator", "sop", "all"],
                "default": "all",
            },
            "status": {
                "type": "string",
            },
        },
    },
    call=_dashboard_list_call,
    prompt="""\
Use this tool to list entries on the cross-system task-progress
dashboard. It is the read-only, system-wide counterpart to
``TaskList``: where ``TaskList`` only sees the agent's own
TaskCreate / TaskUpdate tasks, ``DashboardList`` also surfaces
goal state, orchestrator jobs, and SOP stages.

## When to Use This Tool

- Before deciding whether to claim the next ``TaskList`` task,
  check whether the broader system is in a state where claiming
  makes sense (e.g. a goal is paused, a budget is exhausted).
- When the user asks "what's the overall progress?" — this
  returns one entry per active goal / task / orchestrator / SOP
  stage.
- When ``DashboardGet`` is unavailable or you want to scan ids.

## Output

Returns ``{"entries": [...], "count": N}``. Each entry is the
same shape :class:`DashboardEntry` exposes: id, source, title,
status, progress_pct, detail, owner, tags, updated_at_ms.

This tool is **read-only**. Use ``TaskCreate`` / ``TaskUpdate``
for task writes, ``GoalSet`` for goal writes, etc.
""",
    description="List entries on the cross-system task-progress dashboard.",
    strict=True,
    max_result_size_chars=50_000,
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
)
