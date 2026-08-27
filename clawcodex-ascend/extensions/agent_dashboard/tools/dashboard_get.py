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

"""DashboardGet — agent-side single-entry reader for the dashboard.

Read-only counterpart to :class:`TaskGet`. Returns one
:class:`DashboardEntry` (or ``None``) for the given id, with the
same shape ``/dashboard --id`` would print in the TUI.

Wire-up:

  * ``input_schema`` requires ``entry_id`` (string) and
    optionally accepts ``source`` (a coarse hint; if the id is
    already globally unique we just resolve it directly).
  * ``is_read_only = True`` — the model can't use this tool to
    mutate any subsystem.
  * ``is_concurrency_safe = True`` — pulling a cached entry
    doesn't touch any shared mutable state.
  * ``max_result_size_chars = 4_000`` — a single entry is small;
    the limit just guards against a future entry that
    accidentally embeds a huge ``detail`` blob.

The tool resolves the dashboard store via
:func:`extensions.agent_dashboard.get_default_store` by default
but accepts a per-call override through
``context.options.refresh_tools`` is irrelevant here — the
real override lives in :data:`extensions.agent_dashboard.store.DashboardStore`
which the REPL/TUI may set on the context.
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

__all__ = ["DashboardGetTool"]


def _resolve_store(context: ToolContext) -> Any:
    """Return the active :class:`DashboardStore`.

    The default store is process-wide; callers that want
    isolation (tests, agent-loop sub-agents) can stash a custom
    store on ``context.dashboard_store`` and we honour that.
    """
    custom = getattr(context, "dashboard_store", None)
    if custom is not None:
        return custom
    return get_default_store()


def _dashboard_get_call(
    tool_input: dict[str, Any],
    context: ToolContext,
) -> ToolResult:
    entry_id = tool_input.get("entry_id")
    if not isinstance(entry_id, str) or not entry_id.strip():
        raise ToolInputError("entry_id must be a non-empty string")
    entry_id = entry_id.strip()
    source = tool_input.get("source")
    if source is not None and not isinstance(source, str):
        raise ToolInputError("source must be a string when provided")
    normalized_source = normalize_source_name(source) if source else None
    store = _resolve_store(context)
    try:
        # Fast path: get_by_id. If the caller hinted a source we
        # also try get_by_source as a fallback in case the id is
        # scoped (e.g. multiple "task:1" ids across sources).
        entry = store.get_by_id(entry_id)
        if entry is None and normalized_source:
            for candidate in store.get_by_source(normalized_source):
                if candidate.id == entry_id:
                    entry = candidate
                    break
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("DashboardGet: store lookup failed: %s", exc)
        return ToolResult(
            name="DashboardGet",
            output={"entry": None, "error": f"dashboard lookup failed: {exc}"},
        )
    payload: dict[str, Any] = {"entry": entry.to_dict() if isinstance(entry, DashboardEntry) else None}
    if entry is None and normalized_source:
        payload["hint"] = f"no entry with id={entry_id!r} in source={normalized_source!r}"
    return ToolResult(name="DashboardGet", output=payload)


DashboardGetTool: Tool = build_tool(
    name="DashboardGet",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "entry_id": {"type": "string"},
            "source": {
                "type": "string",
                "enum": ["goal", "task", "orchestrator", "sop", "all"],
            },
        },
        "required": ["entry_id"],
    },
    call=_dashboard_get_call,
    prompt="""\
Use this tool to fetch a single dashboard entry by id.

## When to Use This Tool

- After ``DashboardList`` returns an id you want to inspect.
- When the user asks for the full status of a specific goal /
  task / orchestrator job / SOP stage.

## Output

Returns a single :class:`DashboardEntry` shape with the same
fields ``DashboardList`` exposes, or ``entry: null`` if no
entry matches.

This tool is **read-only**. It does not create, update, or
delete any state — use ``TaskCreate`` / ``TaskUpdate`` /
``GoalSet`` / etc. for writes.
""",
    description="Fetch a single dashboard entry by id.",
    strict=True,
    max_result_size_chars=4_000,
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
)
