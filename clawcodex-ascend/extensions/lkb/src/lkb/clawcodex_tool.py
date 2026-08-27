#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Agent-callable LKB command surface.

Slash commands are consumed by the interactive host before a prompt reaches
the model. This tool exposes the same non-interactive command handlers to an
Agent without duplicating Board reads or mutation logic.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from clawcodex_ext.tool_system.build_tool import build_tool
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.errors import ToolInputError
from clawcodex_ext.tool_system.protocol import ToolResult


_READ_ACTIONS = {"board", "status", "explain", "audit", "plan_current", "plan_list"}
_TASK_ACTIONS = {"explain", "audit", "revalidate"}


def _is_enabled() -> bool:
    try:
        from lkb.flags import is_plan_graph_enabled

        return is_plan_graph_enabled()
    except ModuleNotFoundError as exc:
        if exc.name in {"lkb", "lkb.flags"}:
            return False
        raise


def _command_args(tool_input: dict[str, Any]) -> str:
    action = str(tool_input.get("action") or "").strip()
    task_id = str(tool_input.get("taskId") or "").strip()
    if action in _TASK_ACTIONS and not task_id:
        raise ToolInputError(f"taskId is required for Lkb action {action!r}")

    if action == "board":
        return "board --compact" if bool(tool_input.get("compact")) else "board"
    if action in ("status",):
        return action
    if action in ("explain", "audit", "revalidate"):
        return f"{action} {task_id}"
    if action == "plan_current":
        return "plan current"
    if action == "plan_list":
        return "plan list"
    raise ToolInputError(f"Unsupported Lkb action: {action!r}")


def _lkb_tool_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    from lkb.clawcodex_commands import _lkb_call

    args = _command_args(tool_input)
    result = _lkb_call(args, SimpleNamespace(tool_context=context))
    text = result.text
    return ToolResult(
        name="Lkb",
        output={
            "command": f"/lkb {args}",
            "success": result.success,
            "text": text,
            "error_code": result.error_code,
        },
        is_error=not result.success,
    )


LkbTool = build_tool(
    name="Lkb",
    input_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "board",
                    "status",
                    "explain",
                    "audit",
                    "revalidate",
                    "plan_current",
                    "plan_list",
                ],
                "description": "LKB view or mutation to execute.",
            },
            "taskId": {
                "type": "string",
                "description": "Required for explain, audit, and revalidate.",
            },
            "compact": {
                "type": "boolean",
                "default": False,
                "description": "Render the board in compact mode.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
    call=_lkb_tool_call,
    prompt="""\
Inspect and operate the active Logical Kanban Board through its public command surface.

Use board/status for the current task graph, explain for blockers and recovery guidance,
and audit for the recorded task history. Use revalidate after an invalidated task's
upstream dependencies are current. Task ownership and status changes still go through
TaskUpdate; claim yourself with owner="$self".
""",
    description=lambda tool_input: f"LKB {tool_input.get('action', '')}".strip(),
    strict=True,
    max_result_size_chars=20_000,
    is_enabled=_is_enabled,
    is_read_only=lambda tool_input: str(tool_input.get("action") or "") in _READ_ACTIONS,
    is_concurrency_safe=lambda tool_input: str(tool_input.get("action") or "") in _READ_ACTIONS,
)


__all__ = ["LkbTool"]
