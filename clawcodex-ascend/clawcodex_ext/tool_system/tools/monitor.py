#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Monitor tool — AI-callable long-running background monitor.

``Monitor`` is a built-in tool that starts a background shell task with
``kind='monitor'``.  It reuses ``spawn_background_bash`` and the ``MonitorController`` so its output is written to the same log directory and
is visible via ``TaskOutput`` / ``TaskStop`` and the TUI monitor panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..build_tool import Tool, ValidationResult, build_tool
from ..context import ToolContext
from ..protocol import ToolResult
from clawcodex_ext.feature_gate import get_registry as _get_feature_registry
from clawcodex_ext.permissions.types import (
    PermissionPassthroughResult,
    PermissionResult,
)
from clawcodex_ext.services.monitor.controller import MonitorController
from src.permissions.bash_security import check_bash_command_safety

_MONITOR_TOOL_NAME = "Monitor"


def _monitor_is_enabled() -> bool:
    return _get_feature_registry().is_enabled("MONITOR_TOOL")


def _monitor_check_permissions(
    tool_input: dict[str, Any],
    context: ToolContext,
) -> PermissionResult:
    command = (tool_input or {}).get("command", "")
    if not command:
        return PermissionPassthroughResult()

    cwd_str = str(context.cwd) if context.cwd else None
    result = check_bash_command_safety(command, cwd=cwd_str, shell="bash")
    if result is not None:
        return result
    return PermissionPassthroughResult()


def _monitor_validate_input(
    tool_input: dict[str, Any],
    context: ToolContext,
) -> ValidationResult:
    command = (tool_input or {}).get("command", "")
    if not isinstance(command, str) or not command.strip():
        return ValidationResult.fail("command must be a non-empty string")
    if "\x00" in command:
        return ValidationResult.fail("command contains NUL byte")
    return ValidationResult.ok()


def _monitor_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    command = tool_input.get("command", "")
    if not isinstance(command, str) or not command.strip():
        return ToolResult(
            name=_MONITOR_TOOL_NAME,
            is_error=True,
            output={"message": "command must be a non-empty string"},
        )
    if "\x00" in command:
        return ToolResult(
            name=_MONITOR_TOOL_NAME,
            is_error=True,
            output={"message": "command contains NUL byte"},
        )

    explicit_cwd = tool_input.get("cwd")
    if explicit_cwd is not None:
        if not isinstance(explicit_cwd, str) or not Path(explicit_cwd).expanduser().is_absolute():
            return ToolResult(
                name=_MONITOR_TOOL_NAME,
                is_error=True,
                output={"message": "cwd must be an absolute path when provided"},
            )
        cwd = context.ensure_allowed_path(explicit_cwd)
    else:
        cwd = context.cwd or context.workspace_root

    interval_sec = tool_input.get("interval_sec")
    if interval_sec is not None:
        if not isinstance(interval_sec, int) or interval_sec < 1:
            return ToolResult(
                name=_MONITOR_TOOL_NAME,
                is_error=True,
                output={"message": "interval_sec must be a positive integer"},
            )

    description = tool_input.get("description")
    if description is not None and not isinstance(description, str):
        return ToolResult(
            name=_MONITOR_TOOL_NAME,
            is_error=True,
            output={"message": "description must be a string"},
        )

    ctrl = MonitorController(context)
    result = ctrl.start(
        command=command,
        kind="monitor",
        interval_sec=interval_sec,
        cwd=cwd,
        description=description,
    )

    label = f" ({description})" if description else ""
    return ToolResult(
        name=_MONITOR_TOOL_NAME,
        output={
            "task_id": result.task_id,
            "output_path": str(result.output_path),
            "kind": result.kind,
            "interval_sec": result.interval_sec,
            "message": (
                f"Monitor{label} started (task_id={result.task_id}). "
                f"Output: {result.output_path}. "
                "Use TaskOutput / TaskStop to interact; Shift+Down to view in TUI."
            ),
        },
    )


def _monitor_map_result_to_api(output: Any, tool_use_id: str) -> dict[str, Any]:
    if isinstance(output, dict) and output.get("task_id"):
        content = output.get("message") or f"Monitor started: {output['task_id']}"
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": str(output),
    }


def _monitor_activity(input_data: dict[str, Any] | None) -> str | None:
    if not input_data:
        return "Running monitor"
    cmd = input_data.get("command", "")
    desc = input_data.get("description")
    if desc:
        return f"Monitoring {desc}"
    return f"Monitoring {cmd[:60]}" if cmd else "Running monitor"


def _monitor_user_facing_name(input_data: dict[str, Any] | None) -> str:
    if not input_data:
        return "Monitor"
    cmd = input_data.get("command", "")
    return f"Monitor: {cmd[:50]}" if cmd else "Monitor"


MonitorTool: Tool = build_tool(
    name=_MONITOR_TOOL_NAME,
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run in the background",
            },
            "interval_sec": {
                "type": "integer",
                "description": ("Optional watch interval in seconds (auto-converts to a PowerShell loop on Windows)"),
            },
            "cwd": {
                "type": "string",
                "description": "Working directory (absolute path)",
            },
            "description": {
                "type": "string",
                "description": "Human-readable label for the monitor",
            },
        },
        "required": ["command"],
    },
    call=_monitor_call,
    prompt=lambda: (
        "Start a long-running background monitor task. Returns a task_id; output is streamed to a log file."
    ),
    description=lambda _input: (
        "Start a long-running background monitor task (e.g. tail -f or watch). "
        "Returns the task_id which can be used with TaskOutput / TaskStop. "
        "Output is streamed to a log file viewable via Shift+Down."
    ),
    map_result_to_api=_monitor_map_result_to_api,
    is_enabled=_monitor_is_enabled,
    is_concurrency_safe=lambda _input: False,
    # The command is arbitrary shell, so starting it can mutate the
    # workspace even though execution continues in the background.
    is_read_only=lambda _input: False,
    is_destructive=lambda _input: False,
    check_permissions=_monitor_check_permissions,
    validate_input=_monitor_validate_input,
    user_facing_name=_monitor_user_facing_name,
    to_auto_classifier_input=lambda input_data: (input_data or {}).get("command", ""),
    get_activity_description=_monitor_activity,
    get_tool_use_summary=_monitor_activity,
)


__all__ = ["MonitorTool"]
