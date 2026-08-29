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

"""Agent-facing tool for inspecting and controlling background sessions.

Supported actions are ``list``, ``inspect``, ``attach``, ``stop``, and
``cleanup``. The tool is synchronous because ``BgSessionManager`` exposes
synchronous operations. When background sessions are disabled, it returns a
disabled result with a runtime-task fallback hint.
"""

from __future__ import annotations

from typing import Any

from clawcodex_ext.tool_system.build_tool import Tool, build_tool
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.errors import ToolInputError
from clawcodex_ext.tool_system.protocol import ToolResult
from clawcodex_ext.tasks.bg_session import (
    BgSessionAttachError,
    BgSessionNotFoundError,
    BgSessionPermissionError,
    BgSessionStopError,
    is_bg_sessions_enabled,
)
from clawcodex_ext.tasks.bg_session_manager import BgSessionManager
from clawcodex_ext.tasks.bg_session_registry import BgSessionRegistry


# ---------------------------------------------------------------------------
# Resolve or create a registry-bound manager from ToolContext
# ---------------------------------------------------------------------------


def _get_manager(context: ToolContext) -> BgSessionManager:
    """Return a cached manager or create one for the context.

    The manager is cached on the context to avoid a scan on every call. Runtime
    tasks are forwarded when the context exposes them.
    """
    cached = getattr(context, "_bg_session_manager", None)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    registry = BgSessionRegistry()
    # The first list or inspect performs a lazy scan of existing markers.
    runtime_tasks = getattr(context, "runtime_tasks", None)
    mgr = BgSessionManager(
        registry=registry,
        runtime_tasks=runtime_tasks,
    )
    try:
        object.__setattr__(context, "_bg_session_manager", mgr)
    except (AttributeError, TypeError):
        pass  # The candidate value is invalid; continue with the existing fallback.
    return mgr


def _safe_workspace(context: ToolContext) -> Any:
    ws = getattr(context, "workspace_root", None)
    if ws is None:
        ws = getattr(context, "cwd", None)
    return ws


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------


def _action_list(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    if not is_bg_sessions_enabled():
        return _disabled_result()
    include_completed = bool(tool_input.get("include_completed", False))
    workspace_only = bool(tool_input.get("workspace_only", False))
    mgr = _get_manager(context)
    # Lazily scan and load the current marker state.
    mgr.registry.scan()
    ws = _safe_workspace(context) if workspace_only else None
    sessions = mgr.list_sessions(include_completed=include_completed, workspace_root=ws)
    return ToolResult(
        name="BgSession",
        output={
            "sessions": [_summarize(s) for s in sessions],
            "count": len(sessions),
        },
    )


def _action_inspect(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    if not is_bg_sessions_enabled():
        return _disabled_result()
    sid = _require_str(tool_input, "bg_session_id")
    mgr = _get_manager(context)
    try:
        sess = mgr.inspect(sid)
    except BgSessionNotFoundError as exc:
        return ToolResult(
            name="BgSession",
            output={
                "error": "not_found",
                "message": str(exc),
                "available": [_summarize(s) for s in mgr.list_sessions()],
            },
            is_error=True,
        )
    return ToolResult(name="BgSession", output=_detail(sess))


def _action_attach(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    if not is_bg_sessions_enabled():
        return _disabled_result()
    sid = _require_str(tool_input, "bg_session_id")
    follow = bool(tool_input.get("follow", True))
    mgr = _get_manager(context)
    ws = _safe_workspace(context)
    try:
        result = mgr.attach(
            sid,
            follow=follow,
            current_workspace=ws,
            allow_cross_workspace=bool(tool_input.get("all_workspaces", False)),
        )
    except BgSessionNotFoundError as exc:
        return ToolResult(
            name="BgSession",
            output={"error": "not_found", "message": str(exc)},
            is_error=True,
        )
    except BgSessionPermissionError as exc:
        return ToolResult(
            name="BgSession",
            output={"error": "permission_denied", "message": str(exc)},
            is_error=True,
        )
    except BgSessionAttachError as exc:
        return ToolResult(
            name="BgSession",
            output={"error": "attach_error", "message": str(exc)},
            is_error=True,
        )
    return ToolResult(
        name="BgSession",
        output={
            "session": _detail(result.session),
            "transcript_tail": result.transcript_tail,
            "resume_hint": result.resume_hint,
        },
    )


def _action_stop(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    if not is_bg_sessions_enabled():
        return _disabled_result()
    sid = _require_str(tool_input, "bg_session_id")
    force = bool(tool_input.get("force", False))
    mgr = _get_manager(context)
    try:
        sess = mgr.stop(sid, force=force)
    except BgSessionNotFoundError as exc:
        return ToolResult(
            name="BgSession",
            output={"error": "not_found", "message": str(exc)},
            is_error=True,
        )
    except BgSessionStopError as exc:
        return ToolResult(
            name="BgSession",
            output={
                "error": "stop_failed",
                "message": str(exc),
                "hint": "retry with force=true to SIGKILL",
            },
            is_error=True,
        )
    return ToolResult(name="BgSession", output={"stopped": True, "session": _detail(sess)})


def _action_cleanup(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    if not is_bg_sessions_enabled():
        return _disabled_result()
    include_failed = bool(tool_input.get("include_failed", False))
    mgr = _get_manager(context)
    removed = mgr.cleanup(include_failed=include_failed)
    return ToolResult(
        name="BgSession",
        output={
            "removed": [_summarize(s) for s in removed],
            "count": len(removed),
        },
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _summarize(s: Any) -> dict[str, Any]:
    return {
        "id": s.id,
        "session_id": s.session_id,
        "status": s.status,
        "pid": s.pid,
        "workspace": str(s.workspace_root),
        "agent_name": s.agent_name,
        "started_at": s.started_at,
    }


def _detail(s: Any) -> dict[str, Any]:
    d = _summarize(s)
    d.update(
        {
            "task_id": s.task_id,
            "team_id": s.team_id,
            "description": s.description,
            "transcript_path": str(s.transcript_path) if s.transcript_path else None,
            "marker_path": str(s.marker_path) if s.marker_path else None,
            "updated_at": s.updated_at,
            "completed_at": s.completed_at,
            "last_activity_at": s.last_activity_at,
            "error": s.error,
        }
    )
    return d


def _disabled_result() -> ToolResult:
    return ToolResult(
        name="BgSession",
        output={
            "disabled": True,
            "message": (
                "BG_SESSIONS is off (CLAWCODEX_BG_SESSIONS=off). "
                "Use TaskList / TaskInspect to inspect runtime tasks instead."
            ),
        },
    )


def _require_str(tool_input: dict[str, Any], key: str) -> str:
    raw = tool_input.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ToolInputError(f"{key} is required (non-empty string)")
    return raw.strip()


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------


def _bg_session_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    action = tool_input.get("action")
    if action not in ("list", "inspect", "attach", "stop", "cleanup"):
        raise ToolInputError("action must be one of: list, inspect, attach, stop, cleanup")
    handlers = {
        "list": _action_list,
        "inspect": _action_inspect,
        "attach": _action_attach,
        "stop": _action_stop,
        "cleanup": _action_cleanup,
    }
    return handlers[action](tool_input, context)


def _bg_session_classifier_input(input_data: dict) -> str:
    return str((input_data or {}).get("action", "list"))


BgSessionTool: Tool = build_tool(
    name="BgSession",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "inspect", "attach", "stop", "cleanup"],
                "description": (
                    "list: list background sessions; inspect: get one session's "
                    "status; attach: tail transcript + resume hint; stop: graceful "
                    "stop (force=true for SIGKILL); cleanup: remove terminal sessions "
                    "from the index."
                ),
            },
            "bg_session_id": {
                "type": "string",
                "description": "Required for inspect/attach/stop.",
            },
            "include_completed": {
                "type": "boolean",
                "default": False,
                "description": "list: include terminal sessions.",
            },
            "workspace_only": {
                "type": "boolean",
                "default": False,
                "description": "list: restrict to current workspace.",
            },
            "follow": {
                "type": "boolean",
                "default": True,
                "description": "attach: whether to follow transcript updates.",
            },
            "all_workspaces": {
                "type": "boolean",
                "default": False,
                "description": "attach: allow cross-workspace attach (default denied).",
            },
            "force": {
                "type": "boolean",
                "default": False,
                "description": "stop: SIGKILL if graceful SIGTERM fails.",
            },
            "include_failed": {
                "type": "boolean",
                "default": False,
                "description": "cleanup: also remove failed sessions.",
            },
        },
        "required": ["action"],
    },
    call=_bg_session_call,
    prompt="""\
List, inspect, attach, stop, or clean up background agent sessions.

Actions:
- list: enumerate background sessions (optionally include_completed, workspace_only)
- inspect: get detailed status of one session (status, pid, transcript path, errors)
- attach: tail the session transcript and get a resume command (cross-workspace denied by default)
- stop: graceful SIGTERM; use force=true for SIGKILL when graceful stop fails
- cleanup: remove terminal/orphaned sessions from the global index (does NOT delete files)

When BG_SESSIONS is off, returns {disabled: true} — fall back to TaskList/TaskInspect.
""",
    description="Query and control background agent sessions.",
    strict=True,
    max_result_size_chars=8000,
    is_read_only=lambda inp: (inp or {}).get("action") in ("list", "inspect"),
    is_concurrency_safe=lambda _inp: True,
    to_auto_classifier_input=_bg_session_classifier_input,
)


__all__ = ["BgSessionTool"]
