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

# AgentSDK migration Parts omit optional package peers; pylint: disable=E0402

"""Route enabled Task-v2 tools through LKB as the single, fail-closed authority.

TaskOutput remains on its independent runtime registry (LKB-ADAPT-014).
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from clawcodex_ext.tool_system.protocol import ToolResult

from .clawcodex_task_projection import build_projection, tool_action as _tool_action
from .clawcodex_task_adapter_host import (
    _SELF_OWNER_SENTINEL,
    _actor,
    _actor_roles,
    _command_id,
    _idempotency_key,
    _metadata_type_error,
    _task_id_of,
    resolve_task_owner,
)
from .error_codes import LkbErrorCode

logger = logging.getLogger(__name__)


class TaskProjectionRefreshError(RuntimeError):
    """The authoritative Store projection could not be refreshed."""


def is_plan_graph_active() -> bool:
    """True when LKB Plan Graph should own Task-v2 writes."""
    try:
        from lkb.flags import is_plan_graph_enabled

        return is_plan_graph_enabled()
    except Exception:
        # Keep configured authority fail-closed when the optional package is unavailable.
        from clawcodex_ext.feature_gate import get_registry, register_defaults

        register_defaults()
        return get_registry().is_enabled("LKB_PLAN_GRAPH")


def _repo():
    from lkb.repository import get_repository

    return get_repository()


def _dispatcher():
    from lkb.plan_graph import plan_command_dispatcher

    return plan_command_dispatcher()


def _service():
    from lkb.application import LkbApplicationService

    return LkbApplicationService(repository=_repo())


def _execute(
    kind: str,
    board_id: str,
    plan_id: str,
    payload: dict[str, Any],
    *,
    actor: str,
    command_id: str,
    reason: str | None = None,
    roles: tuple[str, ...] = (),
):
    from lkb.commands import GraphCommand
    from lkb.refs import plan_task_ref

    svc = _service()
    dispatcher = _dispatcher()
    task_id = str(payload.get("task_id", ""))
    routed_payload = {**payload, "plan_id": plan_id}
    command = GraphCommand(
        command_id=command_id,
        board_id=board_id,
        actor=actor,
        kind=kind,
        primary_subject_ref=plan_task_ref(task_id, graph_id=plan_id) if task_id else None,
        payload=routed_payload,
        reason=reason,
        roles=roles,
    )
    return svc.execute(command, validate=dispatcher.validate, apply=dispatcher.apply)


def _board_id(context: Any) -> str:
    """Resolve the board id for this context's workspace."""
    repo = _repo()
    return repo.resolve_board(
        getattr(context, "workspace_root", None),
        session_id=getattr(context, "session_id", None),
    ).board_id


def _plan_id(context: Any, board_id: str) -> str:
    """Resolve and persist this session's Plan binding inside the Board."""
    from lkb.plan_scope import current_session_id, resolve_plan

    requested = getattr(context, "lkb_plan_id", None)
    if not isinstance(requested, str) or not requested:
        requested = os.getenv("CLAWCODEX_LKB_PLAN_ID") or None
    header = resolve_plan(
        _repo(),
        board_id,
        current_session_id(context),
        requested_plan_id=requested,
    )
    try:
        context.lkb_plan_id = header.plan_id
    # Some host context implementations intentionally reject optional attributes.
    except Exception:  # nosec B110
        pass
    return header.plan_id


def _ensure_task_cutover(context: Any) -> set[str]:
    """Freeze pre-cutover Task-v2 records for this live context only.

    This is intentionally not persisted: native Task-v2 state is host-owned
    and restart restoration does not recreate those host task objects.
    """
    native = getattr(context, "lkb_native_task_ids", None)
    if not isinstance(native, set):
        native = set(native or ())
        setattr(context, "lkb_native_task_ids", native)
    if not bool(getattr(context, "_lkb_task_cutover_initialized", False)):
        tasks = getattr(context, "tasks", {})
        if isinstance(tasks, dict):
            native.update(str(task_id) for task_id, task in tasks.items() if isinstance(task, dict) and task.get("id"))
        setattr(context, "_lkb_task_cutover_initialized", True)
    return native


def _dependency_task_ids(tool_input: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for key in ("addBlocks", "addBlockedBy", "removeBlocks", "removeBlockedBy"):
        raw = tool_input.get(key)
        if isinstance(raw, str) and raw:
            result.add(raw)
        elif isinstance(raw, (list, tuple)):
            result.update(str(value) for value in raw if str(value))
    return result


def _cross_authority_denial(task_id: str, dependency_ids: set[str]) -> ToolResult:
    message = (
        "Task dependencies cannot cross the native Task-v2 and LKB authorities: "
        f"{task_id} -> {', '.join(sorted(dependency_ids))}"
    )
    return ToolResult(
        name="TaskUpdate",
        is_error=True,
        output={
            "success": False,
            "taskId": task_id,
            "updatedFields": [],
            "status": "denied",
            "reason": {
                "code": str(LkbErrorCode.CROSS_AUTHORITY_DEPENDENCY),
                "message": message,
            },
            "nextActions": [],
            "lkb": {
                "decision": "denied",
                "reasonCode": str(LkbErrorCode.CROSS_AUTHORITY_DEPENDENCY),
                "nextActions": [],
            },
        },
    )


def _hydrate(
    context: Any,
    board_id: str,
    plan_id: str,
    *,
    best_effort: bool = False,
) -> dict[str, Any]:
    """Refresh the LKB projection while preserving native cutover tasks.

    TaskGet and TaskList need an authoritative Store read.  UI refreshes can
    opt into ``best_effort`` and retain their existing projection instead.
    """
    try:
        repo = _repo()
        snap = repo.load_snapshot(board_id)
    except Exception as exc:  # noqa: BLE001 - normalize Store adapter boundary
        logger.warning("LKB projection load failed for board %s (%s)", board_id, type(exc).__name__)
        if best_effort:
            return {}
        raise TaskProjectionRefreshError(
            f"LKB projection load failed for board {board_id!r} ({type(exc).__name__})"
        ) from exc
    tasks, lkb_board = build_projection(
        snap,
        plan_id,
        self_owner_sentinel=_SELF_OWNER_SENTINEL,
    )
    native_ids = _ensure_task_cutover(context)
    collisions = native_ids & set(tasks)
    if collisions:
        raise ValueError("task authority collision for ID(s): " + ", ".join(sorted(collisions)))
    if hasattr(context, "tasks"):
        native_tasks = {task_id: task for task_id, task in context.tasks.items() if task_id in native_ids}
        context.tasks.clear()
        context.tasks.update(native_tasks)
        context.tasks.update(tasks)
    return {"lkbBoard": lkb_board}


def refresh_task_projection(context: Any) -> bool:
    """Best-effort read-through refresh for reminder/UI observation boundaries."""
    if not is_plan_graph_active():
        return False
    try:
        _ensure_task_cutover(context)
        board_id = _board_id(context)
        plan_id = _plan_id(context, board_id)
        _hydrate(context, board_id, plan_id, best_effort=True)
    except Exception:
        return False
    return True


def _refresh_after_write(context: Any, board_id: str, plan_id: str) -> str | None:
    """Refresh a native projection without turning a committed write into denial."""
    try:
        _hydrate(context, board_id, plan_id)
    except Exception as exc:  # noqa: BLE001 - preserve durable command result
        logger.warning(
            "LKB projection refresh after write failed for board %s (%s)",
            board_id,
            type(exc).__name__,
        )
        return "projection_refresh_failed"
    return None


def try_handle(tool_name: str, tool_input: dict[str, Any], context: Any) -> tuple[bool, ToolResult | None]:
    """Route a Task-v2 tool call to the LKB Plan Graph.

    Returns ``(True, ToolResult)`` when handled, ``(False, None)`` to fall
    through to the native path.  TaskOutput is never routed here.
    """
    if not is_plan_graph_active():
        return False, None
    if tool_name == "TaskOutput":
        return False, None  # LKB-ADAPT-014
    native_ids = _ensure_task_cutover(context)
    if tool_name in {"TaskGet", "TaskUpdate"}:
        task_id = _task_id_of(tool_input)
        if task_id in native_ids:
            dependency_ids = _dependency_task_ids(tool_input)
            foreign = dependency_ids - native_ids
            if tool_name == "TaskUpdate" and foreign:
                return True, _cross_authority_denial(task_id, foreign)
            return False, None
        if tool_name == "TaskUpdate":
            native_dependencies = _dependency_task_ids(tool_input) & native_ids
            if native_dependencies:
                return True, _cross_authority_denial(task_id, native_dependencies)
    if tool_name == "TaskUpdate":
        tool_input = resolve_task_owner(tool_input, context)

    try:
        board_id = _board_id(context)
        plan_id = _plan_id(context, board_id)
        actor = _actor(context)
        roles = _actor_roles(context)
    except Exception as exc:  # noqa: BLE001 - configured authority is fail-closed
        message = f"LKB board resolution error: {exc}"
        payload = _adapter_error_payload(tool_name, message)
        if tool_name == "TaskCreate":
            payload["task"] = {
                "id": "",
                "subject": str(tool_input.get("subject", "")),
            }
        elif tool_name == "TaskGet":
            payload["task"] = None
        elif tool_name == "TaskUpdate":
            payload["taskId"] = _task_id_of(tool_input)
        return True, ToolResult(name=tool_name, is_error=True, output=payload)

    cid = _command_id(context, tool_name, tool_input, plan_id=plan_id)

    try:
        if tool_name == "TaskCreate":
            return True, _handle_create(tool_input, context, board_id, plan_id, actor, cid)
        if tool_name == "TaskGet":
            return True, _handle_get(tool_input, context, board_id, plan_id)
        if tool_name == "TaskList":
            return True, _handle_list(tool_input, context, board_id, plan_id)
        if tool_name == "TaskUpdate":
            return True, _handle_update(tool_input, context, board_id, plan_id, actor, cid, roles)
    except Exception as exc:  # noqa: BLE001 - adapter must not crash the tool
        # LKB-ADAPT-009: return a denial-shaped payload so legacy clients
        # that ignore ``lkb`` still see a consumable native shape.
        message = f"LKB adapter error: {exc}"
        if tool_name == "TaskCreate":
            payload = {
                "task": {"id": "", "subject": str(tool_input.get("subject", ""))},
                "success": False,
                "status": "denied",
                "reason": {"code": str(LkbErrorCode.ADAPTER_ERROR), "message": message},
                "lkb": {"decision": "denied", "adapterError": message, "toolName": tool_name},
            }
        elif tool_name == "TaskGet":
            payload = {
                "task": None,
                "success": False,
                "status": "denied",
                "reason": {"code": str(LkbErrorCode.ADAPTER_ERROR), "message": message},
                "lkb": {"decision": "denied", "adapterError": message, "toolName": tool_name},
            }
        elif tool_name == "TaskUpdate":
            payload = _adapter_error_payload(tool_name, message)
            payload["taskId"] = _task_id_of(tool_input)
        else:
            payload = {
                "success": False,
                "status": "denied",
                "reason": {"code": str(LkbErrorCode.ADAPTER_ERROR), "message": message},
                "lkb": {"decision": "denied", "adapterError": message, "toolName": tool_name},
            }
        return True, ToolResult(name=tool_name, is_error=True, output=payload)

    return False, None


def _result_error_code(result: Any, fallback: LkbErrorCode) -> str:
    """Read the structured error code; human-readable reasons are never parsed."""

    code = getattr(result, "error_code", None)
    return str(code) if code else str(fallback)


def _recovery_actions(
    task_id: str,
    context: Any,
    error_code: LkbErrorCode | str | None,
) -> list[dict[str, Any]]:
    """Return safe, directly callable recovery actions for a denial."""

    task = context.tasks.get(task_id) if hasattr(context, "tasks") else None
    task = task if isinstance(task, dict) else {}
    status = str(task.get("status") or "")
    owner = task.get("owner")
    code = str(error_code or LkbErrorCode.VALIDATION_DENIED)

    if code == str(LkbErrorCode.ALREADY_RESOLVED) and status == "completed":
        return [
            _tool_action(
                "reopen_task",
                "TaskUpdate",
                {
                    "taskId": task_id,
                    "status": "pending",
                    "reason": "Reopen completed task for additional work.",
                },
                "Explicitly reopen the completed task; LKB will release its old claim "
                "and propagate downstream rechecks.",
            )
        ]
    if (
        code
        in (
            str(LkbErrorCode.OVERRIDE_NOT_AUTHORIZED),
            str(LkbErrorCode.OWNER_REQUIRED),
        )
        and not owner
    ):
        return [
            _tool_action(
                "claim_task",
                "TaskUpdate",
                {"taskId": task_id, "owner": _SELF_OWNER_SENTINEL},
                "Claim with the host-resolved self sentinel; do not invent an Agent id.",
            )
        ]
    if code in (str(LkbErrorCode.INVALID_TRANSITION),) and status == "pending":
        if not owner:
            return [
                _tool_action(
                    "claim_task",
                    "TaskUpdate",
                    {"taskId": task_id, "owner": _SELF_OWNER_SENTINEL},
                    "Claim the pending task before starting it.",
                )
            ]
        return [
            _tool_action(
                "start_task",
                "TaskUpdate",
                {"taskId": task_id, "status": "in_progress"},
                "Move through the required pending to in_progress transition.",
            )
        ]
    if code == str(LkbErrorCode.BLOCKED):
        return [
            _tool_action(
                "explain_blockers",
                "Lkb",
                {"action": "explain", "taskId": task_id},
                "Inspect and complete the active prerequisites before retrying.",
            )
        ]
    return [
        _tool_action(
            "inspect_task",
            "TaskGet",
            {"taskId": task_id},
            "Refresh the authoritative task projection before choosing the next transition.",
        )
    ]


def _lkb_denied_payload(
    result: Any,
    task_id: str = "",
    context: Any | None = None,
) -> dict[str, Any]:
    """Build the ``lkb`` sub-payload for a denied command (spec §6.1, LKB-ADAPT-009).

    LKB-specific audit fields live under ``lkb``; the outer payload preserves
    the native Task-v2 fields so legacy clients that ignore ``lkb`` still see a
    consumable shape.
    """
    payload: dict[str, Any] = {
        "decision": "denied",
        "commandId": getattr(result, "command_id", None),
    }
    validation_run_id = getattr(result, "validation_run_id", None)
    if validation_run_id:
        payload["validationRunId"] = validation_run_id
    if result.reason:
        payload["reason"] = result.reason
    payload["reasonCode"] = _result_error_code(result, LkbErrorCode.VALIDATION_DENIED)
    payload["nextActions"] = (
        _recovery_actions(task_id, context, getattr(result, "error_code", None))
        if task_id and context is not None
        else []
    )
    return payload


def _adapter_error_payload(tool_name: str, message: str) -> dict[str, Any]:
    """Build a denial-shaped payload for an adapter-level exception.

    Follows LKB-ADAPT-009: native fields stay consumable; the LKB-specific
    error detail is isolated under ``lkb``.
    """
    return {
        "success": False,
        "status": "denied",
        "reason": {"code": str(LkbErrorCode.ADAPTER_ERROR), "message": message},
        "lkb": {"decision": "denied", "adapterError": message, "toolName": tool_name},
        "taskId": "",
        "updatedFields": [],
    }


def _handle_create(tool_input, context, board_id, plan_id, actor, cid) -> ToolResult:
    subject = str(tool_input.get("subject", ""))
    description = str(tool_input.get("description", ""))
    active_form = str(tool_input.get("activeForm", "") or subject)
    metadata_error = _metadata_type_error(tool_input)
    if metadata_error is not None:
        # LKB-ADAPT-009: denial keeps the native TaskCreate shape consumable.
        return ToolResult(
            name="TaskCreate",
            is_error=True,
            output={
                "task": {"id": "", "subject": subject},
                "success": False,
                "status": "denied",
                "reason": {
                    "code": str(LkbErrorCode.INVALID_METADATA),
                    "message": metadata_error,
                },
                "lkb": {
                    "decision": "denied",
                    "adapterError": metadata_error,
                    "toolName": "TaskCreate",
                },
            },
        )
    metadata = dict(tool_input.get("metadata", {}) or {})
    key = _idempotency_key(context, tool_input)
    if key:
        # A stable ID makes transport retries reuse the Store result (issue #12, spec §5.10).
        task_id = f"T-{uuid.uuid5(uuid.NAMESPACE_DNS, f'taskcreate/{key}').hex[:8]}"
    else:
        task_id = f"T-{uuid.uuid4().hex[:8]}"
    if task_id in _ensure_task_cutover(context):
        message = f"generated LKB task ID collides with native task {task_id}"
        return ToolResult(
            name="TaskCreate",
            is_error=True,
            output={
                "task": {"id": task_id, "subject": subject},
                "success": False,
                "status": "denied",
                "reason": {
                    "code": str(LkbErrorCode.TASK_ID_COLLISION),
                    "message": message,
                },
                "lkb": {
                    "decision": "denied",
                    "reasonCode": str(LkbErrorCode.TASK_ID_COLLISION),
                    "nextActions": [],
                },
            },
        )
    result = _execute(
        "create_task",
        board_id,
        plan_id,
        {
            "task_id": task_id,
            "subject": subject,
            "description": description,
            "activeForm": active_form,
            "metadata": metadata,
        },
        actor=actor,
        command_id=cid,
    )
    refresh_error = _refresh_after_write(context, board_id, plan_id)
    if result.decision == "committed":
        output: dict[str, Any] = {"task": {"id": task_id, "subject": subject}}
        if refresh_error is not None:
            output["lkb"] = {
                "decision": "committed",
                "projectionRefresh": "failed",
                "refreshError": refresh_error,
            }
        return ToolResult(name="TaskCreate", output=output)
    # LKB-ADAPT-009: keep native TaskCreate fields consumable; LKB detail under ``lkb``.
    return ToolResult(
        name="TaskCreate",
        is_error=True,
        output={
            "task": {"id": task_id, "subject": subject},
            "success": False,
            "status": "denied",
            "reason": {
                "code": _result_error_code(result, LkbErrorCode.CREATE_DENIED),
                "message": result.reason or "create denied",
            },
            "lkb": _lkb_denied_payload(result),
        },
    )


def _handle_get(tool_input, context, board_id, plan_id) -> ToolResult:
    task_id = _task_id_of(tool_input)
    try:
        _hydrate(context, board_id, plan_id)
    except TaskProjectionRefreshError:
        return _projection_unavailable_result("TaskGet")
    task = context.tasks.get(task_id) if hasattr(context, "tasks") else None
    if task is None:
        # LKB-ADAPT-009: native TaskGet returns ``{"task": None}`` when the
        # task is missing — preserve that shape so legacy clients keep working.
        return ToolResult(name="TaskGet", output={"task": None})
    return ToolResult(name="TaskGet", output={"task": task})


def _handle_list(tool_input, context, board_id, plan_id) -> ToolResult:
    try:
        hydration = _hydrate(context, board_id, plan_id)
    except TaskProjectionRefreshError:
        return _projection_unavailable_result("TaskList")
    tasks = list(context.tasks.values()) if hasattr(context, "tasks") else []
    output: dict[str, Any] = {"tasks": tasks}
    # Spec §8.2 / issue #5: top-level lkbBoard aggregate.
    lkb_board = hydration.get("lkbBoard")
    if lkb_board:
        output["lkbBoard"] = lkb_board
    return ToolResult(name="TaskList", output=output)


def _projection_unavailable_result(tool_name: str) -> ToolResult:
    """Return a stable, payload-safe error for an authoritative read failure."""
    message = "The authoritative LKB task projection is temporarily unavailable."
    return ToolResult(
        name=tool_name,
        is_error=True,
        output={
            "success": False,
            "status": "error",
            "reason": {"code": "projection_refresh_failed", "message": message},
            "lkb": {"decision": "error", "projectionRefresh": "failed"},
        },
    )


def _update_command_intent(
    tool_input: dict[str, Any],
    task_id: str,
    actor: str,
) -> tuple[str, dict[str, Any]]:
    """Translate one TaskUpdate input into its prioritized LKB command."""
    payload: dict[str, Any] = {"task_id": task_id}
    status = tool_input.get("status")
    owner = tool_input.get("owner")
    add_blocked_by = tool_input.get("addBlockedBy")
    add_blocks = tool_input.get("addBlocks")
    remove_blocked_by = tool_input.get("removeBlockedBy")
    remove_blocks = tool_input.get("removeBlocks")
    metadata = tool_input.get("metadata")
    for field in ("subject", "description", "activeForm"):
        if field in tool_input:
            payload[field] = tool_input[field]
    if status is not None:
        payload["status"] = str(status)
    if owner is not None:
        payload["owner"] = owner
    for key, value in (
        ("addBlockedBy", add_blocked_by),
        ("addBlocks", add_blocks),
        ("removeBlockedBy", remove_blocked_by),
        ("removeBlocks", remove_blocks),
    ):
        if value:
            payload[key] = list(value) if isinstance(value, list) else [value]
    if metadata is not None:
        payload["metadata"] = dict(metadata or {})

    sub_intent_count = sum(
        bool(value)
        for value in (
            status is not None,
            owner is not None,
            add_blocked_by,
            add_blocks,
            remove_blocked_by,
            remove_blocks,
            metadata is not None,
            any(field in tool_input for field in ("subject", "description", "activeForm")),
        )
    )
    dependency_item_count = sum(
        len(payload.get(key, ())) for key in ("addBlockedBy", "addBlocks", "removeBlockedBy", "removeBlocks")
    )
    owner_is_transfer = owner is not None and str(owner) != actor and str(owner).strip() != ""
    if sub_intent_count > 1 or dependency_item_count > 1:
        return "patch_task", payload
    if owner_is_transfer:
        return "transfer_task", {"task_id": task_id, "new_owner": str(owner)}
    if status is not None:
        return (
            {
                "pending": "reopen_task",
                "in_progress": "start_task",
                "completed": "complete_task",
                "deleted": "delete_task",
            }.get(str(status), "patch_task"),
            payload,
        )
    if owner is not None:
        return ("claim_task" if str(owner).strip() else "release_task"), {"task_id": task_id}
    if add_blocked_by:
        prerequisite = str(add_blocked_by[0] if isinstance(add_blocked_by, list) else add_blocked_by)
        return "add_dependency", {"task_id": task_id, "depends_on": prerequisite}
    if add_blocks:
        blocked = str(add_blocks[0] if isinstance(add_blocks, list) else add_blocks)
        return "add_dependency", {"task_id": blocked, "depends_on": task_id}
    if remove_blocked_by:
        prerequisite = str(remove_blocked_by[0] if isinstance(remove_blocked_by, list) else remove_blocked_by)
        return "remove_dependency", {"task_id": task_id, "depends_on": prerequisite}
    if remove_blocks:
        blocked = str(remove_blocks[0] if isinstance(remove_blocks, list) else remove_blocks)
        return "remove_dependency", {"task_id": blocked, "depends_on": task_id}
    if metadata is not None:
        return "update_task_fields", {"task_id": task_id, "metadata": dict(metadata or {})}
    if any(field in tool_input for field in ("subject", "description", "activeForm")):
        return (
            "update_task_fields",
            {
                "task_id": task_id,
                **{
                    field: tool_input[field]
                    for field in ("subject", "description", "activeForm")
                    if field in tool_input
                },
            },
        )
    return "update_task_fields", payload


def _handle_update(tool_input, context, board_id, plan_id, actor, cid, roles=()) -> ToolResult:
    task_id = _task_id_of(tool_input)
    metadata_error = _metadata_type_error(tool_input)
    if metadata_error is not None:
        # LKB-ADAPT-009: denial keeps the native TaskUpdate shape consumable.
        return ToolResult(
            name="TaskUpdate",
            is_error=True,
            output={
                "success": False,
                "taskId": task_id,
                "updatedFields": [],
                "status": "denied",
                "reason": {
                    "code": str(LkbErrorCode.INVALID_METADATA),
                    "message": metadata_error,
                },
                "nextActions": [
                    _tool_action(
                        "claim_task",
                        "TaskUpdate",
                        {"taskId": task_id, "owner": _SELF_OWNER_SENTINEL},
                        "Use the ownership field, not metadata.owner, to claim the task.",
                    )
                ],
                "lkb": {
                    "decision": "denied",
                    "adapterError": metadata_error,
                    "toolName": "TaskUpdate",
                    "nextActions": [
                        _tool_action(
                            "claim_task",
                            "TaskUpdate",
                            {"taskId": task_id, "owner": _SELF_OWNER_SENTINEL},
                            "Use the ownership field, not metadata.owner, to claim the task.",
                        )
                    ],
                },
            },
        )
    status = tool_input.get("status")
    add_blocked_by = tool_input.get("addBlockedBy")
    add_blocks = tool_input.get("addBlocks")
    remove_blocked_by = tool_input.get("removeBlockedBy")
    remove_blocks = tool_input.get("removeBlocks")
    metadata = tool_input.get("metadata")
    kind, payload = _update_command_intent(tool_input, task_id, actor)

    reason = tool_input.get("reason") or tool_input.get("overrideReason")
    previous_task = context.tasks.get(task_id) if hasattr(context, "tasks") else None
    previous_status = previous_task.get("status") if isinstance(previous_task, dict) else None
    execute_kwargs: dict[str, Any] = {
        "actor": actor,
        "command_id": cid,
        "reason": reason,
    }
    if roles:
        execute_kwargs["roles"] = tuple(roles)
    result = _execute(kind, board_id, plan_id, payload, **execute_kwargs)
    refresh_error = _refresh_after_write(context, board_id, plan_id)
    if result.decision == "committed":
        updated_fields: list[str] = []
        for field in ("subject", "description", "activeForm", "owner"):
            if field in tool_input:
                updated_fields.append(field)
        if status is not None:
            updated_fields.append("deleted" if str(status) == "deleted" else "status")
        if add_blocked_by or remove_blocked_by:
            updated_fields.append("blockedBy")
        if add_blocks or remove_blocks:
            updated_fields.append("blocks")
        if metadata is not None:
            updated_fields.append("metadata")
        updated_fields = list(dict.fromkeys(updated_fields))
        current_task = context.tasks.get(task_id) if hasattr(context, "tasks") else None
        task_projection = (
            {**dict(current_task), "updated": True}
            if isinstance(current_task, dict)
            else {"id": task_id, "updated": True}
        )
        output: dict[str, Any] = {
            "success": True,
            "taskId": task_id,
            "updatedFields": updated_fields,
            "task": task_projection,
        }
        if result.claim_id:
            output["claimId"] = result.claim_id
        if result.affected_refs:
            output["affectedRefs"] = list(result.affected_refs)
        if status is not None and str(status) != "deleted" and previous_status != str(status):
            output["statusChange"] = {"from": str(previous_status), "to": str(status)}
        if refresh_error is not None:
            output["lkb"] = {
                "decision": "committed",
                "projectionRefresh": "failed",
                "refreshError": refresh_error,
            }
        return ToolResult(name="TaskUpdate", output=output)
    # LKB-ADAPT-009: native TaskUpdate denial carries success/taskId/updatedFields
    # at the top level so legacy clients that ignore ``lkb`` still consume the
    # result; LKB-specific audit detail lives under ``lkb``.
    denial_lkb = _lkb_denied_payload(result, task_id, context)
    return ToolResult(
        name="TaskUpdate",
        is_error=True,
        output={
            "success": False,
            "taskId": task_id,
            "updatedFields": [],
            "status": "denied",
            "reason": {
                "code": _result_error_code(result, LkbErrorCode.VALIDATION_DENIED),
                "message": result.reason or "update denied",
            },
            "nextActions": denial_lkb["nextActions"],
            "lkb": denial_lkb,
        },
    )
