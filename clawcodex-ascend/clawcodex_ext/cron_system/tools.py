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

"""Downstream Cron tool implementations backed by persistent storage.

Implements (kill switch) and (rich prompt docs).
"""

from __future__ import annotations

from typing import Any

from src.tool_system.build_tool import Tool, build_tool
from src.tool_system.context import ToolContext
from src.tool_system.errors import ToolInputError
from clawcodex_ext.tool_system.protocol import ToolResult

from .models import (
    CronTask,
    is_cron_disabled,
)
from .parser import cron_to_human, parse_cron_expression
from .schedule import get_cron_task_detail, manual_fire_cron_task
from .tasks import add_cron_task, read_all_cron_tasks, remove_cron_tasks

# keep in sync with `is_cron_disabled` for the in-process fast path.
CRON_DISABLED_MESSAGE = "Cron is disabled (CLAWCODEX_DISABLE_CRON is set)."

CRON_CREATE_PROMPT = """\
Schedule a recurring or one-shot prompt to run via the cron scheduler.

# Cron expression

Five fields, local time: `minute hour day-of-month month day-of-week`.
Examples:
  - `*/5 * * * *` — every 5 minutes
  - `0 9 * * 1-5` — 09:00 on weekdays
  - `0 0 1 * *` — midnight on the 1st of every month

# Recurring vs one-shot

  - `recurring: true` (default) — fires on every match, reschedules from the
    fire time. Auto-expires after `recurring_max_age_ms` (default 7 days) so
    forgotten jobs do not leak forever.
  - `recurring: false` — fires once and is deleted.

# Jitter

The scheduler applies deterministic per-task jitter to avoid thundering herd
on round wall-clock marks (e.g. `:00`, `:30`):
  - Recurring tasks: forward jitter, up to `recurring_cap_ms` (default 15 min)
    proportional to the interval between fires.
  - One-shot tasks: backward lead (early fire) up to `one_shot_max_ms`
    (default 90 s) when the fire minute matches `one_shot_minute_mod` (default
    30). This keeps `:00`/` `:30` user-pinned reminders from slamming inference.

Avoid scheduling many tasks on the same round mark; stagger via the cron
expression when possible.

# Durable vs session

  - `durable: true` (default) — persisted to `.clawcodex/cron/scheduled_tasks.json`
    and survives process restarts.
  - `durable: false` — kept in the active session only; never written to disk.
    Use for ephemeral follow-ups.

# Scope and limits

  - Maximum 50 scheduled jobs per workspace.
  - `permanent` is a system-only flag (assistant mode installer). CronCreate
    cannot set it; doing so will raise an error.
  - Setting `CLAWCODEX_DISABLE_CRON=1` disables all cron tools — the call
    returns a soft "Cron is disabled" result, not an error.
"""

CRON_LIST_PROMPT = """\
List all scheduled cron jobs (file-backed and session-only) for the current
workspace. Returns per-job `id`, `cron` expression, human-readable schedule,
`recurring`/`durable` flags, plus `createdAt`/`updatedAt`/`lastFiredAt`/
`nextFireAt`/`expiresAt` timestamps.

Use the returned `id` with CronDelete to remove a job. Field reference:
  - `permanent: true` jobs are system-installed (catch-up / morning-checkin /
    dream) and are exempt from auto-expiry — do not delete them.
  - Teammate / agent-scoped jobs (if any) only fire on the owning session.
"""

CRON_DELETE_PROMPT = """\
Delete a scheduled cron job by id. Use CronList first to look up the id; the
field is the 8-char hex returned by CronCreate / CronList.

Deletion is irreversible — the scheduled job definition is removed entirely
(on-disk for durable jobs, in-memory for session-only jobs). Run history
records are retained for audit. Permanent jobs (system-installed) cannot be
deleted via CronDelete; remove them manually from the task file if needed.
"""

CRON_RUN_PROMPT = """\
Manually fire a scheduled cron job by id. Use CronList first to look up the id.
The call creates a queued scheduled-task run and returns its run id. The caller
is responsible for delivering the queued prompt to the active execution loop.
"""


def _cron_disabled_result(tool_name: str) -> ToolResult:
    return ToolResult(
        name=tool_name,
        output={"success": False, "disabled": True, "message": CRON_DISABLED_MESSAGE},
    )


MAX_CRON_TASKS_PER_WORKSPACE = 50


def _cron_create_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    if is_cron_disabled():
        return _cron_disabled_result("CronCreate")

    cron = tool_input.get("cron")
    prompt = tool_input.get("prompt")
    if not isinstance(cron, str) or not cron.strip():
        raise ToolInputError("cron must be a non-empty string")
    if parse_cron_expression(cron) is None:
        raise ToolInputError("cron must be a valid five-field expression")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ToolInputError("prompt must be a non-empty string")

    # CronCreate cannot set `permanent`. The flag is reserved for
    # the assistant-mode installer (write_if_missing).
    if tool_input.get("permanent") is True:
        raise ToolInputError("permanent is a system-only flag and cannot be set via CronCreate")

    recurring = bool(tool_input.get("recurring", True))
    durable = bool(tool_input.get("durable", True))
    # Derive agent identity from the trusted context, not from LLM-controlled
    # tool_input. This prevents spoofing another agent's identity.
    agent_id = getattr(context, "agent_id", None)

    # Enforce the workspace task limit declared in CRON_CREATE_PROMPT.
    existing_tasks = read_all_cron_tasks(context.workspace_root, context.crons)
    if len(existing_tasks) >= MAX_CRON_TASKS_PER_WORKSPACE:
        raise ToolInputError(f"maximum {MAX_CRON_TASKS_PER_WORKSPACE} scheduled jobs per workspace")
    task = add_cron_task(
        context.workspace_root,
        cron=cron.strip(),
        prompt=prompt,
        recurring=recurring,
        durable=durable,
        session_store=context.crons,
        agent_id=agent_id,
    )
    return ToolResult(
        name="CronCreate",
        output={
            "id": task.id,
            "cron": task.cron,
            "humanSchedule": cron_to_human(task.cron),
            "recurring": task.recurring,
            "durable": task.durable,
            "permanent": task.permanent,
            "agentId": task.agent_id,
            "nextFireAt": task.next_fire_at,
        },
    )


CronCreateTool: Tool = build_tool(
    name="CronCreate",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "cron": {"type": "string"},
            "prompt": {"type": "string"},
            "recurring": {"type": "boolean"},
            "durable": {"type": "boolean"},
        },
        "required": ["cron", "prompt"],
    },
    call=_cron_create_call,
    prompt=CRON_CREATE_PROMPT,
    description=CRON_CREATE_PROMPT.splitlines()[0].lstrip("# ").strip() or "Schedule a recurring or one-shot prompt.",
    strict=True,
    max_result_size_chars=100_000,
    is_read_only=lambda _input: False,
    is_concurrency_safe=lambda _input: True,
    to_auto_classifier_input=lambda input_data: (
        f"{(input_data or {}).get('cron', '')}: {(input_data or {}).get('prompt', '')}"
    ),
)


def _cron_list_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    if is_cron_disabled():
        return _cron_disabled_result("CronList")
    jobs = [_task_output(task) for task in read_all_cron_tasks(context.workspace_root, context.crons)]
    # Derive caller identity from the trusted context. The LLM may pass
    # "*" in tool_input to request an admin (all-tasks) view — this is a
    # query parameter, not an identity claim.
    caller_agent_id = getattr(context, "agent_id", None)
    admin_view = tool_input.get("agent_id") == "*"
    if not admin_view and caller_agent_id is not None:
        jobs = [j for j in jobs if j.get("agentId") is None or j.get("agentId") == caller_agent_id]
    return ToolResult(name="CronList", output={"jobs": jobs})


CronListTool: Tool = build_tool(
    name="CronList",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            # Pass "*" to request an admin (all-agents) view. The caller's
            # actual identity is derived from the tool context, not this field.
            "agent_id": {"type": "string"},
        },
    },
    call=_cron_list_call,
    prompt=CRON_LIST_PROMPT,
    description="List scheduled cron jobs.",
    strict=True,
    max_result_size_chars=100_000,
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: True,
)


def _cron_delete_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    if is_cron_disabled():
        return _cron_disabled_result("CronDelete")
    cron_id = tool_input.get("id")
    if not isinstance(cron_id, str) or not cron_id.strip():
        raise ToolInputError("id must be a non-empty string")
    normalized_id = cron_id.strip()
    # Derive caller identity from the trusted context, not from LLM-controlled
    # tool_input. The admin bypass ("*") is a query parameter, not an identity claim.
    caller_agent_id = getattr(context, "agent_id", None)
    admin_bypass = tool_input.get("agent_id") == "*"
    all_tasks = read_all_cron_tasks(context.workspace_root, context.crons)
    target = next((t for t in all_tasks if t.id == normalized_id), None)
    if target is not None and target.agent_id is not None:
        if admin_bypass or caller_agent_id is None:
            pass  # admin or no runtime context: allow
        elif caller_agent_id != target.agent_id:
            raise ToolInputError(
                f"cron job '{normalized_id}' is owned by agent '{target.agent_id}' "
                f"and cannot be deleted by agent '{caller_agent_id}'"
            )
    if target is not None and target.permanent:
        raise ToolInputError(
            f"cron job '{normalized_id}' is permanent and cannot be deleted via CronDelete; "
            f"remove it manually from the task file if absolutely necessary"
        )
    existed = remove_cron_tasks(context.workspace_root, normalized_id, context.crons)
    if not existed:
        raise ToolInputError(f"No scheduled job with id '{normalized_id}'")
    return ToolResult(name="CronDelete", output={"success": True, "id": normalized_id})


CronDeleteTool: Tool = build_tool(
    name="CronDelete",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            # Pass "*" to request an admin bypass for deleting tasks owned
            # by another agent. The caller's actual identity is derived
            # from the tool context, not this field.
            "agent_id": {"type": "string"},
        },
        "required": ["id"],
    },
    call=_cron_delete_call,
    prompt=CRON_DELETE_PROMPT,
    description="Delete a scheduled cron job by id.",
    strict=True,
    max_result_size_chars=100_000,
    is_read_only=lambda _input: False,
    is_concurrency_safe=lambda _input: True,
    to_auto_classifier_input=lambda input_data: (input_data or {}).get("id", "") or "",
)


def _cron_run_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    if is_cron_disabled():
        return _cron_disabled_result("CronRun")
    cron_id = tool_input.get("id")
    if not isinstance(cron_id, str) or not cron_id.strip():
        raise ToolInputError("id must be a non-empty string")
    normalized_id = cron_id.strip()
    if get_cron_task_detail(context.workspace_root, normalized_id, context.crons) is None:
        return ToolResult(
            name="CronRun",
            output={"success": False, "id": normalized_id, "not_found": True},
        )
    run = manual_fire_cron_task(
        context.workspace_root,
        normalized_id,
        context.crons,
        current_dir=getattr(context, "current_dir", None),
    )
    if run is None:
        return ToolResult(name="CronRun", output={"success": False, "id": normalized_id, "run": None})
    return ToolResult(
        name="CronRun",
        output={
            "success": True,
            "id": normalized_id,
            "run": {
                "id": run.id,
                "task_id": run.task_id,
                "prompt": run.prompt,
                "status": run.status,
            },
        },
    )


CronRunTool: Tool = build_tool(
    name="CronRun",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    },
    call=_cron_run_call,
    prompt=CRON_RUN_PROMPT,
    description="Manually fire a scheduled cron job by id.",
    strict=True,
    max_result_size_chars=100_000,
    is_read_only=lambda _input: False,
    is_concurrency_safe=lambda _input: True,
    to_auto_classifier_input=lambda input_data: (input_data or {}).get("id", "") or "",
)


def _task_output(task: CronTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "cron": task.cron,
        "prompt": task.prompt,
        "humanSchedule": cron_to_human(task.cron),
        "recurring": task.recurring,
        "durable": task.durable,
        "permanent": task.permanent,
        "agentId": task.agent_id,
        "createdAt": task.created_at,
        "updatedAt": task.updated_at,
        "lastFiredAt": task.last_fired_at,
        "nextFireAt": task.next_fire_at,
        "expiresAt": task.expires_at,
    }
