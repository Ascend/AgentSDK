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

"""Optional REPL status integration for the ClawCodex host.

The host owns the prompt toolbar lifecycle; this module owns all LKB-specific
projection refresh and progress formatting so ``clawcodex_ext.repl`` only
needs two best-effort call sites.
"""

from __future__ import annotations

import time
from typing import Any

_STATE_ATTR = "_lkb_repl_status_state"


def _visible_tasks(context: Any) -> list[dict[str, Any]]:
    raw_tasks = getattr(context, "tasks", None) or {}
    if not isinstance(raw_tasks, dict):
        return []
    tasks: list[dict[str, Any]] = []
    for task in raw_tasks.values():
        if not isinstance(task, dict):
            continue
        metadata = task.get("metadata")
        if isinstance(metadata, dict) and bool(metadata.get("_internal")):
            continue
        tasks.append(task)
    return tasks


def format_task_progress(context: Any) -> str:
    """Return the compact Task/LKB progress toolbar segment."""

    tasks = _visible_tasks(context)
    if not tasks:
        return ""
    completed = sum(task.get("status") == "completed" for task in tasks)
    running = sum(task.get("status") == "in_progress" for task in tasks)
    blocked = sum(isinstance(task.get("lkb"), dict) and task["lkb"].get("derivedStatus") == "blocked" for task in tasks)
    detail: list[str] = []
    if running:
        detail.append(f"{running} running")
    if blocked:
        detail.append(f"{blocked} blocked")
    suffix = f" ({', '.join(detail)})" if detail else ""
    return f" · tasks: {completed}/{len(tasks)}{suffix}"


def refresh_task_projection(context: Any, *, force: bool = False) -> bool:
    """Refresh the authoritative LKB projection, throttled to once per second."""

    now = time.monotonic()
    state = getattr(context, _STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {"last_refresh": 0.0, "signature": None}
        setattr(context, _STATE_ATTR, state)
    last_refresh = float(state.get("last_refresh", 0.0) or 0.0)
    if not force and now - last_refresh < 1.0:
        return False
    state["last_refresh"] = now
    if not getattr(context, "lkb_plan_id", None) and not getattr(context, "tasks", None):
        return False

    from lkb.clawcodex_task_adapter import try_handle
    from lkb.flags import is_plan_graph_enabled

    if not is_plan_graph_enabled():
        return False
    handled, result = try_handle("TaskList", {}, context)
    if not handled or result is None or result.is_error:
        return False
    signature = tuple(
        sorted(
            (
                str(task.get("id", "")),
                str(task.get("status", "pending")),
                str(task.get("owner") or ""),
                str((task.get("lkb") or {}).get("derivedStatus") or ""),
                tuple(str(x) for x in (task.get("blockedBy") or [])),
            )
            for task in _visible_tasks(context)
        )
    )
    previous = state.get("signature")
    state["signature"] = signature
    return previous is not None and signature != previous


__all__ = ["format_task_progress", "refresh_task_projection"]
