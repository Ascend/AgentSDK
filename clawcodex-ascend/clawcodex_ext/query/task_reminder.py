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

"""Periodic task-list reminders for long-running parent conversations.

The reminder mirrors Claude Code's ``todo_reminders`` attachment cadence:
after ten assistant turns without a task-list write, and at least ten
assistant turns after the previous reminder, inject a gentle meta message.
It is advisory and refreshes the LKB read projection only when a reminder is
actually due; it never mutates authoritative task or LKB state.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from clawcodex_ext.tool_system.build_tool import Tool
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.types.messages import Message, UserMessage, create_user_message

# AgentSDK migration Parts do not contain every host facade during incremental validation.
from clawcodex_ext.utils.task_flags import is_todo_v2_enabled  # pylint: disable=no-name-in-module

TURNS_SINCE_WRITE = 10
TURNS_BETWEEN_REMINDERS = 10
TASK_REMINDER_MARKER = "<!-- clawcodex:task-reminder -->"

_INTERNAL_QUERY_SOURCES = frozenset(
    {
        "away_summary",
        "compact",
        "forked_agent",
        "session_memory",
        "side_question",
    }
)
_THINKING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})

logger = logging.getLogger(__name__)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _content_blocks(message: Any) -> list[Any]:
    content = _field(message, "content", [])
    return list(content) if isinstance(content, list) else []


def _block_type(block: Any) -> str:
    return str(_field(block, "type", ""))


def _is_assistant_turn(message: Any) -> bool:
    if _field(message, "role") != "assistant" and _field(message, "type") != "assistant":
        return False
    if bool(_field(message, "isApiErrorMessage", False)):
        return False
    blocks = _content_blocks(message)
    return not blocks or not all(_block_type(block) in _THINKING_BLOCK_TYPES for block in blocks)


def _uses_any_tool(message: Any, tool_names: frozenset[str]) -> bool:
    if not _is_assistant_turn(message):
        return False
    return any(
        _block_type(block) == "tool_use" and str(_field(block, "name", "")) in tool_names
        for block in _content_blocks(message)
    )


def _message_text(message: Any) -> str:
    content = _field(message, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(str(_field(block, "text", "")) for block in content if _block_type(block) == "text")


def _is_task_reminder(message: Any) -> bool:
    return TASK_REMINDER_MARKER in _message_text(message)


def _visible_tool_names(tools: Sequence[Tool]) -> frozenset[str]:
    return frozenset(str(_field(tool, "name", "")) for tool in tools)


def _is_parent_query(query_source: str) -> bool:
    normalized = (query_source or "").strip().lower()
    return not normalized.startswith("agent:") and normalized not in _INTERNAL_QUERY_SOURCES


def get_task_reminder_turn_counts(
    messages: Sequence[Message],
    *,
    write_tool_names: frozenset[str],
) -> tuple[int, int]:
    """Return assistant turns since the latest task write and reminder."""

    turns_since_write = 0
    turns_since_reminder = 0
    found_write = False
    found_reminder = False

    for message in reversed(messages):
        if _is_assistant_turn(message):
            if not found_write and _uses_any_tool(message, write_tool_names):
                found_write = True
            if not found_write:
                turns_since_write += 1
            if not found_reminder:
                turns_since_reminder += 1
        elif not found_reminder and _is_task_reminder(message):
            found_reminder = True

        if found_write and found_reminder:
            break

    return turns_since_write, turns_since_reminder


def _task_status(task: Mapping[str, Any]) -> str:
    lkb = task.get("lkb")
    if isinstance(lkb, Mapping) and lkb.get("derivedStatus"):
        return str(lkb["derivedStatus"])
    return str(task.get("status") or "pending")


def _render_task_items(context: ToolContext) -> list[str]:
    lines: list[str] = []
    for task_id, raw_task in context.tasks.items():
        if not isinstance(raw_task, Mapping):
            continue
        normalized_id = str(raw_task.get("id") or task_id)
        subject = str(raw_task.get("subject") or "")
        lines.append(f"#{normalized_id}. [{_task_status(raw_task)}] {subject}")
    return lines


def _render_todo_items(context: ToolContext) -> list[str]:
    lines: list[str] = []
    for index, raw_todo in enumerate(context.todos, start=1):
        if not isinstance(raw_todo, Mapping):
            continue
        status = str(raw_todo.get("status") or "pending")
        content = str(raw_todo.get("content") or "")
        lines.append(f"{index}. [{status}] {content}")
    return lines


def _wrap_in_system_reminder(content: str) -> str:
    return f"<system-reminder>\n{content}\n</system-reminder>"


def build_task_reminder(
    messages: Sequence[Message],
    context: ToolContext,
    tools: Sequence[Tool],
    *,
    query_source: str = "repl_main_thread",
) -> UserMessage | None:
    """Build the due reminder, or ``None`` when cadence/scope gates fail."""

    if not messages or not _is_parent_query(query_source):
        return None

    visible_names = _visible_tool_names(tools)
    task_v2 = is_todo_v2_enabled()
    if task_v2:
        if "TaskUpdate" not in visible_names:
            return None
        write_tool_names = frozenset({"TaskCreate", "TaskUpdate"})
        message = (
            "The task tools haven't been used recently. If the current work benefits from "
            "progress tracking, consider using TaskCreate for newly discovered work and "
            "TaskUpdate to mark tasks in_progress when starting and completed when done. "
            "Review pending, running, blocked, and needs_recheck tasks, and clean up stale "
            "entries only when the work actually changed. This is a gentle reminder; ignore "
            "it when it is not relevant. Never mention this reminder to the user."
        )
        list_heading = "Here are the existing tasks:"
    else:
        if "TodoWrite" not in visible_names:
            return None
        write_tool_names = frozenset({"TodoWrite"})
        message = (
            "The TodoWrite tool hasn't been used recently. If the current work benefits "
            "from progress tracking, consider using TodoWrite to update task status and "
            "clean up stale entries only when the work actually changed. This is a gentle "
            "reminder; ignore it when it is not relevant. Never mention this reminder to "
            "the user."
        )
        list_heading = "Here are the existing contents of your todo list:"

    turns_since_write, turns_since_reminder = get_task_reminder_turn_counts(
        messages,
        write_tool_names=write_tool_names,
    )
    if turns_since_write < TURNS_SINCE_WRITE or turns_since_reminder < TURNS_BETWEEN_REMINDERS:
        return None

    if task_v2:
        try:
            from lkb.clawcodex_task_adapter import refresh_task_projection

            refresh_task_projection(context)
        except Exception as exc:
            # Reminders are advisory; projection refresh failure must never
            # break or delay the parent query.
            logger.warning(
                "Task reminder skipped LKB projection refresh (error_type=%s)",
                type(exc).__name__,
            )
        item_lines = _render_task_items(context)
    else:
        item_lines = _render_todo_items(context)

    parts = [TASK_REMINDER_MARKER, message]
    if item_lines:
        parts.extend(["", list_heading, "", "\n".join(item_lines)])
    return create_user_message(
        content=_wrap_in_system_reminder("\n".join(parts)),
        isMeta=True,
        origin="system_injection",
    )


__all__ = [
    "TASK_REMINDER_MARKER",
    "TURNS_BETWEEN_REMINDERS",
    "TURNS_SINCE_WRITE",
    "build_task_reminder",
    "get_task_reminder_turn_counts",
]
