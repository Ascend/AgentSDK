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

"""Claude Code-compatible ``/goal`` user command."""

from __future__ import annotations

import logging
from typing import Any

# pylint: disable-next=no-name-in-module
from clawcodex_ext.command_system.types import InteractiveCommand, InteractiveOutcome

# pylint: disable-next=no-name-in-module
from clawcodex_ext.types.messages import create_system_message

from .files import MAX_THREAD_GOAL_OBJECTIVE_CHARS, objective_text_for_edit
from .gate import goal_enabled
from .model import GoalCompletionMode, ThreadGoalStatus
from .protocol import (
    ThreadGoalClearParams,
    ThreadGoalDTO,
    ThreadGoalGetParams,
    ThreadGoalProtocol,
    ThreadGoalReplaceParams,
)
from .service import GoalServiceError, goal_thread_id_from_context

logger = logging.getLogger(__name__)

GOAL_USAGE = "/goal [<condition>|clear]"
GOAL_CLEAR_ALIASES = frozenset({"clear", "stop", "off", "reset", "none", "cancel"})


class GoalCommand(InteractiveCommand):
    """Set, inspect, or clear a session-scoped completion condition."""

    async def run(self, args: str, context: Any) -> InteractiveOutcome:
        if not goal_enabled():
            return InteractiveOutcome(message="Goal mode is disabled.", display="system")

        api = _goal_api_from_context(context)
        thread_id = goal_thread_id_from_context(context)
        command = args.strip()

        if not command:
            return _goal_summary(api, thread_id, context)
        if command.lower() in GOAL_CLEAR_ALIASES:
            return _clear_goal(api, thread_id, context)
        if len(command) > MAX_THREAD_GOAL_OBJECTIVE_CHARS:
            raise GoalServiceError(f"Goal condition must be {MAX_THREAD_GOAL_OBJECTIVE_CHARS:,} characters or fewer.")
        return _set_goal_condition(api, thread_id, command, context)


GOAL_COMMAND = GoalCommand(
    name="goal",
    description="Set a goal Claude checks before stopping.",
    argument_hint="[<condition>|clear]",
    aliases=[],
    is_enabled=goal_enabled,
)


def _goal_api_from_context(context: Any) -> ThreadGoalProtocol:
    api = getattr(context, "goal_api", None)
    tool_context = getattr(context, "tool_context", None)
    if api is None:
        service = getattr(context, "goal_service", None) or getattr(
            tool_context,
            "goal_service",
            None,
        )
        events = getattr(context, "goal_events", None)
        kwargs: dict[str, Any] = {}
        if service is not None:
            kwargs["service"] = service
        if events is not None:
            kwargs["events"] = events
        api = ThreadGoalProtocol(**kwargs)

    try:
        context.goal_api = api
        context.goal_service = api.service
        context.goal_events = api.events
    except Exception:
        logger.debug("Goal API context attachment failed", exc_info=True)
    if tool_context is not None:
        try:
            tool_context.goal_service = api.service
        except Exception:
            logger.debug("Goal service tool-context attachment failed", exc_info=True)
    return api


def _goal_summary(
    api: ThreadGoalProtocol,
    thread_id: str,
    context: Any | None = None,
) -> InteractiveOutcome:
    # Persist elapsed idle time before rendering so the explicit status view
    # and the continuously-updated footer report the same live duration.
    tool_context = getattr(context, "tool_context", None)
    if tool_context is not None:
        from .accounting import BudgetLimitedGoalDisposition
        from .runtime import goal_runtime_for_context

        runtime = goal_runtime_for_context(tool_context)
        if runtime is not None and runtime.thread_id == thread_id:
            runtime.account_idle_goal_progress(BudgetLimitedGoalDisposition.KEEP_ACTIVE)

    response = api.thread_goal_get(ThreadGoalGetParams(thread_id=thread_id))
    if response.goal is None:
        _sync_app_goal_status(None, context)
        return InteractiveOutcome(
            message="Goal\n\nNo goal set\n  /goal <condition> to set one",
            display="system",
            transient=True,
        )
    _sync_app_goal_status(response.goal, context)
    return InteractiveOutcome(
        message=_format_goal_summary(response.goal, api=api),
        display="system",
        transient=True,
    )


def _set_goal_condition(
    api: ThreadGoalProtocol,
    thread_id: str,
    condition: str,
    context: Any,
) -> InteractiveOutcome:
    if len(condition) > MAX_THREAD_GOAL_OBJECTIVE_CHARS:
        return InteractiveOutcome(
            message=(f"Goal condition must be {MAX_THREAD_GOAL_OBJECTIVE_CHARS:,} characters or fewer."),
            display="system",
        )

    response = api.thread_goal_replace(
        ThreadGoalReplaceParams(
            thread_id=thread_id,
            objective=condition,
            completion_mode=GoalCompletionMode.EVALUATOR,
        )
    )
    _sync_app_goal_status(response.goal, context)
    persisted = api.service.get_goal(thread_id)
    _append_lifecycle_notice(
        context,
        subtype="goal_set",
        message=f"Goal set: {condition}",
        data={
            "goalId": persisted.goal_id if persisted is not None else None,
            "condition": condition,
            "state": "active",
            "met": None,
            "reason": None,
            "turns": response.goal.evaluation_count,
            "tokens": response.goal.tokens_used,
            "durationSeconds": response.goal.time_used_seconds,
        },
    )
    return InteractiveOutcome(
        message=f"Goal set: {condition}",
        display="system",
        should_query=True,
    )


def _clear_goal(
    api: ThreadGoalProtocol,
    thread_id: str,
    context: Any | None = None,
) -> InteractiveOutcome:
    current = api.thread_goal_get(ThreadGoalGetParams(thread_id=thread_id)).goal
    if current is None or current.status is ThreadGoalStatus.COMPLETE:
        if current is None:
            _sync_app_goal_status(None, context)
        else:
            _sync_app_goal_status(current, context)
        return InteractiveOutcome(message="No goal set", display="system")

    persisted = api.service.get_goal(thread_id)
    response = api.thread_goal_clear(ThreadGoalClearParams(thread_id=thread_id))
    if not response.cleared:
        return InteractiveOutcome(message="No goal set", display="system")

    _sync_app_goal_status(None, context)
    condition = objective_text_for_edit(
        current.objective,
        codex_home=getattr(api.service, "codex_home", None),
    )
    _append_lifecycle_notice(
        context,
        subtype="goal_cleared",
        message=f"Goal cleared: {condition}",
        data={
            "goalId": persisted.goal_id if persisted is not None else None,
            "condition": condition,
            "state": "cleared",
            "met": False,
            "reason": "cleared by user",
            "turns": current.evaluation_count,
            "tokens": current.tokens_used,
            "durationSeconds": current.time_used_seconds,
        },
    )
    return InteractiveOutcome(
        message=f"Goal cleared: {condition}",
        display="system",
    )


def _append_lifecycle_notice(
    context: Any | None,
    *,
    subtype: str,
    message: str,
    data: dict[str, Any],
) -> None:
    """Persist command-side goal transitions as replayable transcript facts."""

    conversation = getattr(context, "conversation", None)
    if conversation is None:
        return
    notice = create_system_message(
        message,
        subtype=subtype,
        data=data,
    )
    add_existing = getattr(conversation, "add_existing_message", None)
    if callable(add_existing):
        add_existing(notice)
        return
    add_message = getattr(conversation, "add_message", None)
    if callable(add_message):
        add_message(notice.role, notice.content)


def _format_goal_summary(goal: ThreadGoalDTO, *, api: ThreadGoalProtocol) -> str:
    condition = objective_text_for_edit(
        goal.objective,
        codex_home=getattr(api.service, "codex_home", None),
    )
    if goal.status is ThreadGoalStatus.COMPLETE:
        title = "✓ Goal achieved"
        hint = ""
    elif goal.status is ThreadGoalStatus.ACTIVE:
        title = "◎ Goal active"
        hint = "\n\n  /goal clear to stop early"
    else:
        # Legacy ClawCodex rows can still carry internal failure states. Keep
        # them inspectable without advertising those states as new commands.
        title = "Goal inactive"
        hint = "\n\n  Set a new /goal <condition> to replace it"
    turns = ""
    last_check = ""
    if goal.evaluation_count > 0:
        turn_label = "turn" if goal.evaluation_count == 1 else "turns"
        turns = f" · {goal.evaluation_count} {turn_label}"
        reason = goal.last_evaluation_reason if goal.last_evaluation_reason is not None else "not available"
        last_check = f"\n\n  Last check: {reason}"
    return (
        f"{title}\n\n"
        f"  running {_format_elapsed(goal.time_used_seconds)}"
        f"{turns} · {goal.tokens_used} tokens\n\n"
        f"  Goal: {condition}"
        f"{last_check}"
        f"{hint}"
    )


def _format_elapsed(seconds: int) -> str:
    seconds = max(int(seconds), 0)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h" if remaining_minutes == 0 else f"{hours}h {remaining_minutes}m"


def _sync_app_goal_status(goal: ThreadGoalDTO | None, context: Any | None) -> None:
    if context is None:
        return
    app_state = getattr(context, "app_state", None)
    if app_state is None:
        return
    setter = getattr(app_state, "set_goal_status", None)
    if callable(setter):
        setter(goal.to_dict() if goal is not None else None)


__all__ = [
    "GOAL_CLEAR_ALIASES",
    "GOAL_COMMAND",
    "GOAL_USAGE",
    "GoalCommand",
]
