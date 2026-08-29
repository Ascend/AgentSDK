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

"""Model-callable tools for upstream-compatible thread goals."""

from __future__ import annotations

import logging
from typing import Any, Iterable

from clawcodex_ext.tool_system.build_tool import Tool, build_tool
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.protocol import ToolResult

from clawcodex_ext.goal.gate import goal_enabled
from clawcodex_ext.goal.model import GoalCompletionMode, ThreadGoal, ThreadGoalStatus
from clawcodex_ext.goal.protocol import ThreadGoalDTO
from clawcodex_ext.goal.service import GoalService, GoalServiceError

logger = logging.getLogger(__name__)

GET_GOAL_TOOL_NAME = "get_goal"
CREATE_GOAL_TOOL_NAME = "create_goal"
UPDATE_GOAL_TOOL_NAME = "update_goal"

GOAL_MODEL_TOOL_NAMES = frozenset({GET_GOAL_TOOL_NAME, CREATE_GOAL_TOOL_NAME, UPDATE_GOAL_TOOL_NAME})

GET_GOAL_DESCRIPTION = "Read this thread's persisted objective, state, budget, and measured usage."

CREATE_GOAL_DESCRIPTION = f"""Start a persisted goal only after an explicit user or system request.
Ordinary tasks must not create goals implicitly. Supply a budget only when the request names one. An unfinished goal cannot be replaced; use {UPDATE_GOAL_TOOL_NAME} for its terminal state."""

UPDATE_GOAL_DESCRIPTION = """Set the terminal state of the current persisted goal.
Use complete only after current evidence verifies every required outcome. Use blocked only when the same obstacle has prevented meaningful progress for three consecutive goal turns; a resumed goal begins a new count. Difficulty, uncertainty, an expiring budget, or the end of a turn are not terminal conditions. Pause, resume, and limit states remain under user or runtime control. This tool is unavailable while an evaluator-managed goal is active because the evaluator owns completion. Report the returned usage when completing a budgeted goal."""

_OBJECTIVE_DESCRIPTION = "The explicit, concrete outcome that the persisted goal must achieve."

# Bandit B105 is a false positive: this is a public schema description.
_TOKEN_BUDGET_DESCRIPTION = "Optional positive token ceiling, allowed only when the request supplies it."  # nosec B105

_UPDATE_STATUS_DESCRIPTION = (
    "Terminal state: complete after verified success, or blocked after the same "
    "impasse spans three consecutive goal turns."
)

_NO_ARGUMENTS_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}


def make_goal_model_tools() -> list[Tool]:
    """Build the three model-visible goal tools."""
    return [_build_goal_reader(), _build_goal_creator(), _build_goal_updater()]


def _build_goal_reader() -> Tool:
    return build_tool(
        name=GET_GOAL_TOOL_NAME,
        input_schema=_NO_ARGUMENTS_SCHEMA,
        call=_get_goal,
        prompt=GET_GOAL_DESCRIPTION,
        description=GET_GOAL_DESCRIPTION,
        is_enabled=goal_enabled,
        is_concurrency_safe=lambda _input: True,
        is_read_only=lambda _input: True,
        user_facing_name=lambda _input: GET_GOAL_TOOL_NAME,
    )


def _build_goal_creator() -> Tool:
    schema = {
        "type": "object",
        "properties": {
            "objective": {"type": "string", "description": _OBJECTIVE_DESCRIPTION},
            "token_budget": {"type": "integer", "description": _TOKEN_BUDGET_DESCRIPTION},
        },
        "required": ["objective"],
        "additionalProperties": False,
    }
    return build_tool(
        name=CREATE_GOAL_TOOL_NAME,
        input_schema=schema,
        call=_create_goal,
        prompt=CREATE_GOAL_DESCRIPTION,
        description=CREATE_GOAL_DESCRIPTION,
        is_enabled=goal_enabled,
        user_facing_name=lambda _input: CREATE_GOAL_TOOL_NAME,
    )


def _build_goal_updater() -> Tool:
    schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["complete", "blocked"],
                "description": _UPDATE_STATUS_DESCRIPTION,
            }
        },
        "required": ["status"],
        "additionalProperties": False,
    }
    return build_tool(
        name=UPDATE_GOAL_TOOL_NAME,
        input_schema=schema,
        call=_update_goal,
        prompt=UPDATE_GOAL_DESCRIPTION,
        description=UPDATE_GOAL_DESCRIPTION,
        is_enabled=goal_enabled,
        user_facing_name=lambda _input: UPDATE_GOAL_TOOL_NAME,
    )


def filter_goal_model_tools_for_context(
    tools: Iterable[Tool],
    context: ToolContext | Any | None,
) -> list[Tool]:
    """Drop goal tools from model-visible lists when upstream would hide them."""
    visible = goal_model_tools_visible(context)
    if visible:
        visible_tools = list(tools)
        if _has_active_evaluator_goal(context):
            return [tool for tool in visible_tools if tool.name != UPDATE_GOAL_TOOL_NAME]
        return visible_tools
    return [tool for tool in tools if tool.name not in GOAL_MODEL_TOOL_NAMES]


def goal_model_tools_visible(context: ToolContext | Any | None) -> bool:
    """Return whether goal tools should be exposed for this model context."""
    if not goal_enabled():
        return False
    if context is None:
        return False
    if _is_review_subagent_context(context):
        return False
    return _thread_id_from_context(context) is not None


def _has_active_evaluator_goal(context: ToolContext | Any | None) -> bool:
    if context is None:
        return False
    thread_id = _thread_id_from_context(context)
    if thread_id is None:
        return False
    service = getattr(context, "goal_service", None)
    if service is None:
        return False
    try:
        goal = service.get_goal(thread_id)
    except Exception:
        logger.exception("Goal lookup failed while filtering model tools")
        return True
    return bool(
        goal is not None
        and goal.status is ThreadGoalStatus.ACTIVE
        and goal.completion_mode is GoalCompletionMode.EVALUATOR
    )


def _get_goal(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    del tool_input
    try:
        thread_id, service = _goal_context(context)
        goal = service.get_goal(thread_id)
        return ToolResult(
            name=GET_GOAL_TOOL_NAME,
            output=_goal_response(goal, include_completion_budget_report=False),
        )
    except Exception as exc:  # noqa: BLE001 - tool errors are model-facing data
        return _tool_error(GET_GOAL_TOOL_NAME, exc)


def _create_goal(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    try:
        thread_id, service = _goal_context(context)
        objective = _objective_from_input(tool_input)
        token_budget = _parse_create_token_budget(tool_input)
        goal = service.create_goal(
            thread_id,
            objective,
            token_budget,
        )
        return ToolResult(
            name=CREATE_GOAL_TOOL_NAME,
            output=_goal_response(goal, include_completion_budget_report=False),
        )
    except Exception as exc:  # noqa: BLE001 - tool errors are model-facing data
        return _tool_error(CREATE_GOAL_TOOL_NAME, exc)


def _update_goal(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    try:
        thread_id, service = _goal_context(context)
        status = _parse_update_status(tool_input.get("status"))
        current = service.get_goal(thread_id)
        if (
            current is not None
            and current.status is ThreadGoalStatus.ACTIVE
            and current.completion_mode is GoalCompletionMode.EVALUATOR
        ):
            raise GoalServiceError("this /goal is completed only by its independent evaluator")
        goal = service.update_goal(thread_id, status)
        if goal is None:
            raise GoalServiceError("cannot update goal because this thread has no goal")
        return ToolResult(
            name=UPDATE_GOAL_TOOL_NAME,
            output=_goal_response(
                goal,
                include_completion_budget_report=status is ThreadGoalStatus.COMPLETE,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - tool errors are model-facing data
        return _tool_error(UPDATE_GOAL_TOOL_NAME, exc)


def _goal_response(
    goal: ThreadGoal | None,
    *,
    include_completion_budget_report: bool,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "goal": None,
        "remainingTokens": None,
        "completionBudgetReport": None,
    }
    if goal is None:
        return response

    response["goal"] = ThreadGoalDTO.from_model(goal).to_dict()
    response["remainingTokens"] = _remaining_tokens(goal)
    if include_completion_budget_report and goal.status is ThreadGoalStatus.COMPLETE:
        response["completionBudgetReport"] = _completion_budget_report(goal)
    return response


def _remaining_tokens(goal: ThreadGoal) -> int | None:
    if goal.token_budget is None:
        return None
    return max(0, goal.token_budget - goal.tokens_used)


def _completion_budget_report(goal: ThreadGoal) -> str | None:
    if goal.token_budget is None and goal.time_used_seconds <= 0:
        return None
    return (
        "The goal is complete. Use the structured goal payload to report its "
        "tokensUsed and tokenBudget values when present, plus a concise elapsed "
        "time summary when timeUsedSeconds is positive."
    )


def _goal_context(context: ToolContext | Any) -> tuple[str, GoalService]:
    thread_id = _thread_id_from_context(context)
    if thread_id is None:
        raise GoalServiceError("goal tools require a saved session")
    service = getattr(context, "goal_service", None)
    if service is None:
        raise GoalServiceError("goal service is unavailable for this session")
    return thread_id, service


def _thread_id_from_context(context: ToolContext | Any) -> str | None:
    return _context_text(context, "goal_thread_id") or _context_text(context, "session_id")


def _is_review_subagent_context(context: ToolContext | Any) -> bool:
    agent_type = (_context_text(context, "agent_type") or "").lower()
    options = getattr(context, "options", None)
    query_source = (_context_text(options, "query_source") or "").lower()
    review_sources = {
        "agent:builtin:review",
        "agent:template:review",
        "agent:builtin:code-reviewer",
    }
    return agent_type in {"review", "code-reviewer"} or query_source in review_sources


def _context_text(context: Any, attribute: str) -> str | None:
    value = getattr(context, attribute, None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _objective_from_input(tool_input: dict[str, Any]) -> str:
    value = tool_input.get("objective")
    if not isinstance(value, str):
        raise GoalServiceError("goal objective is required")
    objective = value.strip()
    if not objective:
        raise GoalServiceError("goal objective cannot be empty")
    return objective


def _parse_create_token_budget(tool_input: dict[str, Any]) -> int | None:
    raw_budget = tool_input.get("token_budget")
    if raw_budget is None:
        return None
    token_budget = int(raw_budget)
    if token_budget <= 0:
        raise GoalServiceError("goal budgets must be positive when provided")
    return token_budget


def _parse_update_status(value: Any) -> ThreadGoalStatus:
    try:
        status = ThreadGoalStatus.from_wire(str(value))
    except ValueError as exc:
        raise GoalServiceError(f"invalid goal status: {value!r}") from exc

    if status not in {ThreadGoalStatus.COMPLETE, ThreadGoalStatus.BLOCKED}:
        raise GoalServiceError(
            "update_goal accepts only complete or blocked; user and runtime controls own pause, resume, and limit states"
        )
    return status


def _tool_error(name: str, exc: BaseException) -> ToolResult:
    if isinstance(exc, (GoalServiceError, ValueError)):
        message = str(exc)
    else:
        logger.error(
            "Goal tool %s failed",
            name,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        message = "goal operation failed"
    return ToolResult(name=name, output={"error": message}, is_error=True)


__all__ = [
    "CREATE_GOAL_TOOL_NAME",
    "GET_GOAL_TOOL_NAME",
    "GOAL_MODEL_TOOL_NAMES",
    "UPDATE_GOAL_TOOL_NAME",
    "filter_goal_model_tools_for_context",
    "goal_model_tools_visible",
    "make_goal_model_tools",
]
