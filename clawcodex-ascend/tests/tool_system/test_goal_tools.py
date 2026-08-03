"""Upstream-compatible model goal tool tests for F-122 Spec 4."""

from __future__ import annotations


# pylint: disable=E0611
from pathlib import Path

import pytest

from clawcodex_ext.feature_gate import get_registry, reset_registry
from clawcodex_ext.goal.gate import ensure_goals_feature_registered
from clawcodex_ext.goal.model import GoalCompletionMode, ThreadGoalStatus
from clawcodex_ext.goal.service import GoalService
from clawcodex_ext.goal.store import GoalStore, goals_db_filename
from clawcodex_ext.goal.tools import (
    CREATE_GOAL_TOOL_NAME,
    GET_GOAL_TOOL_NAME,
    GOAL_MODEL_TOOL_NAMES,
    UPDATE_GOAL_TOOL_NAME,
    filter_goal_model_tools_for_context,
    make_goal_model_tools,
)
from clawcodex_ext.tool_system import get_team_aware_tool_list
from clawcodex_ext.tool_system.errors import ToolInputError
from clawcodex_ext.tool_system.protocol import ToolCall
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _fresh_feature_registry():
    reset_registry()
    ensure_goals_feature_registered()
    yield
    reset_registry()


def _goal_context(
    tmp_path: Path,
    *,
    session_id: str | None = "thread-1",
    agent_type: str | None = None,
) -> ToolContext:
    return ToolContext(
        workspace_root=tmp_path,
        session_id=session_id,
        agent_type=agent_type,
        goal_service=GoalService(store=GoalStore(tmp_path / goals_db_filename())),
    )


def _goal_registry() -> ToolRegistry:
    return ToolRegistry(make_goal_model_tools())


def _dispatch(
    registry: ToolRegistry,
    name: str,
    tool_input: dict,
    context: ToolContext,
):
    return registry.dispatch(
        ToolCall(name=name, input=tool_input, tool_use_id=f"{name}-1"),
        context,
    )


def test_default_registry_registers_three_upstream_goal_tools_without_legacy_goal():
    registry = build_default_registry(include_user_tools=False, load_agent_tools=False)

    names = {tool.name for tool in registry.list_tools()}

    assert GOAL_MODEL_TOOL_NAMES <= names
    assert "Goal" not in names


def test_goal_tool_schemas_match_upstream_spec_fields():
    registry = _goal_registry()

    get_goal = registry.get(GET_GOAL_TOOL_NAME)
    create_goal = registry.get(CREATE_GOAL_TOOL_NAME)
    update_goal = registry.get(UPDATE_GOAL_TOOL_NAME)

    assert get_goal is not None
    assert get_goal.input_schema == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    assert get_goal.prompt() == (
        "Get the current goal for this thread, including status, budgets, "
        "token and elapsed-time usage, and remaining token budget."
    )
    assert get_goal.is_read_only({})

    assert create_goal is not None
    assert create_goal.input_schema == {
        "type": "object",
        "properties": {
            "objective": {
                "type": "string",
                "description": (
                    "Required. The concrete objective to start pursuing. "
                    "This starts a new active goal when no goal exists or "
                    "replaces the current goal when it is complete."
                ),
            },
            "token_budget": {
                "type": "integer",
                "description": ("Positive token budget for the new goal. Omit unless explicitly requested."),
            },
        },
        "required": ["objective"],
        "additionalProperties": False,
    }
    assert create_goal.prompt() == (
        "Create a goal only when explicitly requested by the user or "
        "system/developer instructions; do not infer goals from ordinary tasks.\n"
        "Set token_budget only when an explicit token budget is requested. "
        "Fails if an unfinished goal exists; use update_goal only for status."
    )
    assert not create_goal.is_read_only({})

    assert update_goal is not None
    assert update_goal.input_schema == {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["complete", "blocked"],
                "description": (
                    "Required. Set to `complete` only when the objective is "
                    "achieved and no required work remains. Set to `blocked` "
                    "only after the same blocking condition has recurred for "
                    "at least three consecutive goal turns and the agent is at "
                    "an impasse. After a previously blocked goal is resumed, "
                    "the resumed run starts a fresh blocked audit."
                ),
            }
        },
        "required": ["status"],
        "additionalProperties": False,
    }
    assert "pause, resume, budget-limit, or usage-limit" in update_goal.prompt()
    assert "objective" not in update_goal.input_schema["properties"]
    assert not update_goal.is_read_only({})


def test_goal_tools_are_hidden_when_feature_disabled(tmp_path: Path):
    registry = _goal_registry()
    context = _goal_context(tmp_path)
    get_registry().set_override("goals", False)

    tools = get_team_aware_tool_list(registry, team=None, context=context)

    assert [tool.name for tool in tools] == []


def test_goal_tools_are_hidden_without_persistent_session(tmp_path: Path):
    registry = _goal_registry()
    context = _goal_context(tmp_path, session_id=None)

    tools = get_team_aware_tool_list(registry, team=None, context=context)

    assert [tool.name for tool in tools] == []


def test_goal_tools_are_hidden_for_review_subagents(tmp_path: Path):
    registry = _goal_registry()
    context = _goal_context(tmp_path, agent_type="review")

    tools = get_team_aware_tool_list(registry, team=None, context=context)

    assert [tool.name for tool in tools] == []


def test_goal_context_preserves_default_non_team_tools(tmp_path: Path):
    registry = build_default_registry(include_user_tools=False, load_agent_tools=False)
    context = _goal_context(tmp_path)

    tools = get_team_aware_tool_list(registry, team=None, context=context)
    names = {tool.name for tool in tools}

    assert GOAL_MODEL_TOOL_NAMES <= names
    assert {"Agent", "Bash", "Read", "TeamCreate", "WebFetch", "WebSearch"} <= names
    assert "SendMessage" not in names
    assert "TeamDelete" not in names


def test_review_subagent_context_only_hides_goal_tools(tmp_path: Path):
    registry = build_default_registry(include_user_tools=False, load_agent_tools=False)
    context = _goal_context(tmp_path, agent_type="review")

    tools = get_team_aware_tool_list(registry, team=None, context=context)
    names = {tool.name for tool in tools}

    assert GOAL_MODEL_TOOL_NAMES.isdisjoint(names)
    assert {"Agent", "Bash", "Read", "WebFetch", "WebSearch"} <= names
    assert "SendMessage" not in names


def test_coordinator_mode_keeps_worker_and_goal_tools_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    registry = build_default_registry(include_user_tools=False, load_agent_tools=False)
    context = _goal_context(tmp_path)
    monkeypatch.setenv("CLAUDE_CODE_COORDINATOR_MODE", "1")

    tools = get_team_aware_tool_list(registry, team=None, context=context)
    names = {tool.name for tool in tools}

    assert {
        "Agent",
        "SendMessage",
        "TeamCreate",
        "TaskStop",
        "Read",
        "WebFetch",
        "WebSearch",
        GET_GOAL_TOOL_NAME,
        CREATE_GOAL_TOOL_NAME,
        UPDATE_GOAL_TOOL_NAME,
    } <= names
    assert "Bash" not in names
    assert "TeamDelete" not in names


def test_get_goal_returns_structured_empty_response(tmp_path: Path):
    registry = _goal_registry()
    context = _goal_context(tmp_path)

    result = _dispatch(registry, GET_GOAL_TOOL_NAME, {}, context)

    assert not result.is_error
    assert result.output == {
        "goal": None,
        "remainingTokens": None,
        "completionBudgetReport": None,
    }


def test_get_goal_returns_current_goal(tmp_path: Path):
    registry = _goal_registry()
    context = _goal_context(tmp_path)
    _dispatch(registry, CREATE_GOAL_TOOL_NAME, {"objective": "read current"}, context)

    result = _dispatch(registry, GET_GOAL_TOOL_NAME, {}, context)

    assert not result.is_error
    assert result.output["goal"]["objective"] == "read current"
    assert result.output["goal"]["status"] == "active"
    assert result.output["remainingTokens"] is None


def test_create_goal_creates_active_goal_and_reports_remaining_budget(tmp_path: Path):
    registry = _goal_registry()
    context = _goal_context(tmp_path)

    result = _dispatch(
        registry,
        CREATE_GOAL_TOOL_NAME,
        {"objective": "  ship spec 4  ", "token_budget": 100},
        context,
    )

    assert not result.is_error
    assert result.output["goal"] == {
        "threadId": "thread-1",
        "objective": "ship spec 4",
        "status": "active",
        "completionMode": "tool",
        "evaluationCount": 0,
        "lastEvaluationReason": None,
        "tokenBudget": 100,
        "tokensUsed": 0,
        "timeUsedSeconds": 0,
        "createdAt": result.output["goal"]["createdAt"],
        "updatedAt": result.output["goal"]["updatedAt"],
    }
    assert "goalId" not in result.output["goal"]
    assert "completedAt" not in result.output["goal"]
    assert result.output["remainingTokens"] == 100
    assert result.output["completionBudgetReport"] is None


def test_create_goal_rejects_unfinished_goal_without_overwriting(tmp_path: Path):
    registry = _goal_registry()
    context = _goal_context(tmp_path)
    _dispatch(registry, CREATE_GOAL_TOOL_NAME, {"objective": "first"}, context)

    result = _dispatch(registry, CREATE_GOAL_TOOL_NAME, {"objective": "second"}, context)

    assert result.is_error
    assert "unfinished goal" in result.output["error"]
    goal = context.goal_service.get_goal("thread-1")
    assert goal is not None
    assert goal.objective == "first"


@pytest.mark.parametrize(
    ("tool_input", "expected_error"),
    [
        ({"objective": "  "}, "goal objective cannot be empty"),
        ({"objective": "budget", "token_budget": 0}, "goal budgets must be positive"),
    ],
)
def test_create_goal_returns_model_facing_errors_for_invalid_inputs(
    tmp_path: Path,
    tool_input: dict,
    expected_error: str,
):
    registry = _goal_registry()
    context = _goal_context(tmp_path)
    tool = registry.get(CREATE_GOAL_TOOL_NAME)
    assert tool is not None

    result = tool.call(tool_input, context)

    assert result.is_error
    assert expected_error in result.output["error"]
    assert context.goal_service.get_goal("thread-1") is None


def test_create_goal_after_complete_replaces_with_new_active_goal(tmp_path: Path):
    registry = _goal_registry()
    context = _goal_context(tmp_path)
    _dispatch(registry, CREATE_GOAL_TOOL_NAME, {"objective": "first"}, context)
    first = context.goal_service.get_goal("thread-1")
    assert first is not None
    _dispatch(registry, UPDATE_GOAL_TOOL_NAME, {"status": "complete"}, context)

    result = _dispatch(registry, CREATE_GOAL_TOOL_NAME, {"objective": "second"}, context)

    assert not result.is_error
    assert result.output["goal"]["objective"] == "second"
    assert result.output["goal"]["status"] == "active"
    current = context.goal_service.get_goal("thread-1")
    assert current is not None
    assert current.goal_id != first.goal_id


def test_update_goal_can_mark_complete_and_returns_completion_budget_report(
    tmp_path: Path,
):
    registry = _goal_registry()
    context = _goal_context(tmp_path)
    _dispatch(
        registry,
        CREATE_GOAL_TOOL_NAME,
        {"objective": "finish", "token_budget": 100},
        context,
    )
    goal = context.goal_service.get_goal("thread-1")
    assert goal is not None
    context.goal_service.account_usage(
        "thread-1",
        expected_goal_id=goal.goal_id,
        token_delta=25,
        elapsed_seconds=12,
    )

    result = _dispatch(registry, UPDATE_GOAL_TOOL_NAME, {"status": "complete"}, context)

    assert not result.is_error
    assert result.output["goal"]["status"] == "complete"
    assert result.output["goal"]["tokensUsed"] == 25
    assert "completedAt" not in result.output["goal"]
    assert result.output["remainingTokens"] == 75
    assert "Goal achieved" in result.output["completionBudgetReport"]


def test_evaluator_goal_hides_and_rejects_model_completion(tmp_path: Path):
    registry = _goal_registry()
    context = _goal_context(tmp_path)
    context.goal_service.replace_goal(
        "thread-1",
        "independently verified",
        completion_mode=GoalCompletionMode.EVALUATOR,
    )

    visible = filter_goal_model_tools_for_context(registry.list_tools(), context)
    result = _dispatch(registry, UPDATE_GOAL_TOOL_NAME, {"status": "complete"}, context)

    assert UPDATE_GOAL_TOOL_NAME not in {tool.name for tool in visible}
    assert result.is_error
    assert "independent evaluator" in result.output["error"]
    goal = context.goal_service.get_goal("thread-1")
    assert goal is not None
    assert goal.status is ThreadGoalStatus.ACTIVE


def test_update_goal_can_mark_blocked(tmp_path: Path):
    registry = _goal_registry()
    context = _goal_context(tmp_path)
    _dispatch(registry, CREATE_GOAL_TOOL_NAME, {"objective": "blocked"}, context)

    result = _dispatch(registry, UPDATE_GOAL_TOOL_NAME, {"status": "blocked"}, context)

    assert not result.is_error
    assert result.output["goal"]["status"] == "blocked"
    assert result.output["completionBudgetReport"] is None


def test_update_goal_returns_model_facing_error_without_existing_goal(tmp_path: Path):
    registry = _goal_registry()
    context = _goal_context(tmp_path)

    result = _dispatch(registry, UPDATE_GOAL_TOOL_NAME, {"status": "complete"}, context)

    assert result.is_error
    assert "this thread has no goal" in result.output["error"]


@pytest.mark.parametrize(
    "status",
    ["active", "paused", "usage_limited", "budget_limited"],
)
def test_update_goal_rejects_model_forbidden_statuses(tmp_path: Path, status: str):
    registry = _goal_registry()
    context = _goal_context(tmp_path)
    _dispatch(registry, CREATE_GOAL_TOOL_NAME, {"objective": "stay active"}, context)
    tool = registry.get(UPDATE_GOAL_TOOL_NAME)
    assert tool is not None

    result = tool.call({"status": status}, context)

    assert result.is_error
    assert "complete or blocked" in result.output["error"]
    goal = context.goal_service.get_goal("thread-1")
    assert goal is not None
    assert goal.status is ThreadGoalStatus.ACTIVE


def test_update_goal_schema_rejects_objective_replacement(tmp_path: Path):
    registry = _goal_registry()
    context = _goal_context(tmp_path)
    _dispatch(registry, CREATE_GOAL_TOOL_NAME, {"objective": "original"}, context)

    with pytest.raises(ToolInputError, match="unexpected field"):
        _dispatch(
            registry,
            UPDATE_GOAL_TOOL_NAME,
            {"status": "complete", "objective": "replacement"},
            context,
        )

    goal = context.goal_service.get_goal("thread-1")
    assert goal is not None
    assert goal.objective == "original"


def test_goal_tools_return_clear_error_without_persistent_thread(tmp_path: Path):
    registry = _goal_registry()
    context = _goal_context(tmp_path, session_id=None)

    result = _dispatch(
        registry,
        CREATE_GOAL_TOOL_NAME,
        {"objective": "cannot persist"},
        context,
    )

    assert result.is_error
    assert "saved session" in result.output["error"]
