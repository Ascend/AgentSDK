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

# AgentSDK validates these split-package and target-lint diagnostics in the complete tested source.
# pylint: disable=W0621
"""Phase 0 characterization tests for Task-v2 ToolResult shapes.

These tests FREEZE the current ToolResult shapes returned by each Task-v2
tool, for both LKB-off and LKB-on configurations.  They serve as regression
pins for the Phase 4 thin-adapter refactor — they assert CURRENT behavior
(including known quirks like the soft "Task not found" error), NOT a
specification of "correctness".

After the flag merge (``logical_kanban`` → ``LKB_PLAN_GRAPH``), LKB-on pins
the Plan Graph host-adapter shapes: claim-before-start, denial payloads with
``lkb.decision``/``validationRunId``, and hydrated per-task ``lkb`` views.

Run:
    wsl.exe -e bash -lc 'cd /mnt/e/code/clawcodex && .venv/bin/python -m pytest extensions/lkb/tests/unit/test_characterize_task_tools.py -q'
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from clawcodex_ext.feature_gate import get_registry
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.protocol import ToolResult
from clawcodex_ext.tool_system.tools.tasks_v2 import (
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskOutputTool,
    TaskUpdateTool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ToolContext:
    """A ToolContext with tmp workspace and TaskV2 force-enabled."""
    # Ensure TaskV2 tools are enabled regardless of session mode.
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TASKS", "1")
    return ToolContext(workspace_root=tmp_path)


@pytest.fixture
def lkb_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure LKB_PLAN_GRAPH is OFF regardless of persisted user config."""
    monkeypatch.delenv("CLAWCODEX_FEATURE_LKB_PLAN_GRAPH", raising=False)
    get_registry()._overrides["LKB_PLAN_GRAPH"] = False


@pytest.fixture
def lkb_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable LKB_PLAN_GRAPH via env + runtime override (store in tmp home)."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("CLAWCODEX_HOME", str(home))
    monkeypatch.setenv("CLAWCODEX_FEATURE_LKB_PLAN_GRAPH", "1")
    get_registry()._overrides["LKB_PLAN_GRAPH"] = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call_tool(tool: Any, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """Invoke a Task-v2 tool's callable directly and return the ToolResult.

    Handles both sync and async tool callables.  TaskOutput is async;
    all other Task-v2 tools are sync.
    """
    import inspect

    result = tool.call(tool_input, context)
    if inspect.iscoroutine(result):
        # asyncio.run() creates a fresh loop each call - robust against
        # loop-policy state left by prior async tests in the same session
        # (asyncio.get_event_loop() is deprecated/fragile in 3.11+).
        result = asyncio.run(result)
    return result


def _create_task(
    ctx: ToolContext,
    *,
    subject: str = "Test task",
    description: str = "A test task description",
) -> str:
    """Create a task via TaskCreate and return its id."""
    result = _call_tool(
        TaskCreateTool,
        {"subject": subject, "description": description},
        ctx,
    )
    return result.output["task"]["id"]


# ===================================================================
# TaskCreate
# ===================================================================


class TestTaskCreateShape:
    """Freeze TaskCreate ToolResult shape."""

    def test_lkb_off_success_shape(self, ctx: ToolContext, lkb_off: None) -> None:
        result = _call_tool(
            TaskCreateTool,
            {"subject": "My task", "description": "My description"},
            ctx,
        )
        assert isinstance(result, ToolResult)
        assert result.name == "TaskCreate"
        assert result.is_error is False
        assert isinstance(result.output, dict)

        output = result.output
        assert "task" in output
        assert "lkb" not in output  # LKB off -> no lkb key

        task = output["task"]
        assert isinstance(task["id"], str) and len(task["id"]) > 0
        assert task["subject"] == "My task"
        # Only id+subject in the create response (full detail via TaskGet)
        assert set(task.keys()) == {"id", "subject"}

    def test_lkb_on_success_shape_and_hydrated_view(self, ctx: ToolContext, lkb_on: None) -> None:
        """Plan Graph routing: the create response keeps the minimal
        ``{"task": {id, subject}}`` shape (no ``lkb`` key); the LKB derived
        view is hydrated onto ``context.tasks`` instead.
        """
        result = _call_tool(
            TaskCreateTool,
            {"subject": "My task", "description": "My description"},
            ctx,
        )
        assert result.is_error is False
        output = result.output
        assert "task" in output
        assert "lkb" not in output  # adapter create response has no lkb key

        task = output["task"]
        assert set(task.keys()) == {"id", "subject"}

        hydrated = ctx.tasks[task["id"]]
        assert hydrated["status"] == "pending"
        assert hydrated["lkb"]["derivedStatus"] == "ready"
        assert hydrated["lkb"]["claimable"] is True


# ===================================================================
# TaskGet
# ===================================================================


class TestTaskGetShape:
    """Freeze TaskGet ToolResult shape."""

    def test_lkb_off_task_found_shape(self, ctx: ToolContext, lkb_off: None) -> None:
        tid = _create_task(ctx, subject="Alpha", description="Alpha desc")
        result = _call_tool(TaskGetTool, {"taskId": tid}, ctx)

        assert result.name == "TaskGet"
        assert result.is_error is False

        task = result.output["task"]
        assert task is not None
        assert task["id"] == tid
        assert task["subject"] == "Alpha"
        assert task["description"] == "Alpha desc"
        assert task["status"] == "pending"
        assert isinstance(task["blocks"], list)
        assert isinstance(task["blockedBy"], list)
        assert "lkb" not in task  # LKB off

        # Exact key set — pin current shape
        assert set(task.keys()) == {
            "id",
            "subject",
            "description",
            "status",
            "blocks",
            "blockedBy",
        }

    def test_lkb_on_task_found_has_lkb_view(self, ctx: ToolContext, lkb_on: None) -> None:
        tid = _create_task(ctx, subject="Beta", description="Beta desc")
        result = _call_tool(TaskGetTool, {"taskId": tid}, ctx)

        task = result.output["task"]
        assert "lkb" in task  # LKB on -> task has lkb view

        lkb = task["lkb"]
        assert isinstance(lkb, dict)
        # Pin the Plan Graph derived view shape (host adapter hydration).
        expected_keys = {
            "derivedStatus",
            "claimable",
            "activeBlockers",
            "validation",
            "consistency",
            "nextActions",
        }
        assert expected_keys.issubset(set(lkb.keys()))
        assert lkb["derivedStatus"] in {"ready", "blocked", "needs_recheck", "running", "verified"}
        assert lkb["validation"]["status"] in {"unvalidated", "validated"}

    def test_lkb_off_task_not_found(self, ctx: ToolContext, lkb_off: None) -> None:
        result = _call_tool(TaskGetTool, {"taskId": "nonexistent"}, ctx)
        assert result.is_error is False
        assert result.output["task"] is None

    def test_lkb_on_task_not_found(self, ctx: ToolContext, lkb_on: None) -> None:
        result = _call_tool(TaskGetTool, {"taskId": "nonexistent"}, ctx)
        assert result.is_error is False
        # Task not found -> task is None (no lkb view attached)
        assert result.output["task"] is None


# ===================================================================
# TaskList
# ===================================================================


class TestTaskListShape:
    """Freeze TaskList ToolResult shape."""

    def test_lkb_off_empty_list(self, ctx: ToolContext, lkb_off: None) -> None:
        result = _call_tool(TaskListTool, {}, ctx)
        assert result.name == "TaskList"
        assert result.is_error is False
        assert isinstance(result.output["tasks"], list)
        assert result.output["tasks"] == []

    def test_lkb_off_row_shape(self, ctx: ToolContext, lkb_off: None) -> None:
        tid = _create_task(ctx, subject="List task", description="desc")
        result = _call_tool(TaskListTool, {}, ctx)

        tasks = result.output["tasks"]
        assert len(tasks) == 1
        row = tasks[0]
        assert row["id"] == tid
        assert row["subject"] == "List task"
        assert row["status"] == "pending"
        assert isinstance(row["blockedBy"], list)
        assert row["owner"] is None
        assert "lkb" not in row  # LKB off
        expected = {
            "id",
            "subject",
            "description",
            "activeForm",
            "status",
            "owner",
            "blocks",
            "blockedBy",
            "metadata",
            "output",
        }
        assert set(row.keys()) == expected

    def test_lkb_off_row_with_owner(self, ctx: ToolContext, lkb_off: None) -> None:
        tid = _create_task(ctx)
        ctx.tasks[tid]["owner"] = "agent-1"
        result = _call_tool(TaskListTool, {}, ctx)
        row = result.output["tasks"][0]
        assert row["owner"] == "agent-1"
        assert "owner" in row

    def test_lkb_on_row_has_lkb(self, ctx: ToolContext, lkb_on: None) -> None:
        _create_task(ctx, subject="LKB list task", description="desc")
        result = _call_tool(TaskListTool, {}, ctx)

        tasks = result.output["tasks"]
        assert len(tasks) == 1
        row = tasks[0]
        assert "lkb" in row  # LKB on -> per-row lkb
        assert isinstance(row["lkb"], dict)
        # Row lkb does NOT include proofTraceSummary (include_proof_trace=False default)
        assert "proofTraceSummary" not in row["lkb"]


# ===================================================================
# TaskUpdate
# ===================================================================


class TestTaskUpdateShape:
    """Freeze TaskUpdate ToolResult shape."""

    # -- success paths --

    def test_lkb_off_update_subject_shape(self, ctx: ToolContext, lkb_off: None) -> None:
        tid = _create_task(ctx)
        result = _call_tool(
            TaskUpdateTool,
            {"taskId": tid, "subject": "New subject"},
            ctx,
        )
        assert result.name == "TaskUpdate"
        assert result.is_error is False

        out = result.output
        assert out["success"] is True
        assert out["taskId"] == tid
        assert out["updatedFields"] == ["subject"]
        assert "statusChange" not in out
        assert "lkb" not in out  # LKB off

    def test_lkb_off_status_change_shape(self, ctx: ToolContext, lkb_off: None) -> None:
        tid = _create_task(ctx)
        result = _call_tool(
            TaskUpdateTool,
            {"taskId": tid, "status": "in_progress"},
            ctx,
        )
        out = result.output
        assert out["success"] is True
        assert "status" in out["updatedFields"]
        assert out["statusChange"] == {"from": "pending", "to": "in_progress"}
        assert "lkb" not in out

    def test_lkb_on_update_shape(self, ctx: ToolContext, lkb_on: None) -> None:
        tid = _create_task(ctx)
        result = _call_tool(
            TaskUpdateTool,
            {"taskId": tid, "subject": "Updated"},
            ctx,
        )
        out = result.output
        assert out["success"] is True
        assert out["updatedFields"] == ["subject"]
        assert "lkb" not in out  # adapter success response has no lkb key

    def test_lkb_on_status_transition_shape(self, ctx: ToolContext, lkb_on: None) -> None:
        tid = _create_task(ctx)
        # Plan Graph: claim before start (self-claim via owner == actor).
        ctx.agent_id = "agent-a"
        claim = _call_tool(TaskUpdateTool, {"taskId": tid, "owner": "agent-a"}, ctx)
        assert claim.output["success"] is True
        assert claim.output["claimId"].startswith("C-")
        result = _call_tool(
            TaskUpdateTool,
            {"taskId": tid, "status": "in_progress"},
            ctx,
        )
        out = result.output
        assert out["success"] is True
        assert out["statusChange"] == {"from": "pending", "to": "in_progress"}
        assert "lkb" not in out

    # -- not-found path (soft error, pin this behavior) --

    def test_lkb_off_not_found_is_soft_error(self, ctx: ToolContext, lkb_off: None) -> None:
        """T2-GAP-09 baseline: not-found returns success=False but is_error=False.

        This is a characterization pin, NOT an assertion of correctness.
        """
        result = _call_tool(
            TaskUpdateTool,
            {"taskId": "does-not-exist", "subject": "x"},
            ctx,
        )
        # Soft error: is_error is False, success is False in payload
        assert result.is_error is False
        out = result.output
        assert out["success"] is False
        assert out["taskId"] == "does-not-exist"
        assert out["updatedFields"] == []
        assert out["error"] == "Task not found"
        assert "lkb" not in out

    def test_lkb_on_not_found_is_denied(self, ctx: ToolContext, lkb_on: None) -> None:
        """LKB on + not-found: the Plan Graph validator denies unknown task IDs.

        Characterization pin: when LKB is enabled, updating a nonexistent
        task returns a *denied* result (is_error=True) rather than the
        LKB-off soft-error shape.
        """
        result = _call_tool(
            TaskUpdateTool,
            {"taskId": "does-not-exist", "subject": "x"},
            ctx,
        )
        # LKB denies unknown tasks — is_error=True
        assert result.is_error is True
        out = result.output
        assert out["success"] is False
        assert out["taskId"] == "does-not-exist"
        assert out["status"] == "denied"
        assert "task_not_found" in out["reason"]["message"]
        assert "lkb" in out
        assert out["lkb"]["decision"] == "denied"
        assert out["lkb"]["validationRunId"].startswith("V-")

    # -- denied path (LKB on, blocked task start) --

    def test_lkb_on_start_blocked_task_is_denied(self, ctx: ToolContext, lkb_on: None) -> None:
        """Starting a blocked task is denied: is_error=True with denied payload."""
        # Create two tasks: A and B, B is blocked by A
        a_id = _create_task(ctx, subject="A", description="dep target")
        b_id = _create_task(ctx, subject="B", description="blocked task")
        # Set up B blocked by A via TaskUpdate addBlockedBy
        _call_tool(
            TaskUpdateTool,
            {"taskId": b_id, "addBlockedBy": [a_id]},
            ctx,
        )

        # Now try to start B while A is still pending
        result = _call_tool(
            TaskUpdateTool,
            {"taskId": b_id, "status": "in_progress"},
            ctx,
        )

        # Denied -> is_error is True
        assert result.is_error is True
        out = result.output
        assert out["success"] is False
        assert out["status"] == "denied"
        assert "blocked" in out["reason"]["message"]
        assert "lkb" in out
        assert out["lkb"]["decision"] == "denied"
        assert out["lkb"]["validationRunId"].startswith("V-")

    # -- delete --

    def test_lkb_off_delete_shape(self, ctx: ToolContext, lkb_off: None) -> None:
        tid = _create_task(ctx)
        result = _call_tool(
            TaskUpdateTool,
            {"taskId": tid, "status": "deleted"},
            ctx,
        )
        out = result.output
        assert out["success"] is True
        assert out["taskId"] == tid
        assert out["updatedFields"] == ["deleted"]
        assert "statusChange" not in out  # delete does not produce statusChange
        assert "lkb" not in out
        # Task is actually removed
        assert tid not in ctx.tasks

    def test_lkb_on_delete_shape(self, ctx: ToolContext, lkb_on: None) -> None:
        tid = _create_task(ctx)
        result = _call_tool(
            TaskUpdateTool,
            {"taskId": tid, "status": "deleted"},
            ctx,
        )
        out = result.output
        assert out["success"] is True
        assert out["updatedFields"] == ["deleted"]
        assert "lkb" not in out


# ===================================================================
# TaskOutput
# ===================================================================


class TestTaskOutputShape:
    """Freeze TaskOutput ToolResult shape — pin that it does NOT go through
    the task-v2 validation adapter (it uses the runtime output registry).
    """

    def test_lkb_off_task_not_found(self, ctx: ToolContext, lkb_off: None) -> None:
        result = _call_tool(TaskOutputTool, {"task_id": "ghost"}, ctx)
        assert result.name == "TaskOutput"
        assert result.is_error is False
        out = result.output
        assert out["retrieval_status"] == "success"
        assert out["task"] is None

    def test_lkb_off_task_list_task_no_output(self, ctx: ToolContext, lkb_off: None) -> None:
        tid = _create_task(ctx, subject="Out task", description="out desc")
        result = _call_tool(TaskOutputTool, {"task_id": tid}, ctx)
        out = result.output

        assert out["retrieval_status"] == "not_ready"  # no output yet
        task = out["task"]
        assert task is not None
        assert task["task_id"] == tid
        assert task["task_type"] == "task_list"
        assert task["status"] == "pending"
        assert task["description"] == "out desc"
        assert task["output"] == ""
        # TaskOutput (task_list branch) does NOT have task-level lkb when LKB off
        assert "lkb" not in task

        # Pin the exact key set for task_list branch
        assert set(task.keys()) == {
            "task_id",
            "task_type",
            "status",
            "description",
            "output",
        }

    def test_lkb_on_task_list_task_has_lkb_view(self, ctx: ToolContext, lkb_on: None) -> None:
        """LKB on: TaskOutput exposes the hydrated Plan Graph read projection."""
        tid = _create_task(ctx, subject="Out task LKB", description="out desc lkb")
        result = _call_tool(TaskOutputTool, {"task_id": tid}, ctx)
        out = result.output
        task = out["task"]
        assert task is not None
        assert "lkb" in task

        lkb = task["lkb"]
        expected_subset = {
            "derivedStatus",
            "claimable",
            "activeBlockers",
            "validation",
            "consistency",
            "nextActions",
            "nextActionCommands",
        }
        assert expected_subset.issubset(set(lkb.keys()))

    def test_task_output_uses_read_projection_not_mutation_result(self, ctx: ToolContext, lkb_on: None) -> None:
        """TaskOutput returns task state, not a mutation decision envelope."""
        tid = _create_task(ctx)
        result = _call_tool(TaskOutputTool, {"task_id": tid}, ctx)
        lkb = result.output["task"]["lkb"]

        mutation_fields = {
            "decision",
            "validationRunId",
            "updatedFields",
            "statusChange",
        }
        assert mutation_fields.isdisjoint(lkb)
        assert lkb["derivedStatus"] == "ready"
        assert lkb["claimable"] is True

    def test_lkb_on_task_with_output_is_success(self, ctx: ToolContext, lkb_on: None) -> None:
        """Task with output text -> retrieval_status=success."""
        tid = _create_task(ctx)
        ctx.tasks[tid]["output"] = "some result"
        result = _call_tool(TaskOutputTool, {"task_id": tid}, ctx)
        assert result.output["retrieval_status"] == "success"
        assert result.output["task"]["output"] == "some result"
