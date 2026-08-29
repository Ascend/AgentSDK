#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

from __future__ import annotations

# pylint: disable=C1803,W0404,W0621

import pytest

from clawcodex_ext.cron_system.runs import read_cron_runs
from clawcodex_ext.cron_system.runtime import attach_cron_runtime, replace_cron_tools
from clawcodex_ext.cron_system.tools import CronCreateTool, CronListTool, CronRunTool
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.errors import ToolInputError
from src.tool_system.tools.cron import CronCreateTool as FallbackCronCreateTool


class _Runtime:
    def __init__(self, tmp_path):
        self.workspace_root = tmp_path
        self.tool_context = ToolContext(workspace_root=tmp_path)


def test_replace_cron_tools_swaps_fallback_implementation() -> None:
    registry = build_default_registry(provider=None)
    assert registry.get("CronCreate") is FallbackCronCreateTool
    replace_cron_tools(registry)
    assert registry.get("CronCreate") is CronCreateTool
    assert registry.get("CronRun") is CronRunTool


def test_extension_tools_persist_durable_tasks_by_default(tmp_path) -> None:
    """CronCreate defaults durable=True (matches the LLM prompt docs)."""
    ctx = ToolContext(workspace_root=tmp_path)
    created = CronCreateTool.call({"cron": "*/5 * * * *", "prompt": "ping"}, ctx).output
    assert len(created["id"]) == 8
    assert created["durable"] is True
    assert (tmp_path / ".clawcodex" / "cron" / "scheduled_tasks.json").exists()
    listed = registry_tool("CronList").call({}, ctx).output
    assert [job["id"] for job in listed["jobs"]] == [created["id"]]
    deleted = registry_tool("CronDelete").call({"id": created["id"]}, ctx).output
    assert deleted["success"] is True


def test_extension_tools_store_session_tasks_when_durable_false(tmp_path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    created = CronCreateTool.call({"cron": "*/5 * * * *", "prompt": "ping", "durable": False}, ctx).output
    assert created["durable"] is False
    listed = registry_tool("CronList").call({}, ctx).output
    assert [job["id"] for job in listed["jobs"]] == [created["id"]]
    assert not (tmp_path / ".clawcodex" / "cron" / "scheduled_tasks.json").exists()


def test_extension_tools_persist_durable_tasks(tmp_path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    created = CronCreateTool.call({"cron": "*/5 * * * *", "prompt": "ping", "durable": True}, ctx).output
    assert created["durable"] is True
    assert (tmp_path / ".clawcodex" / "cron" / "scheduled_tasks.json").exists()
    listed = registry_tool("CronList").call({}, ctx).output
    assert [job["id"] for job in listed["jobs"]] == [created["id"]]


def test_extension_delete_missing_task_errors(tmp_path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    with pytest.raises(ToolInputError, match="No scheduled job"):
        registry_tool("CronDelete").call({"id": "missing"}, ctx)


def test_mutating_cron_tools_are_not_read_only() -> None:
    assert CronCreateTool.is_read_only({}) is False
    assert registry_tool("CronDelete").is_read_only({}) is False
    assert registry_tool("CronRun").is_read_only({}) is False
    assert registry_tool("CronList").is_read_only({}) is True


def test_cron_run_tool_creates_queued_run(tmp_path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    created = CronCreateTool.call(
        {"cron": "*/5 * * * *", "prompt": "manual ping", "durable": True},
        ctx,
    ).output

    result = registry_tool("CronRun").call({"id": created["id"]}, ctx).output

    assert result["success"] is True
    assert result["id"] == created["id"]
    assert result["run"]["task_id"] == created["id"]
    assert result["run"]["prompt"] == "manual ping"
    assert result["run"]["status"] == "queued"
    runs = read_cron_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].id == result["run"]["id"]


def test_cron_run_tool_blocks_duplicate_active_run(tmp_path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    created = CronCreateTool.call(
        {"cron": "*/5 * * * *", "prompt": "manual ping", "durable": True},
        ctx,
    ).output
    run_tool = registry_tool("CronRun")

    first = run_tool.call({"id": created["id"]}, ctx).output
    second = run_tool.call({"id": created["id"]}, ctx).output

    assert first["success"] is True
    assert second == {"success": False, "id": created["id"], "run": None}
    assert len(read_cron_runs(tmp_path)) == 1


def test_cron_run_tool_reports_missing_task(tmp_path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)

    result = registry_tool("CronRun").call({"id": "missing"}, ctx).output

    assert result == {"success": False, "id": "missing", "not_found": True}
    assert read_cron_runs(tmp_path) == []


def test_cron_run_tool_respects_kill_switch(tmp_path, monkeypatch) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    created = CronCreateTool.call(
        {"cron": "*/5 * * * *", "prompt": "manual ping", "durable": True},
        ctx,
    ).output

    monkeypatch.setenv("CLAWCODEX_DISABLE_CRON", "1")
    result = registry_tool("CronRun").call({"id": created["id"]}, ctx).output

    assert result["disabled"] is True
    assert read_cron_runs(tmp_path) == []


# ---------------------------------------------------------------------------
# Phase D-3: dual-durable coverage at the tool-API level.
# ---------------------------------------------------------------------------


def test_durable_false_and_true_both_visible_in_list(tmp_path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    session = CronCreateTool.call({"cron": "*/5 * * * *", "prompt": "session", "durable": False}, ctx).output
    durable = CronCreateTool.call({"cron": "*/5 * * * *", "prompt": "durable", "durable": True}, ctx).output
    assert session["durable"] is False
    assert durable["durable"] is True
    listed = registry_tool("CronList").call({}, ctx).output
    prompts = {job["prompt"] for job in listed["jobs"]}
    assert {"session", "durable"} <= prompts


def test_durable_false_delete_works(tmp_path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    created = CronCreateTool.call({"cron": "*/5 * * * *", "prompt": "session", "durable": False}, ctx).output
    # Session tasks live in ctx.crons (in-memory). CronDelete must find
    # them there since the file is never written.
    deleted = registry_tool("CronDelete").call({"id": created["id"]}, ctx).output
    assert deleted["success"] is True
    listed = registry_tool("CronList").call({}, ctx).output
    assert listed["jobs"] == []


def test_durable_false_path_not_written_to_disk(tmp_path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    CronCreateTool.call({"cron": "*/5 * * * *", "prompt": "session", "durable": False}, ctx)
    # durable=False tasks must NOT touch the persisted file
    assert not (tmp_path / ".clawcodex" / "cron" / "scheduled_tasks.json").exists()
    # but they must be visible in the session store
    assert "crons" in dir(ctx) or hasattr(ctx, "crons")
    assert any(t.prompt == "session" for t in ctx.crons.values())


def registry_tool(name: str):
    registry = build_default_registry(provider=None)
    replace_cron_tools(registry)
    tool = registry.get(name)
    assert tool is not None
    return tool


# ---- Regression: issue #14 — custom is_killed must gate outbox callbacks ----


def test_custom_is_killed_gates_outbox_callbacks(tmp_path) -> None:
    """When a custom is_killed returns True, on_fire/on_fire_task/on_missed
    must NOT append to the outbox (they should use is_killed, not is_cron_disabled).
    """

    class _Ctx:
        workspace_root = tmp_path
        outbox: list = []
        crons: dict = {}

    ctx = _Ctx()
    killed = {"state": True}

    scheduler = attach_cron_runtime(
        ctx,
        autostart=False,
        is_killed=lambda: killed["state"],
    )
    # Even though is_cron_disabled() is False (no env var set), the custom
    # is_killed=True should suppress the outbox event.
    from clawcodex_ext.cron_system.models import is_cron_disabled

    assert is_cron_disabled() is False  # env var not set
    scheduler.on_fire("test prompt")
    assert ctx.outbox == [], f"outbox should be empty when is_killed=True, got {ctx.outbox}"

    # When is_killed returns False, the event should flow through.
    killed["state"] = False
    scheduler.on_fire("test prompt")
    assert len(ctx.outbox) == 1


# ---- Regression: issue #26 — agent_id derived from context, not tool_input ----


def test_cron_create_uses_context_agent_id_not_tool_input(tmp_path) -> None:
    """agent_id in tool_input must NOT override context.agent_id."""
    from src.tool_system.context import ToolContext

    ctx = ToolContext(workspace_root=tmp_path, crons={}, agent_id="real-agent")
    # LLM tries to spoof agent-B's identity via tool_input — must be ignored.
    CronCreateTool.call(
        {"cron": "*/5 * * * *", "prompt": "spoof", "durable": True, "agent_id": "evil-agent"},
        ctx,
    )
    from clawcodex_ext.cron_system.tasks import read_all_cron_tasks

    tasks = read_all_cron_tasks(tmp_path, ctx.crons)
    assert len(tasks) == 1
    assert tasks[0].agent_id == "real-agent"


def test_cron_list_cannot_spoof_agent_id(tmp_path) -> None:
    """CronList must filter by context.agent_id, not tool_input agent_id."""
    from src.tool_system.context import ToolContext
    from clawcodex_ext.cron_system.tasks import add_cron_task

    add_cron_task(tmp_path, cron="*/5 * * * *", prompt="a-task", durable=True, agent_id="agent-A")
    add_cron_task(tmp_path, cron="*/5 * * * *", prompt="b-task", durable=True, agent_id="agent-B")

    ctx = ToolContext(workspace_root=tmp_path, crons={}, agent_id="agent-A")
    # LLM passes agent_id="agent-B" in tool_input — must still see only agent-A's tasks.
    result = CronListTool.call({"agent_id": "agent-B"}, ctx)
    prompts = {job["prompt"] for job in result.output["jobs"]}
    assert prompts == {"a-task"}


def test_cron_delete_cannot_spoof_agent_id(tmp_path) -> None:
    """CronDelete must use context.agent_id for ownership, not tool_input."""
    from src.tool_system.context import ToolContext
    from clawcodex_ext.cron_system.tasks import add_cron_task
    from src.tool_system.errors import ToolInputError

    task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="owned-by-B", durable=True, agent_id="agent-B")
    ctx = ToolContext(workspace_root=tmp_path, crons={}, agent_id="agent-A")
    # LLM passes agent_id="agent-B" in tool_input — must still be rejected.
    with pytest.raises(ToolInputError, match="owned by agent"):
        registry_tool("CronDelete").call({"id": task.id, "agent_id": "agent-B"}, ctx)


# ---- Regression: issue #28 — max 50 tasks per workspace ----


def test_cron_create_rejects_when_max_tasks_reached(tmp_path) -> None:
    """CronCreate must reject when 50 tasks already exist."""
    from clawcodex_ext.cron_system.tasks import add_cron_task
    from clawcodex_ext.cron_system.tools import MAX_CRON_TASKS_PER_WORKSPACE
    from src.tool_system.errors import ToolInputError

    for i in range(MAX_CRON_TASKS_PER_WORKSPACE):
        add_cron_task(tmp_path, cron="*/5 * * * *", prompt=f"task-{i}", durable=True)
    ctx = ToolContext(workspace_root=tmp_path)
    with pytest.raises(ToolInputError, match="maximum"):
        CronCreateTool.call({"cron": "*/5 * * * *", "prompt": "overflow"}, ctx)
