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

"""Focused unit tests for deterministic decomposition."""

import json
from types import SimpleNamespace

import pytest

from extensions.orchestrator.task_decomposition import (
    TaskDecomposer,
    build_swarm_prompt,
    validate_task_execution,
    write_task_plan,
)


@pytest.mark.asyncio
async def test_explicit_tasks_form_dependency_waves(tmp_path) -> None:
    issue = SimpleNamespace(
        id="F-118",
        identifier="F-118",
        title="Migrate feature",
        description="- Inspect src/app.py\n- Implement src/app.py\n- Test the result",
    )
    plan = await TaskDecomposer(max_parallel=2).decompose_issue(issue)

    assert [task.id for task in plan.subtasks] == ["task-1", "task-2", "task-3"]
    assert plan.subtasks[-1].depends_on == ("task-1", "task-2")

    path = write_task_plan(plan, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["goal"] == "Migrate feature"
    assert "Work wave by wave" in build_swarm_prompt(issue, plan, path)


def test_execution_evidence_must_cover_every_task(tmp_path) -> None:
    issue = SimpleNamespace(id="F-118", identifier="F-118", title="One", description="")
    decomposer = TaskDecomposer(max_subtasks=1, max_parallel=1, max_waves=1)
    plan = decomposer._bounded_fallback_tasks(issue.title, issue.description)
    seed = __import__(
        "extensions.orchestrator.task_decomposition.models",
        fromlist=["TaskPlan"],
    ).TaskPlan(goal="One", subtasks=tuple(plan), waves=(("task-1",),), max_parallel=1)
    path = tmp_path / "task_decomposition.json"
    path.write_text('{"subtasks": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="exactly the seed task ids"):
        validate_task_execution(path, seed)


@pytest.mark.asyncio
async def test_markdown_file_references_are_serialized() -> None:
    issue = SimpleNamespace(
        id="issue-1",
        identifier="issue-1",
        title="Update application",
        description=("- Update `src/app.py` implementation\n- Refactor `src/app.py` interface"),
    )

    plan = await TaskDecomposer(max_parallel=2).decompose_issue(issue)

    assert plan.subtasks[0].affected_files == ("src/app.py",)
    assert plan.subtasks[1].affected_files == ("src/app.py",)
    assert plan.subtasks[1].depends_on == ("task-1",)
    assert plan.waves == (("task-1",), ("task-2",))


@pytest.mark.asyncio
async def test_llm_cannot_raise_configured_parallel_limit() -> None:
    async def llm_client(_prompt: str) -> str:
        return json.dumps(
            {
                "goal": "Parallel work",
                "subtasks": [{"id": f"task-{index}", "title": f"Task {index}"} for index in range(1, 5)],
                "waves": [["task-1", "task-2", "task-3", "task-4"]],
                "max_parallel": 4,
            }
        )

    issue = SimpleNamespace(
        id="issue-2",
        identifier="issue-2",
        title="Parallel work",
        description="Implement the change",
    )
    decomposer = TaskDecomposer(
        max_parallel=2,
        planner_strategy="restructure",
        llm_client=llm_client,
    )

    plan = await decomposer.decompose_issue(issue)

    assert plan.max_parallel == 2
    assert all(len(wave) <= 2 for wave in plan.waves)


@pytest.mark.asyncio
async def test_explicit_plan_fallback_logs_when_wave_limit_is_exceeded(caplog) -> None:
    issue = SimpleNamespace(
        id="issue-3",
        identifier="issue-3",
        title="Sequential work",
        description="- First task\n- Then second task\n- Then third task",
    )

    with caplog.at_level("WARNING"):
        plan = await TaskDecomposer(max_waves=2).decompose_issue(issue)

    assert "dropping explicit task list" in caplog.text
    assert plan.fallback_reason == "explicit plan required 3 waves; configured maximum is 2"


@pytest.mark.asyncio
async def test_llm_non_object_json_falls_back_to_seed_plan(caplog) -> None:
    async def llm_client(_prompt: str) -> str:
        return "[]"

    issue = SimpleNamespace(
        id="issue-non-object",
        identifier="issue-non-object",
        title="Keep seed plan",
        description="Implement the change",
    )
    seed = await TaskDecomposer(max_parallel=2).decompose_issue(issue)

    with caplog.at_level("WARNING"):
        revised = await TaskDecomposer(
            max_parallel=2,
            planner_strategy="restructure",
            llm_client=llm_client,
        ).decompose_issue(issue)

    assert revised == seed
    assert "response must be a JSON object" in caplog.text


def test_task_plan_namespaces_isolate_control_files(tmp_path) -> None:
    issue = SimpleNamespace(id="issue-4", identifier="issue-4", title="One", description="")
    decomposer = TaskDecomposer(max_subtasks=1, max_parallel=1, max_waves=1)
    subtasks = tuple(decomposer._bounded_fallback_tasks(issue.title, issue.description))
    task_plan_type = __import__(
        "extensions.orchestrator.task_decomposition.models",
        fromlist=["TaskPlan"],
    ).TaskPlan
    plan = task_plan_type(goal="One", subtasks=subtasks, waves=(("task-1",),), max_parallel=1)

    first = write_task_plan(plan, tmp_path, namespace="issue-4")
    second = write_task_plan(plan, tmp_path, namespace="issue-5")

    assert first != second
    assert first.parent.name == "issue-4"
    assert second.parent.name == "issue-5"


@pytest.mark.asyncio
async def test_refine_cannot_change_task_identity_or_waves() -> None:
    async def llm_client(_prompt: str) -> str:
        return json.dumps(
            {
                "goal": "Changed",
                "subtasks": [
                    {"id": "replacement", "title": "Replacement"},
                    {"id": "extra", "title": "Extra"},
                ],
                "waves": [["replacement", "extra"]],
                "max_parallel": 2,
            }
        )

    issue = SimpleNamespace(
        id="issue-5",
        identifier="issue-5",
        title="Keep seed structure",
        description="Implement the change",
    )
    decomposer = TaskDecomposer(
        max_parallel=2,
        planner_strategy="refine",
        llm_client=llm_client,
    )

    seed = await TaskDecomposer(max_parallel=2).decompose_issue(issue)
    refined = await decomposer.decompose_issue(issue)

    assert refined == seed
