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

"""Focused construction test for Swarm mode after A.12 is available."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace


git_sync = ModuleType("extensions.orchestrator.git_sync")


class VerificationFailed(RuntimeError):
    """Minimal A.3 boundary used while Git Sync migrates independently."""


git_sync.VerificationFailed = VerificationFailed
sys.modules.setdefault("extensions.orchestrator.git_sync", git_sync)

from extensions.orchestrator.multi_agent.modes.swarm import (  # noqa: E402
    SwarmModeRunner,
    _control_namespace,
)
from extensions.orchestrator.provider_routing import provider_name  # noqa: E402
from extensions.orchestrator.task_decomposition.models import Subtask, TaskPlan  # noqa: E402


def test_swarm_configures_bounded_decomposer() -> None:
    runner = SwarmModeRunner(object(), max_subtasks=5, max_parallel=2, max_waves=4)

    assert runner._decomposer.max_subtasks == 5
    assert runner._decomposer.max_parallel == 2
    assert runner._decomposer.max_waves == 4


def test_control_namespace_is_stable_and_issue_scoped() -> None:
    first = SimpleNamespace(issue=SimpleNamespace(id="owner/issue-1", identifier="ISSUE-1"))
    second = SimpleNamespace(issue=SimpleNamespace(id="owner/issue-2", identifier="ISSUE-2"))

    assert _control_namespace(first) == "swarm-owner_issue-1"
    assert _control_namespace(first) != _control_namespace(second)


class _PlanDecomposer:
    async def decompose_issue(self, issue):  # noqa: ANN001, ANN201, ARG002
        return TaskPlan(
            goal="route swarm",
            subtasks=(Subtask(id="task-1", title="task", description="task"),),
            waves=(("task-1",),),
            max_parallel=1,
        )


class _RecordingAgentRunner:
    def __init__(self) -> None:
        self.calls = []

    async def run(self, session, workflow, **hooks):  # noqa: ANN001, ANN201, ARG002
        self.calls.append(dict(hooks))
        session.status = "failed"


def test_swarm_routes_provider_and_model_without_shared_mutation(tmp_path) -> None:
    agent_runner = _RecordingAgentRunner()
    runner = SwarmModeRunner(agent_runner, max_subtasks=1, max_parallel=1, max_waves=1)
    runner._decomposer = _PlanDecomposer()
    workflow = SimpleNamespace(
        agent=SimpleNamespace(
            provider="deepseek",
            model="default-model",
            stage_overrides={
                "swarm": {"provider": "openrouter", "model": "swarm-model"},
            },
        )
    )
    session = SimpleNamespace(
        issue=SimpleNamespace(id="issue-1", title="swarm", description="body"),
        workspace=SimpleNamespace(path=tmp_path),
        prompt_override=None,
        run_kind="issue",
        status="running",
    )

    asyncio.run(runner.run(session, workflow))

    assert len(agent_runner.calls) == 1
    assert provider_name(agent_runner.calls[0]["provider_override"]) == "openrouter"
    assert agent_runner.calls[0]["model_override"] == "swarm-model"
    assert workflow.agent.model == "default-model"
