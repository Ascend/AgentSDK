# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Integration tests for A.10 routing in Pipeline and Debate modes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from extensions.orchestrator.contracts.provider_routing import StageModel
from extensions.orchestrator.multi_agent.modes.coordinator import CoordinatorModeRunner
from extensions.orchestrator.multi_agent.modes.debate import DebateModeRunner
from extensions.orchestrator.multi_agent.modes.pipeline import PipelineModeRunner
from extensions.orchestrator.provider_routing import (
    ProviderReference,
    StaticProviderRouter,
    provider_name,
)


class _RecordingAgentRunner:
    def __init__(self) -> None:
        self.max_turns = 20
        self.calls: list[dict[str, object]] = []

    async def run(self, session, workflow, **hooks) -> None:  # noqa: ANN001
        self.calls.append(
            {
                "run_kind": session.run_kind,
                "hooks": dict(hooks),
                "shared_model": workflow.agent.model,
            }
        )
        session.status = "completed"
        session.output_text = f"completed {session.run_kind}"


def _session(tmp_path):  # noqa: ANN001, ANN202
    return SimpleNamespace(
        issue=SimpleNamespace(id="issue-1", title="route models", description="body"),
        workspace=SimpleNamespace(path=tmp_path),
        turn_count=0,
        status="running",
        output_text="",
        session_end_reason=None,
        session_end_summary="",
        run_id=None,
        run_kind="",
        prompt_override=None,
        consecutive_429_count=0,
        rate_limit_pending_turn=None,
    )


def _router(provider: str) -> StaticProviderRouter:
    return StaticProviderRouter(
        ProviderReference(provider),
        "default-model",
        stage_models=(
            StageModel("analyzer", "router-analyzer"),
            StageModel("security", "router-proposer"),
            StageModel("judge", "router-judge"),
        ),
    )


def _workflow(*, stage_overrides=None):  # noqa: ANN001, ANN202
    return SimpleNamespace(
        agent=SimpleNamespace(
            provider="deepseek",
            model="shared-default",
            stage_overrides=stage_overrides or {},
        )
    )


def test_pipeline_passes_route_without_mutating_shared_workflow(tmp_path) -> None:
    provider = "deepseek"
    agent_runner = _RecordingAgentRunner()
    workflow = _workflow()
    runner = PipelineModeRunner(
        agent_runner,
        stages=("analyzer",),
        stage_models={"analyzer": "configured-analyzer"},
        provider_router=_router(provider),
    )

    results = asyncio.run(runner.run(_session(tmp_path), workflow))

    assert [result.status for result in results] == ["completed"]
    hooks = agent_runner.calls[0]["hooks"]
    assert provider_name(hooks["provider_override"]) == provider
    assert hooks["model_override"] == "router-analyzer"
    assert agent_runner.calls[0]["shared_model"] == "shared-default"
    assert workflow.agent.model == "shared-default"


def test_debate_routes_proposer_and_judge_without_shared_mutation(tmp_path) -> None:
    provider = "deepseek"
    agent_runner = _RecordingAgentRunner()
    workflow = _workflow()
    runner = DebateModeRunner(
        agent_runner,
        proposers=("security",),
        proposer_models={"security": "configured-proposer"},
        judge_model="configured-judge",
        isolation="none",
        provider_router=_router(provider),
    )

    results = asyncio.run(runner.run(_session(tmp_path), workflow))

    assert [result.stage for result in results] == ["security", "judge"]
    assert [call["hooks"]["model_override"] for call in agent_runner.calls] == [
        "router-proposer",
        "router-judge",
    ]
    assert all(provider_name(call["hooks"]["provider_override"]) == provider for call in agent_runner.calls)
    assert all(call["shared_model"] == "shared-default" for call in agent_runner.calls)
    assert workflow.agent.model == "shared-default"


def test_pipeline_builds_router_from_workflow_without_injection(tmp_path) -> None:
    agent_runner = _RecordingAgentRunner()
    workflow = _workflow(
        stage_overrides={
            "analyzer": {"provider": "openrouter", "model": "canonical-model"},
        }
    )
    runner = PipelineModeRunner(
        agent_runner,
        stages=("analyzer",),
        stage_models={"analyzer": "legacy-model"},
    )

    asyncio.run(runner.run(_session(tmp_path), workflow))

    hooks = agent_runner.calls[0]["hooks"]
    assert provider_name(hooks["provider_override"]) == "openrouter"
    assert hooks["model_override"] == "canonical-model"
    assert agent_runner.calls[0]["shared_model"] == "shared-default"
    assert workflow.agent.model == "shared-default"


def test_coordinator_builds_router_for_swarm_stage(tmp_path) -> None:
    agent_runner = _RecordingAgentRunner()
    workflow = _workflow(
        stage_overrides={
            "swarm": {"provider": "openrouter", "model": "swarm-model"},
        }
    )
    runner = CoordinatorModeRunner(agent_runner, route_stage="swarm")
    session = _session(tmp_path)

    asyncio.run(runner.run(session, workflow))

    hooks = agent_runner.calls[0]["hooks"]
    assert provider_name(hooks["provider_override"]) == "openrouter"
    assert hooks["model_override"] == "swarm-model"
    assert workflow.agent.model == "shared-default"


def test_parallel_debate_uses_immutable_per_proposer_route() -> None:
    workflow = _workflow(
        stage_overrides={
            "security": {"provider": "openrouter", "model": "security-model"},
        }
    )
    runner = DebateModeRunner(
        _RecordingAgentRunner(),
        proposers=("security",),
        proposer_models={"security": "legacy-model"},
        isolation="worktree",
        parallel=True,
    )

    hooks = runner._routed_hooks(workflow, "security", {})

    assert provider_name(hooks["provider_override"]) == "openrouter"
    assert hooks["model_override"] == "security-model"
    assert workflow.agent.model == "shared-default"
