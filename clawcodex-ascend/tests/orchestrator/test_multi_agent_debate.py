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

"""Focused configuration test for debate coordination."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from extensions.orchestrator.multi_agent.modes.debate import DebateModeRunner
from extensions.orchestrator.provider_routing import provider_name


def test_debate_exposes_configured_roles() -> None:
    runner = DebateModeRunner(object(), proposers=("security", "maintainer"))
    assert runner.proposers == ("security", "maintainer")
    assert runner.judge_mode == "pick"


def _session(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        issue=SimpleNamespace(id="42", title="", description=""),
        workspace=SimpleNamespace(path=tmp_path),
        turn_count=0,
        status="running",
        output_text="",
        session_end_reason=None,
        session_end_summary="",
        run_id=None,
        consecutive_429_count=0,
        rate_limit_pending_turn=None,
        prompt_override=None,
        run_kind="",
    )


@pytest.mark.asyncio
async def test_debate_model_selection_does_not_mutate_shared_config(tmp_path) -> None:
    agent_config = SimpleNamespace(model="default", provider="deepseek", stage_overrides={})
    calls: list[tuple[str, str, str, str, str]] = []

    class RecordingRunner:
        def __init__(self) -> None:
            self.agent_config = agent_config

        async def run(self, session, workflow, **hooks) -> None:
            calls.append(
                (
                    session.run_kind,
                    self.agent_config.model,
                    workflow.agent.model,
                    provider_name(hooks["provider_override"]),
                    hooks["model_override"],
                )
            )
            session.status = "completed"
            session.output_text = "done"

    shared_runner = RecordingRunner()
    workflow = SimpleNamespace(agent=agent_config)
    runner = DebateModeRunner(
        shared_runner,
        proposers=("proposal",),
        proposer_models={"proposal": "proposal-model"},
        judge_model="judge-model",
        isolation="none",
    )

    await runner.run(_session(tmp_path), workflow)

    assert calls == [
        ("debate:proposal", "default", "default", "deepseek", "proposal-model"),
        ("debate:judge", "default", "default", "deepseek", "judge-model"),
    ]
    assert shared_runner.agent_config is agent_config
    assert workflow.agent.model == "default"


@pytest.mark.asyncio
async def test_parallel_proposer_failure_updates_original_session(tmp_path) -> None:
    runner = DebateModeRunner(object(), parallel=True, isolation="worktree")
    runner._run_proposers_parallel = AsyncMock(  # type: ignore[method-assign]
        return_value=[SimpleNamespace(stage="proposal", status="failed", output="boom")]
    )
    session = _session(tmp_path)

    results = await runner.run(session, SimpleNamespace(agent=SimpleNamespace(model="default")))

    assert results[0].status == "failed"
    assert session.status == "failed"
    assert session.session_end_reason == "debate_proposer_failed"
