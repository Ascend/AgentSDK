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

"""Focused constructor tests for pipeline coordination."""

from types import SimpleNamespace

import pytest

from extensions.orchestrator.multi_agent.modes.pipeline import PipelineModeRunner
from extensions.orchestrator.provider_routing import provider_name


def test_pipeline_normalizes_stage_overrides() -> None:
    runner = PipelineModeRunner(
        object(),
        stages=("analyze", "implement"),
        stage_models={"analyze": "  model-a  ", "implement": ""},
        stage_max_turns={"analyze": 4, "implement": 0},
    )

    assert runner.stages == ("analyze", "implement")
    assert runner.stage_models == {"analyze": "model-a"}
    assert runner.stage_max_turns == {"analyze": 4}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stages": ()}, "at least one stage"),
        ({"handoff": "invalid"}, "handoff"),
        ({"max_retries_per_stage": -1}, "must be >= 0"),
    ],
)
def test_pipeline_rejects_invalid_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        PipelineModeRunner(object(), **kwargs)


@pytest.mark.asyncio
async def test_stage_overrides_do_not_mutate_shared_runtime(tmp_path) -> None:
    agent_config = SimpleNamespace(model="default", provider="deepseek", stage_overrides={})

    class RecordingRunner:
        def __init__(self) -> None:
            self.agent_config = agent_config
            self.max_turns = 20

        async def run(self, session, workflow, **hooks) -> None:
            session.seen_runtime = (
                self.agent_config.model,
                self.max_turns,
                workflow.agent.model,
                provider_name(hooks["provider_override"]),
                hooks["model_override"],
            )
            session.status = "completed"

    shared_runner = RecordingRunner()
    shared_workflow = SimpleNamespace(agent=agent_config)
    session = SimpleNamespace(
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
    pipeline = PipelineModeRunner(
        shared_runner,
        stages=("analyzer",),
        stage_models={"analyzer": "stage-model"},
        stage_max_turns={"analyzer": 4},
    )

    await pipeline.run(session, shared_workflow)

    assert session.seen_runtime == ("default", 4, "default", "deepseek", "stage-model")
    assert shared_runner.agent_config is agent_config
    assert shared_runner.max_turns == 20
    assert shared_workflow.agent.model == "default"
