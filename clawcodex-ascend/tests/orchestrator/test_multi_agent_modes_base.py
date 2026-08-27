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

"""Focused tests for the multi-agent mode contract and registry."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from extensions.orchestrator.multi_agent import modes
from extensions.orchestrator.multi_agent.modes.base import ModeDecision, ModeRunner
from extensions.orchestrator.multi_agent.modes.single import SingleModeRunner


@pytest.mark.asyncio
async def test_single_mode_is_transparent() -> None:
    agent_runner = SimpleNamespace(run=AsyncMock(return_value="done"))
    runner = SingleModeRunner(agent_runner)
    session = object()
    workflow = object()

    assert await runner.run(session, workflow, tracker="tracker") == "done"
    agent_runner.run.assert_awaited_once_with(session, workflow, tracker="tracker")
    assert isinstance(runner, ModeRunner)


def test_registry_and_default_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modes, "_registry", {})
    runner = SingleModeRunner(SimpleNamespace(run=AsyncMock()))
    modes.register("single-test", runner)

    assert modes.get("single-test") is runner
    assert "single-test" in modes.available()
    assert ModeDecision().mode == "single"


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_mode_decision_rejects_out_of_range_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        ModeDecision(confidence=confidence)
