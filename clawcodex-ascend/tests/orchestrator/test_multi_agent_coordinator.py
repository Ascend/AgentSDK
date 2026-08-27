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

"""Focused tests for per-session coordinator mode isolation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from extensions.orchestrator.multi_agent.modes.coordinator import CoordinatorModeRunner


@pytest.mark.asyncio
async def test_coordinator_mode_is_restored_after_run() -> None:
    agent_runner = SimpleNamespace(run=AsyncMock(return_value="done"))
    session = SimpleNamespace(issue=SimpleNamespace(id="42"), coordinator_mode=False)

    result = await CoordinatorModeRunner(agent_runner).run(session, object())

    assert result == "done"
    assert session.coordinator_mode is False


@pytest.mark.asyncio
async def test_temporary_attribute_is_removed() -> None:
    agent_runner = SimpleNamespace(run=AsyncMock(return_value=None))
    session = SimpleNamespace(issue=SimpleNamespace(id="42"))

    await CoordinatorModeRunner(agent_runner).run(session, object())

    assert not hasattr(session, "coordinator_mode")


@pytest.mark.asyncio
@pytest.mark.parametrize("has_original", [False, True])
async def test_coordinator_mode_is_restored_when_run_raises(has_original: bool) -> None:
    agent_runner = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("boom")))
    session = SimpleNamespace(issue=SimpleNamespace(id="42"))
    if has_original:
        session.coordinator_mode = False

    with pytest.raises(RuntimeError, match="boom"):
        await CoordinatorModeRunner(agent_runner).run(session, object())

    if has_original:
        assert session.coordinator_mode is False
    else:
        assert not hasattr(session, "coordinator_mode")
