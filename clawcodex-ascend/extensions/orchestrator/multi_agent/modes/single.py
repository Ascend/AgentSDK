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
"""Single-agent mode — pass-through wrapper over the existing AgentRunner.

This is Phase-1's only registered mode and the safe fallback for every
later phase. Behavior must be byte-identical to calling
``self.agent_runner.run(session, workflow, ...)`` directly so the 270+
existing orchestrator tests keep passing without modification.

There is intentionally **no** branching here: ``run`` simply delegates.
If you find yourself adding logic to this class, you probably want a
new mode (Pipeline / Coordinator / Debate) instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from extensions.orchestrator.agent_runner import AgentRunner, AgentSession
    from extensions.orchestrator.config.schema import WorkflowConfig


class SingleModeRunner:
    """Wraps an ``AgentRunner`` to satisfy the ``ModeRunner`` Protocol."""

    def __init__(self, agent_runner: "AgentRunner") -> None:
        self._agent_runner = agent_runner

    async def run(
        self,
        session: "AgentSession",
        workflow: "WorkflowConfig",
        **hooks: Any,
    ) -> Any:
        return await self._agent_runner.run(session, workflow, **hooks)


__all__ = ["SingleModeRunner"]
