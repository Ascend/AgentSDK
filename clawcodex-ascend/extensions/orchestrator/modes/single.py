#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

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
    from ..agent_runner import AgentRunner, AgentSession
    from ..config.schema import WorkflowConfig


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
