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

"""Coordinator/Worker mode — one coordinator agent fans out to workers.

Mechanism
---------

The coordinator/worker machinery is already wired in the runtime:

* ``clawcodex_ext/coordinator/mode.py`` defines the restricted coordinator
  tool set and the worker tool set.
* ``clawcodex_ext/entrypoints/headless.py`` reads
  ``CLAUDE_CODE_COORDINATOR_MODE`` at agent startup and, if truthy,
  filters the tool registry down to the 6-tool coordinator set.
* ``AgentRunner.run`` toggles that env var based on
  ``agent_config.coordinator_mode`` (a per-workflow boolean).

What this runner adds
---------------------

It lets the orchestrator turn coordinator mode ON for a **single issue**
without flipping the workflow-wide config. The runner:

1. Caches the current value of ``agent_config.coordinator_mode``.
2. Sets it to ``True`` just before delegating to ``AgentRunner.run``.
3. Restores the original value in a ``finally`` block — even if the
   underlying run raises — so the next issue isn't accidentally pinned
   into coordinator mode.

This makes ``mode:coordinator`` a per-issue decision routed by
``ModeSelector`` instead of a global workflow.md commitment.

Limitations / honest scope
--------------------------

This Phase-3 runner does **not** spawn TeamCreate / SendMessage scaffolding
on the operator's behalf — that's the agent's job once it's running in
coordinator mode. What we guarantee is: when ``ModeSelector`` picks
``coordinator``, the agent for that issue boots with the coordinator
tool set + ``tool_context.team`` populated (when ``.clawcodex/team.json``
exists in the workspace + ``CLAUDE_CODE_AGENT_NAME`` is set).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agent_runner import AgentRunner, AgentSession
    from ..config.schema import WorkflowConfig

logger = logging.getLogger(__name__)


class CoordinatorModeRunner:
    """Run one issue with coordinator tool filtering enabled."""

    def __init__(self, agent_runner: "AgentRunner") -> None:
        self._agent_runner = agent_runner

    async def run(
        self,
        session: "AgentSession",
        workflow: "WorkflowConfig",
        **hooks: Any,
    ) -> Any:
        sentinel = object()
        original = getattr(session, "coordinator_mode", sentinel)
        logger.info(
            "CoordinatorModeRunner: enabling coordinator_mode for issue=%s (was=%s)",
            session.issue.id,
            None if original is sentinel else original,
        )
        try:
            session.coordinator_mode = True
            return await self._agent_runner.run(session, workflow, **hooks)
        finally:
            if original is sentinel:
                delattr(session, "coordinator_mode")
            else:
                session.coordinator_mode = original
            logger.info(
                "CoordinatorModeRunner: restored session coordinator_mode=%s after issue=%s",
                None if original is sentinel else original,
                session.issue.id,
            )


__all__ = ["CoordinatorModeRunner"]
