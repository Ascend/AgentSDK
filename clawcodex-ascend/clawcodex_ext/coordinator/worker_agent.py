#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""WORKER agent definition — Chunk G / WI-8.5.

Mirrors ``typescript/src/coordinator/workerAgent.ts``. Workers in
coordinator mode are spread from ``GENERAL_PURPOSE_AGENT`` with two
edits:

* ``agent_type`` becomes ``"worker"`` so the coordinator system
  prompt's ``subagent_type: "worker"`` calls resolve correctly.
* ``when_to_use`` is rephrased for the coordinator-mode role.

Workers do NOT lose tools at the agent-definition level — the
``INTERNAL_WORKER_TOOLS`` filter (``coordinator/mode.py``) handles
that at tool-set construction time. Keeping the filter at the
mode-level (rather than baking it into the agent definition) means
the same WORKER definition can be reused for non-coordinator paths
in the future without re-editing.

``get_coordinator_agents()`` returns the canonical agent list for
coordinator-mode sessions (matches ``workerAgent.ts:16-18``).

Originally lived at ``src/coordinator/worker_agent.py``; moved to
``clawcodex_ext/`` per the Decoupling Mandate. The ``src/`` shim is a
``sys.modules`` facade so ``from clawcodex_ext.coordinator.worker_agent import ...``
continues to work for the small number of legacy callers in
``extensions/`` and tests.
"""

from __future__ import annotations

from dataclasses import replace

from clawcodex_ext.agent.agent_definitions import (
    EXPLORE_AGENT,
    GENERAL_PURPOSE_AGENT,
    PLAN_AGENT,
    AgentDefinition,
)


# Spread GENERAL_PURPOSE → tweak agent_type + when_to_use. Mirrors
# ``workerAgent.ts:9-14``.
WORKER_AGENT: AgentDefinition = replace(
    GENERAL_PURPOSE_AGENT,
    agent_type="worker",
    when_to_use=(
        "Worker agent for coordinator mode. Executes tasks autonomously — research, implementation, or verification."
    ),
)


def get_coordinator_agents() -> list[AgentDefinition]:
    """Return the canonical agent list for coordinator-mode sessions.

    Mirrors ``workerAgent.ts:16-18``: ``[WORKER, GENERAL_PURPOSE,
    EXPLORE, PLAN]``. ``WORKER`` first so the coordinator system
    prompt's ``subagent_type: "worker"`` resolves to the right
    definition; ``GENERAL_PURPOSE`` / ``EXPLORE`` / ``PLAN`` follow
    so the legacy spawning paths still work in coordinator mode.
    """
    return [WORKER_AGENT, GENERAL_PURPOSE_AGENT, EXPLORE_AGENT, PLAN_AGENT]


__all__ = [
    "WORKER_AGENT",
    "get_coordinator_agents",
]
