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
"""ModeRunner Protocol + ModeDecision dataclass.

Why a Protocol instead of an ABC
--------------------------------

Phase 1 deliberately matches the existing ``AgentRunner.run`` shape
duck-typed in ``orchestrator.py:1579``: any object exposing an awaitable
``run(session, workflow, **hooks)`` works there. Keeping ``ModeRunner``
as a ``Protocol`` means ``SingleModeRunner`` can be a one-line wrapper
and the orchestrator's call site stays unchanged — we just dispatch
through ``modes.get(mode_key)`` first instead of taking
``self.agent_runner`` directly.

Future modes (Pipeline / Coordinator / Debate) implement the same
``run(...)`` signature but internally orchestrate multiple agent runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from extensions.orchestrator.agent_runner import AgentSession
    from extensions.orchestrator.config.schema import WorkflowConfig


DEFAULT_MODE: str = "single"
"""Fallback mode used when ModeSelector fails or returns unknown."""


@dataclass
class ModeDecision:
    """The choice ``ModeSelector.choose`` returns for one issue.

    Attributes
    ----------
    mode
        The collaboration mode key (e.g. ``"single"``, ``"pipeline"``,
        ``"coordinator"``, ``"debate"``).
    reason
        Human-readable explanation of why this mode was picked. Persisted
        to ``IssueRecord.mode_decision_reason`` so operators can audit
        router decisions later.
    source
        Where the decision came from:
        ``"label"`` — explicit ``mode:*`` label on the issue
        ``"router"`` — LLM router agent picked it
        ``"fallback"`` — selector failed; using ``DEFAULT_MODE``
        ``"config"`` — workflow.md forced a mode for all issues
    agents
        Optional roster of role names the mode expects (used by Pipeline
        and Debate). Phase-1 leaves this empty for ``single``.
    confidence
        Router's self-reported confidence (``0.0``–``1.0``). Phase-1
        modes ignore this; future code may threshold on it.
    """

    mode: str = DEFAULT_MODE
    reason: str = ""
    source: str = "fallback"
    agents: list[str] = field(default_factory=list)
    confidence: float = 1.0


@runtime_checkable
class ModeRunner(Protocol):
    """Anything the orchestrator can call ``await runner.run(...)`` on.

    The signature mirrors ``AgentRunner.run`` exactly so Phase-1's
    ``SingleModeRunner`` is a literal pass-through. Future modes accept
    the same kwargs and orchestrate multiple internal agent runs.
    """

    async def run(
        self,
        session: "AgentSession",
        workflow: "WorkflowConfig",
        **hooks: Any,
    ) -> Any: ...


# Re-exports used by callers that import from .base directly.
__all__ = ["DEFAULT_MODE", "ModeDecision", "ModeRunner"]
