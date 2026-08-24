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
#
"""Team-only tool visibility filter.

When ``ToolContext.team`` is not an active team (i.e. the session has
no team context), drop tools that can only do useful work inside an
existing team: ``SendMessage`` (inter-agent messaging / broadcast) and
``TeamDelete``. ``TeamCreate`` stays visible as the bootstrap entry
that can create the first team context. The model's self-introduction
in single-user REPL sessions used to mis-call ``SendMessage { to: "*" }``
as a way of "replying"; hiding that tool from the API schemas removes
the path entirely.

This module is pure (no I/O, no globals, no upstream-tool mutation)
so it can be unit-tested in isolation and reused anywhere upstream
or downstream assembles a tool list.
"""

from __future__ import annotations

from typing import Iterable, TypeVar

# Tools that require an already-active team context. We intentionally do
# not mirror coordinator ``INTERNAL_WORKER_TOOLS`` exactly: ``TeamCreate``
# is a bootstrap tool and must remain visible before a team exists.
TEAM_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "SendMessage",
        "TeamDelete",
    }
)


_T = TypeVar("_T")


def has_team_context(team: object) -> bool:
    """Return True iff a team context is active.

    A team context is active when ``team`` is a non-empty dict
    (the shape set by ``src.tool_system.tools.team.TeamCreate`` at
    ``context.team = team_dict``). Anything else — ``None``, empty
    dict, malformed value — is treated as "no team".
    """
    return isinstance(team, dict) and bool(team)


def filter_team_only_tools(
    tools: Iterable[_T],
    has_team: bool,
    *,
    name_attr: str = "name",
) -> list[_T]:
    """Return ``tools`` minus any team-only entries when no team is active.

    Args:
        tools: An iterable of tool-like objects. Each must expose a
            ``name`` attribute (or whatever ``name_attr`` points to).
        has_team: Result of :func:`has_team_context` on the relevant
            ``ToolContext.team``. ``True`` short-circuits — all tools
            pass through.
        name_attr: Attribute name used to read the tool's display
            name. Defaults to ``"name"`` to match the ``Tool`` dataclass.

    Returns:
        A list copy of ``tools`` with team-only entries removed when
        ``has_team`` is False; the original elements (same order) when
        ``has_team`` is True.
    """
    if has_team:
        return list(tools)
    return [t for t in tools if getattr(t, name_attr, None) not in TEAM_ONLY_TOOL_NAMES]


__all__ = [
    "TEAM_ONLY_TOOL_NAMES",
    "has_team_context",
    "filter_team_only_tools",
]
