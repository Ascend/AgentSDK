#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""Third-party agent extension helpers.

A third-party extension is any subdirectory of ``extensions/`` that
contains an ``agents/`` subdirectory. ``*.md`` files inside ``agents/``
are picked up by
:func:`clawcodex_ext.agent.markdown_discovery.discover_extension_agents`
and registered as ``source="extensions"`` agents, taking priority over
``clawcodex_ext`` but yielding to user / project / managed overrides.

For programmatic registration, import :func:`register` from this module;
it defaults registrations to ``source="extensions"``. Extensions can also
use :class:`clawcodex_ext.agent.registry.AgentRegistry` directly when they
need full control.
"""

from __future__ import annotations

from typing import Callable

from clawcodex_ext.agent.registry import AgentRegistry, SOURCE_EXTENSIONS  # pylint: disable=import-error


def register(
    agent_type: str,
    *,
    when_to_use: str,
    tools: list[str] | None = None,
    disallowed_tools: list[str] | None = None,
    model: str | None = None,
    permission_mode: str | None = None,
    max_turns: int | None = None,
    background: bool = False,
    color: str | None = None,
    memory: str | None = None,
    omit_claude_md: bool = False,
    skills: list[str] | None = None,
    isolation: str | None = None,
    base_dir: str = "extensions",
) -> Callable[[Callable[..., str]], object]:
    return AgentRegistry.register(
        agent_type,
        when_to_use=when_to_use,
        tools=tools,
        disallowed_tools=disallowed_tools,
        model=model,
        permission_mode=permission_mode,
        max_turns=max_turns,
        background=background,
        color=color,
        memory=memory,
        omit_claude_md=omit_claude_md,
        skills=skills,
        isolation=isolation,
        source=SOURCE_EXTENSIONS,
        base_dir=base_dir,
    )


__all__ = ["register"]
