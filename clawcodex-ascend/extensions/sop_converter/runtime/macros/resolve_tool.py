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

"""Unified tool lookup: session overlay > registry > options.tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from clawcodex_ext.tool_system.build_tool import Tool, tool_matches_name

from .session import is_session_macro_tool

if TYPE_CHECKING:
    from clawcodex_ext.tool_system.context import ToolContext
    from clawcodex_ext.tool_system.registry import ToolRegistry


def _owner_matches(context: ToolContext, owner_session_id: str) -> bool:
    return bool(context.session_id) and context.session_id == owner_session_id


def _find_in_options(
    tools: list[Tool],
    name: str,
    *,
    skip_session_macros: bool,
) -> Tool | None:
    for tool in tools:
        if skip_session_macros and is_session_macro_tool(tool):
            continue
        if tool_matches_name(tool, name):
            return tool
    return None


def resolve_tool_for_context(
    context: ToolContext,
    name: str,
    *,
    base_registry: ToolRegistry | None = None,
) -> Tool | None:
    """Resolve a tool preferring a bound session overlay snapshot.

    Order:
    1. If overlay snapshot owner matches ``context.session_id``, look up
       ``snapshot.tools[name]`` (lower-key map, same as ToolRegistry).
    2. Else ``base_registry.get(name)`` when provided.
    3. Else ``options.tools``, skipping stale session-provenance tools when
       the overlay owner does not match the current session.
    """
    overlay = getattr(context, "session_macro_overlay", None)
    snapshot = overlay.read() if overlay is not None else None
    owner_ok = False
    if snapshot is not None:
        owner_ok = _owner_matches(context, snapshot.owner_session_id)
        if owner_ok:
            found = snapshot.tools.get(name.lower())
            if found is not None:
                return found

    if base_registry is not None:
        found = base_registry.get(name)
        if found is not None:
            return found

    options = getattr(context, "options", None)
    tools = list(getattr(options, "tools", None) or [])
    skip_stale = snapshot is not None and not owner_ok
    return _find_in_options(tools, name, skip_session_macros=skip_stale)
