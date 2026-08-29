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

"""Macro definition and routing data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MacroRoute:
    """Direct route for macro tool recall.

    MacroRoute is independent of PriorityRoute. It can actively recall
    a target tool that would not otherwise appear in normal ToolSearch results.
    """

    phrases: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    negative_keywords: list[str] = field(default_factory=list)
    target_tool: str = ""
    match_mode: Literal["exact", "all", "any"] = "all"
    selection: Literal["exclusive", "prefer"] = "prefer"
    priority: int = 100
    verified: bool = False
    enabled: bool = True
    # Narrow retrieval intent and the atomic tools shadowed by this
    # macro.  These are deliberately separate from lifecycle groups,
    # which may contain several actions (for example create + invoke).
    intent_key: str = ""
    covered_tools: list[str] = field(default_factory=list)
    unavailable_policy: Literal["restore-covered"] = "restore-covered"
    # Resolution order: session > bundle > builtin (builtin safety exclusives protected)
    scope: Literal["session", "bundle", "builtin"] = "bundle"


@dataclass
class MacroDefinition:
    """Persistent macro definition with routing and provenance."""

    version: int = 1
    name: str = ""
    description: str = ""
    scope: Literal["bundle", "session", "builtin"] = "bundle"
    enabled: bool = True
    workflow: dict = field(default_factory=dict)
    routing: MacroRoute = field(default_factory=MacroRoute)
    provenance: dict = field(default_factory=dict)
