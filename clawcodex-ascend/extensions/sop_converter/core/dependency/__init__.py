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

"""Tool Dependency Graph.

Builds a directed graph of tool lifecycle dependencies from a parsed
``list[SourceComponent]`` and persists it to a bundle-local
``tool-dependencies.yaml`` so downstream consumers (task guide, system
prompts, ToolSearch ranker) can reason about create->invoke chains
without re-deriving the heuristics at runtime.

Layering
--------
* ``models``     - pure data classes
* ``heuristics``  - pair / shared-param rules
* ``detector``    - ``detect_lifecycle_patterns()`` entry point
* ``writer``      - YAML writer (with a built-in YAML-subset fallback when PyYAML missing)
* ``reader``      - load + tolerant corruption handling
"""

from __future__ import annotations

from .detector import detect_lifecycle_patterns
from .models import (
    HiddenStep,
    IntentGroup,
    PriorityRoute,
    ToolDependency,
    ToolDependencyGraph,
)
from .reader import load_tool_dependencies, merge_overrides
from .writer import write_tool_dependencies

__all__ = [
    "HiddenStep",
    "IntentGroup",
    "PriorityRoute",
    "ToolDependency",
    "ToolDependencyGraph",
    "detect_lifecycle_patterns",
    "load_tool_dependencies",
    "merge_overrides",
    "write_tool_dependencies",
]
