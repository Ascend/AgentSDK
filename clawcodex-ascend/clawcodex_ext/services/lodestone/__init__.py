#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""F-97 LODESTONE: unified deep-link anchor layer for ClawCodex.

LODESTONE recognises ``path:line[:column]`` / git refs / tracker issues /
function symbols that appear in any agent output, REPL panel or markdown
document and converts them into navigable URLs targeting the user's IDE,
the git remote, or the configured tracker.

This package is the core (Layer 1) implementation. Public surface:

- ``models``     : data classes (LodestoneAnchor / AnchorTarget / context)
- ``parser``     : tokenise text and classify anchors
- ``targets``    : built-in target registry (vscode / cursor / idea / …)
- ``resolver``   : pick the right target given context + user preference
- ``renderer``   : text / markdown / OSC 8 rendering
- ``fingerprint``: workspace detection (git remote, host platform)
- ``config``     : default config + persistence
- ``service``    : one-stop ``LodestoneService`` facade

Concrete consumers: ``clawcodex_ext.command_system.lodestone_commands``
provides ``/link`` sub-commands; ``clawcodex_ext.tool_system.tools.lodestone``
exposes a ``LodestoneTool`` for agent-driven rendering.
"""

from __future__ import annotations

from .models import (
    AnchorContext,
    AnchorKind,
    AnchorTarget,
    AnchorTargetRegistry,
    LodestoneAnchor,
    LodestoneConfig,
    RenderedAnchor,
    Sink,
)
from .service import LodestoneService, get_lodestone_service, reset_default_service

__all__ = [
    "AnchorContext",
    "AnchorKind",
    "AnchorTarget",
    "AnchorTargetRegistry",
    "LodestoneAnchor",
    "LodestoneConfig",
    "LodestoneService",
    "RenderedAnchor",
    "Sink",
    "get_lodestone_service",
    "reset_default_service",
]
