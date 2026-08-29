#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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
#

"""PermissionContext Protocol — minimal boundary for SOP permission checks.

Mirrors the field subset the SOP converter (``sop_exploration_guard``)
consumes today; the default adapter wraps
``clawcodex_ext.permissions.types.ToolPermissionContext``. Field aliases
per ``docs/DECOUPLE_SOP_CONVERTER_PLAN.md`` §3.3: ``mode``,
``is_bypass`` (upstream ``is_bypass_permissions_mode_available``),
``should_avoid_prompts`` (upstream ``should_avoid_permission_prompts``),
``blocks(tool_name)`` predicate.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["PermissionContextProtocol"]


PermissionModeLiteral = str


@runtime_checkable
class PermissionContextProtocol(Protocol):
    """Minimal contract for an SOP-visible permission context.

    Implementations MUST expose ``mode`` (permission mode literal),
    ``is_bypass`` (bypassPermissions selectable), ``should_avoid_prompts``
    (no interactive prompts), and ``blocks(tool_name)`` (always-deny check).
    """

    mode: PermissionModeLiteral
    is_bypass: bool
    should_avoid_prompts: bool

    def blocks(self, tool_name: str) -> bool: ...
