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
