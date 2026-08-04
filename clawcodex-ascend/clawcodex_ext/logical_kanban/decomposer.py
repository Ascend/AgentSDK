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

"""Compatibility shim — delegate to lkb.decomposer with ToolContext boundary adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lkb.decomposer import (  # noqa: F401
    DecompositionPlan,
    ProposedTask,
    TaskDecomposer as _LkbTaskDecomposer,
    TaskDecompositionError,
    _collect_method_references,
    _count_method_task_usage,
    _LKB_METADATA_KEYS,
)
from lkb.types import LkbValidationContext

if TYPE_CHECKING:
    from clawcodex_ext.tool_system.context import ToolContext


def _to_clawcodex_ctx(lkb_ctx: LkbValidationContext) -> ToolContext:
    """Convert LkbValidationContext → clawcodex ToolContext (boundary adapter)."""
    from clawcodex_ext.tool_system.context import ToolContext as _ClawContext

    ctx = _ClawContext(workspace_root=lkb_ctx.workspace_root)
    ctx.session_id = lkb_ctx.session_id
    ctx.tasks.update(lkb_ctx.tasks)
    return ctx


class TaskDecomposer(_LkbTaskDecomposer):
    """Subclass that overrides _build_validation_context to emit ToolContext."""

    @staticmethod
    def _build_validation_context(
        tasks: tuple[ProposedTask, ...],
        existing_tasks: tuple[dict[str, Any], ...],
    ) -> ToolContext:
        lkb_ctx = _LkbTaskDecomposer._build_validation_context(tasks, existing_tasks)
        return _to_clawcodex_ctx(lkb_ctx)
