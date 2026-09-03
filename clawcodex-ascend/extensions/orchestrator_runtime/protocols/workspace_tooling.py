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

"""WorkspaceTooling protocol.

Workspace metadata and tool-context construction. AgentRunner should
call ``WorkspaceTooling.build_tool_context`` instead of
``ToolContext(...)``, using a minimal structural ``ToolContextLike``.
"""

from __future__ import annotations
# pylint: disable=W2301

from pathlib import Path
from typing import Awaitable, Callable, Protocol, runtime_checkable


@runtime_checkable
class ToolContextLike(Protocol):
    """Minimal structural type — runtime doesn't introspect fields beyond
    :attr:`workspace_root` and :attr:`cwd`.

    Compatible with ``clawcodex_ext.tool_system.context.ToolContext`` and
    with Phase 2's wrapper in
    ``extensions.orchestrator_runtime.utils.git_backend_impl`` if it ever
    needs a tool context (it currently doesn't).

    Phase 3 adds optional fields (``plan_mode``, ``permission_context``)
    when AgentRunner starts building tool contexts.
    """

    workspace_root: Path | None
    cwd: Path | None


@runtime_checkable
class WorkspaceTooling(Protocol):
    """Informs the agent runtime about the workspace being orchestrated.

    The orchestrator's :class:`AgentRunner` calls into the tooling to
    * register custom progress-report tools
    * expose workspace metadata (branch, focus area, rules…)
    """

    def build_tool_context(
        self,
        workspace: Path,
        *,
        branch: str | None = None,
        focus_files: tuple[str, ...] = (),
        rule_hints: tuple[str, ...] = (),
    ) -> ToolContextLike:
        """Return an opaque tool context the runtime passes to tool registry."""
        ...

    def progress_report_callback(self) -> Callable[[str], Awaitable[None]] | None:
        """Hook the agent invokes via the internal ``progress_report`` tool.

        ``None`` if the workspace doesn't support progress reports.
        """
        ...

    def task_update_callback(self) -> Callable[[str, str], Awaitable[None]] | None:
        """Hook the agent invokes via the internal ``task_update`` tool.

        ``None`` if the workspace doesn't support task updates.
        """
        ...


__all__ = ["ToolContextLike", "WorkspaceTooling"]
