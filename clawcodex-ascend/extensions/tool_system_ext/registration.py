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
# pylint: disable=ungrouped-imports

"""Extension tool registration.

Registers 二开 tools that are not part of upstream's ALL_STATIC_TOOLS.
Called by ``src/tool_system/defaults.py:build_default_registry``.
"""

from __future__ import annotations

from clawcodex_ext.goal.tools import make_goal_model_tools
from clawcodex_ext.tool_system.build_tool import Tool
from clawcodex_ext.tool_system.tools.bg_session import BgSessionTool
from clawcodex_ext.tool_system.tools.create_agent_tool import make_create_agent_tool
from clawcodex_ext.tool_system.tools.lodestone import LodestoneTool
from clawcodex_ext.tool_system.tools.progress_report import ProgressReportTool
from clawcodex_ext.tool_system.tools.task_directives import TaskDirectivesTool
from clawcodex_ext.tool_system.tools.task_inspect import TaskInspectTool
from extensions.sop_converter.runtime.macros.register_tool import (
    PromoteMacroWorkflowTool,
    RegisterMacroFromTraceTool,
    RegisterMacroWorkflowTool,
)

EXTENSION_TOOLS: list[Tool] = [
    *make_goal_model_tools(),
    ProgressReportTool,
    TaskDirectivesTool,
    TaskInspectTool,
    make_create_agent_tool(),
    # Agent-facing background session query/control.
    # Tool self-gates on CLAWCODEX_BG_SESSIONS (returns {disabled: true}
    # when off), so unconditional registration is safe.
    BgSessionTool,
    # Deep-link anchor parser + resolver.
    # Self-gates on ``LODESTONE=off`` (renderer falls back to plain
    # text), so unconditional registration is safe.
    LodestoneTool,
    # Session macro register / from-trace / promote
    # (capability-gated; confirm wired by TUI/REPL).
    RegisterMacroWorkflowTool,
    RegisterMacroFromTraceTool,
    PromoteMacroWorkflowTool,
]

# Chrome browser automation — seven ``chrome_*`` tools
# registered when the optional chrome service module is importable.
# The chrome controller depends on Playwright / Pillow / the MCP
# SDK as optional dependencies; the tools themselves are built
# unconditionally and degrade gracefully to a ``NullChromeController``
# that surfaces an install-hint error on every call.
try:
    from clawcodex_ext.services.chrome import build_chrome_tools

    EXTENSION_TOOLS.extend(build_chrome_tools())
except Exception:  # noqa: BLE001 — defensive, never break tool registration  # nosec B110
    # If the chrome module can't be imported (e.g. mid-refactor
    # or a partial install), the rest of the extension tools
    # still register. The chrome public API is loaded lazily by
    # ``build_chrome_controller`` at first use.
    pass

__all__ = [
    "EXTENSION_TOOLS",
]
