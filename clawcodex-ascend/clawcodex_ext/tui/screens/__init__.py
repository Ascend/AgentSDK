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

"""Textual screens for the Claw Codex TUI — lazy proxy.

All screen modules are in ``clawcodex_ext.tui.screens.*``.
Eager imports are avoided here to break the circular chain:
  screens/__init__ → .ask_user_question → ..app → ..screens → ...
"""

# Screen module name prefix
_PREFIX = "clawcodex_ext.tui.screens."

# Map each exported name to its source module
_NAME_TO_MODULE = {
    "AskUserQuestionModal": "ask_user_question",
    "CostThresholdScreen": "cost_threshold",
    "DialogScreen": "dialog_base",
    "DiffDialogScreen": "diff_dialog",
    "FileDiff": "diff_dialog",
    "EffortPickerScreen": "effort_picker",
    "ForecastPickerScreen": "forecast_picker",
    "GenericInputScreen": "generic_input",
    "GenericSelectScreen": "generic_select",
    "ExitFlowScreen": "exit_flow",
    "HistoryEntry": "history_search",
    "HistorySearchScreen": "history_search",
    "IdleReturnScreen": "idle_return",
    "McpElicitationScreen": "mcp_dialogs",
    "McpListScreen": "mcp_dialogs",
    "McpServer": "mcp_dialogs",
    "McpToolListScreen": "mcp_dialogs",
    "MessageSelectorScreen": "message_selector",
    "ModelPickerScreen": "model_picker",
    "MonitorPanel": "monitor_panel",
    "PermissionModal": "permission_modal",
    "PermissionModePickerScreen": "permission_mode_picker",
    "REPLScreen": "repl",
    "ThemePickerScreen": "theme_picker",
    "TranscriptMessage": "message_selector",
    "fuzzy_score": "history_search",
}

__all__ = sorted(_NAME_TO_MODULE.keys())


def __getattr__(name: str):
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(_PREFIX + module_name)
    val = getattr(mod, name)
    globals()[name] = val
    return val
