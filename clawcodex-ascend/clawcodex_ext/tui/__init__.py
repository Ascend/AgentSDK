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

"""Downstream TUI extensions — lazy proxy for circular-safety.

Eager imports avoided to break the circular chain:
  __init__ → .app → .screens.* → ..app → ...
"""

# pylint: disable=E0603
__all__ = [
    "ClawCodexTUI",
    "AdvisorEventMessage",
    "AgentRunFinished",
    "AgentRunStarted",
    "AssistantChunk",
    "AssistantMessage",
    "ToolEventMessage",
]

_NAME_TO_MODULE = {
    "ClawCodexTUI": "clawcodex_ext.tui.app",
    "AdvisorEventMessage": "clawcodex_ext.tui.messages",
    "AgentRunFinished": "clawcodex_ext.tui.messages",
    "AgentRunStarted": "clawcodex_ext.tui.messages",
    "AssistantChunk": "clawcodex_ext.tui.messages",
    "AssistantMessage": "clawcodex_ext.tui.messages",
    "ToolEventMessage": "clawcodex_ext.tui.messages",
}


def __getattr__(name: str):
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(module_name)
    val = getattr(mod, name)
    globals()[name] = val
    return val
