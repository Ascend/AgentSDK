#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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
