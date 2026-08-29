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

"""Monitor extension installation hooks.

Called once per process from ``clawcodex_ext.ensure_eager_extensions_installed``
after all ``src/`` modules are loaded.  Command and tool registration happen
statically via ``clawcodex_ext.command_system.builtins`` and
``clawcodex_ext.tool_system.tools``; this module is responsible for runtime
wiring that must happen after upstream is initialised (e.g. TUI keybindings,
stall-watchdog exemption hooks).
"""

from __future__ import annotations

_installed: bool = False


def install_monitor_extensions() -> None:
    """Install runtime extensions once per process.

    Idempotent no-op when the feature gate is disabled; the command/tool
    ``is_enabled`` predicates already gate user-visible behaviour.
    """
    global _installed
    if _installed:
        return
    _installed = True

    # Future integration points:
    # 1. Stall-watchdog exemption: when a stall detector is introduced for
    #    background bash tasks, it should consult
    #    ``StallWatchdogExemptor.should_skip_stall_check(state)`` for
    #    ``kind='monitor'`` entries.
    # 2. TUI Shift+Down keybinding: when the monitor panel is loaded, its
    #    binding is injected here.


__all__ = ["install_monitor_extensions"]
