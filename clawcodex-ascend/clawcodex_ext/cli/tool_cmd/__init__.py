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

"""expose tools automatically as CLI slash commands.

Public API
----------
* :class:`DynamicCommandDiscovery` — scan a ``ToolRegistry`` for non-core
 tools and produce ``LocalCommand``s.
* :class:`DynamicToolCommand` — single-tool adapter; bind to REPL/TUI
 ``CommandContext`` for invocation.
* :func:`install_tool_subcommand` — register the ``clawcodex-dev tool``
 CLI subcommand (idempotent).
* :func:`register_tool_commands` — register per-tool slash commands in
 a REPL/TUI ``CommandRegistry``.

Usage
-----
At REPL/TUI startup (after the runtime context is built)::

 from clawcodex_ext.cli.tool_cmd import register_tool_commands
 from clawcodex_ext.command_system.registry import CommandRegistry

 command_registry = CommandRegistry()
 register_tool_commands(command_registry, ctx.tool_registry)

From the CLI::

 clawcodex-dev tool --list
 clawcodex-dev tool detect_modality --path /data/sample.mp4
"""

from __future__ import annotations

from .command import DynamicToolCommand
from .discovery import DynamicCommandDiscovery
from .hooks import install_tool_subcommand, register_tool_commands
from . import core_filter, schema_parser

__all__ = [
    "DynamicCommandDiscovery",
    "DynamicToolCommand",
    "core_filter",
    "install_tool_subcommand",
    "register_tool_commands",
    "schema_parser",
]
