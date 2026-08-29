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

"""DynamicCommandDiscovery — scan of ``ToolRegistry`` for non-core tools.

The discovery walks the supplied ``ToolRegistry``, filters out core /
built-in tools (via :mod:`clawcodex_ext.cli.tool_cmd.core_filter`), and
wraps each remaining tool in a :class:`DynamicToolCommand`. The result
is an iterable of :class:`LocalCommand` objects ready to be registered in a
:class:`CommandRegistry`.

Two scan modes
--------------
1. **Eager** (:meth:`discover`): one-shot scan at startup; returns the
   full command list. Cheap (one ``list_tools()`` call + O(n) wrapping).
2. **Incremental** (:meth:`rediscover`): scan a fresh registry against
   the previously-seen set; return only newly-registered tools. Useful
   for resume / hot-swap scenarios where the registry mutates after
   startup.

Conflict policy
---------------
If a discovered tool's name collides with an already-registered
command (e.g. a dynamic command or a skill command), the
discovery skips it. The core set itself is filtered *before* this
check, so collisions with built-ins like ``/read`` cannot happen.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from . import command as _command_mod
from . import core_filter

if TYPE_CHECKING:
    from clawcodex_ext.command_system.types import LocalCommand
    from clawcodex_ext.tool_system.build_tool import Tool
    from clawcodex_ext.tool_system.registry import ToolRegistry

log = logging.getLogger(__name__)


class DynamicCommandDiscovery:
    """Scan a ``ToolRegistry`` and produce ``LocalCommand``s for non-core tools."""

    def __init__(self, tool_registry: "ToolRegistry") -> None:
        self.registry = tool_registry
        # Name → DynamicToolCommand. Acts as a dedup index so re-discoveries
        # don't re-wrap the same tool and produce distinct command objects.
        self._seen: dict[str, _command_mod.DynamicToolCommand] = {}

    # --- Public API ----------------------------------------------------

    def discover(self) -> list["LocalCommand"]:
        """Return commands for all non-core tools in the registry.

        Order is preserved from ``ToolRegistry.list_tools()`` (which is
        insertion order in practice). Returns a list of
        :class:`LocalCommand`; caller is responsible for registering them.
        """
        commands: list[LocalCommand] = []
        for tool in self.registry.list_tools():
            if not _is_discoverable(tool):
                continue
            cmd = self._wrap(tool)
            if cmd is not None:
                commands.append(cmd.local_command)
        return commands

    def discover_commands(self) -> list["_command_mod.DynamicToolCommand"]:
        """Return the underlying :class:`DynamicToolCommand` objects (with
        full accessor methods, useful for tests).
        """
        commands: list[_command_mod.DynamicToolCommand] = []
        for tool in self.registry.list_tools():
            if not _is_discoverable(tool):
                continue
            cmd = self._wrap(tool)
            if cmd is not None:
                commands.append(cmd)
        return commands

    def rediscover(self) -> list["LocalCommand"]:
        """Re-scan and return only NEW commands not seen in previous calls.

        Useful for resume / hot-swap flows where the registry mutates
        after the REPL/TUI has already started.
        """
        new_commands: list[LocalCommand] = []
        for tool in self.registry.list_tools():
            if not _is_discoverable(tool):
                continue
            if tool.name in self._seen:
                continue
            cmd = self._wrap(tool)
            if cmd is not None:
                new_commands.append(cmd.local_command)
        return new_commands

    def known_tool_names(self) -> set[str]:
        """Return the names of tools already wrapped by this discovery."""
        return set(self._seen.keys())

    # --- Internals -----------------------------------------------------

    def _wrap(self, tool: "Tool") -> "_command_mod.DynamicToolCommand | None":
        existing = self._seen.get(tool.name)
        if existing is not None:
            return existing
        try:
            cmd = _command_mod.DynamicToolCommand(tool)
        except Exception as exc:  # noqa: BLE001
            log.debug("could not wrap tool %r: %s", tool.name, exc)
            return None
        self._seen[tool.name] = cmd
        return cmd


def _is_discoverable(tool: "Tool") -> bool:
    """Return True if *tool* should be exposed as a slash command."""
    if not getattr(tool, "name", None):
        return False
    if core_filter.is_core_tool(tool):
        return False
    # Skip tools that explicitly opt out of CLI exposure. The ``Tool``
    # dataclass does not yet have an ``expose_as_cli`` flag, so we
    # treat ``is_mcp`` and ``is_lsp`` as opt-out (they're integrated via
    # dedicated command paths already, e.g. ``/mcp``, ``/lsp``).
    if getattr(tool, "is_mcp", False):
        return False
    if getattr(tool, "is_lsp", False):
        return False
    return True
