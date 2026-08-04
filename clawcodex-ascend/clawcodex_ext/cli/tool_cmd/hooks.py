#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Registration hooks for F-53.

This module glues F-53 to the two existing command surfaces:

1. **CLI argv level** (:mod:`clawcodex_ext.cli.subcommand_registry`):
   A single ``tool`` subcommand is registered that dispatches to any
   discoverable tool by name. See :mod:`clawcodex_ext.cli.tool_cmd.runtime`.

2. **REPL/TUI level** (:mod:`clawcodex_ext.command_system.registry`):
   Individual :class:`LocalCommand` objects are registered for each
   discoverable tool, so the user can invoke ``/detect_modality …`` in
   the REPL. See :func:`register_tool_commands` below.

The CLI hook is wired from ``subcommand_registry.load_builtin_subcommands``
(per the F-53 spec §1.6). The REPL/TUI hook is exposed as a public
function and called from the REPL/TUI startup code in the same place
``register_runtime_commands`` is called.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from . import discovery

if TYPE_CHECKING:
    from clawcodex_ext.command_system.registry import CommandRegistry
    from clawcodex_ext.tool_system.registry import ToolRegistry

log = logging.getLogger(__name__)

_INSTALLED = False


def install_tool_subcommand() -> None:
    """Register the ``tool`` CLI subcommand in ``subcommand_registry``.

    Idempotent — calling more than once is a no-op. Wired from
    :func:`clawcodex_ext.cli.subcommand_registry.load_builtin_subcommands`
    per F-53 spec §1.6.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from clawcodex_ext.cli.subcommand_registry import register

    @register("tool")
    def _tool_subcommand(args: list[str]) -> int:
        from .runtime import run_tool_subcommand

        return run_tool_subcommand(args)

    _INSTALLED = True
    log.debug("F-53: installed 'tool' CLI subcommand")


def register_tool_commands(
    command_registry: "CommandRegistry | None",
    tool_registry: "ToolRegistry | None" = None,
) -> int:
    """Register dynamic tool commands in *command_registry*.

    Returns the number of commands actually registered (after skipping
    name collisions and core tools). A *None* registry is treated as a
    no-op — call sites that pass ``None`` (e.g. "register globally") get
    a zero count, which matches the pattern used by
    :func:`register_runtime_commands` for the global registry.
    """
    if command_registry is None:
        return 0

    if tool_registry is None:
        # Lazy import: avoid pulling the heavy tool graph for callers
        # that only want a no-op registration. Tests typically pass
        # an explicit registry; production callers can pass the
        # runtime's ``ctx.tool_registry``.
        from clawcodex_ext.tool_system.defaults import build_default_registry

        tool_registry = build_default_registry()

    disc = discovery.DynamicCommandDiscovery(tool_registry)
    registered = 0
    for local_cmd in disc.discover():
        # Skip if a command with this name is already registered (e.g.
        # by a skill, plugin, or F-49 agent command). F-53 is purely
        # additive — it never overwrites an existing handler.
        try:
            if command_registry.has(local_cmd.name):
                log.debug(
                    "F-53: skipping %r — already registered in command registry",
                    local_cmd.name,
                )
                continue
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        try:
            command_registry.register(local_cmd)
        except Exception as exc:  # noqa: BLE001
            log.debug("F-53: failed to register %r: %s", local_cmd.name, exc)
            continue
        registered += 1
    return registered
