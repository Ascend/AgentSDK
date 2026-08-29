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

# pylint: disable=no-name-in-module
"""CLI argv entry point.

The spec calls for tool names to be routable from the CLI sieve, but
registering each tool as its own subcommand in
``subcommand_registry`` would require either a wildcard sieve (not
supported by ``argparse``) or eagerly registering at import time (which
defeats "no manual config"). The pragmatic compromise implemented here:

* Register **one** subcommand, ``tool``, in
  :mod:`clawcodex_ext.cli.subcommand_registry`. The first argument is
 the tool name; the rest are forwarded to the tool's schema-derived
 parser.
* Also register a short alias ``t`` for ergonomic invocation.

Example::

 clawcodex-dev tool detect_modality --path /data/sample.mp4
 clawcodex-dev t detect_modality --path /data/sample.mp4
 clawcodex-dev tool --list # list discoverable tools
 clawcodex-dev tool --help <name> # show usage for a specific tool

This keeps the sieve deterministic (no dynamic name pollution) and
makes the auto-discovery transparent in the CLI.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from clawcodex_ext.tool_system.defaults import build_default_registry

from . import core_filter, discovery

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


def _build_tool_registry():
    """Build a fresh default registry for CLI invocations.

    We don't share state with the REPL/TUI (different process) so a
    fresh build is the only option. The tool resolver inside
    :class:`DynamicToolCommand` will also fall back to this if the
    supplied registry is empty.
    """
    return build_default_registry()


def _build_tool_context():
    """Build a default ``ToolContext`` for standalone CLI invocations.

    Without this every dispatch hits ``context.ensure_tool_allowed``
    on ``None`` and dies with an ``AttributeError`` — the catch-all in
    :meth:`DynamicToolCommand.invoke_from_argv` would then print a
    misleading "tool failed" message instead of a clean error.

    The CLI runs in bypass mode (same as ``clawcodex-dev headless``)
    because we are not under any user-facing permission flow — the
    operator explicitly typed ``clawcodex-dev tool <name>`` and
    accepts the tool's effects as part of that command. The workspace
    root defaults to ``$PWD`` so ``Read``/``Write`` style tools resolve
    paths against the operator's CWD.
    """
    from clawcodex_ext.permissions.types import ToolPermissionContext
    from clawcodex_ext.tool_system.context import ToolContext

    return ToolContext(
        workspace_root=Path.cwd(),
        permission_context=ToolPermissionContext(mode="bypassPermissions"),
    )


def _print_list(registry, json_output: bool = False) -> int:
    """Print the discoverable tools and exit 0."""
    disc = discovery.DynamicCommandDiscovery(registry)
    names = sorted(cmd.tool_name for cmd in disc.discover_commands())
    if json_output:
        print(json.dumps({"discoverable_tools": names}, indent=2))
        return 0
    if not names:
        print("No discoverable tools in the current registry.")
        return 0
    print("Discoverable tools (non-core, exposed as /<name>):")
    for name in names:
        print(f"  {name}")
    return 0


def run_tool_subcommand(args: list[str]) -> int:
    """Handle ``clawcodex-dev tool <name> [--args]`` (or ``tool --list``).

    Returns the process exit code.
    """
    if not args or args[0] in ("--help", "-h"):
        _print_root_help()
        return 0 if args else 2

    if args[0] == "--list":
        return _print_list(_build_tool_registry(), json_output=False)

    if args[0] == "--json":
        return _print_list(_build_tool_registry(), json_output=True)

    tool_name = args[0]
    rest = args[1:]

    if tool_name.startswith("-"):
        print(f"error: unknown flag {tool_name!r}", file=sys.stderr)
        _print_root_help()
        return 2

    registry = _build_tool_registry()
    tool = registry.get(tool_name)
    if tool is None:
        # Check if the user is asking for a core tool.
        if core_filter.is_core_tool_name(tool_name):
            print(
                f"error: '{tool_name}' is a built-in core tool and is not exposed as a CLI subcommand",
                file=sys.stderr,
            )
            return 2
        print(f"error: unknown tool '{tool_name}'", file=sys.stderr)
        print("hint: run `clawcodex-dev tool --list` to see discoverable tools", file=sys.stderr)
        return 1

    if core_filter.is_core_tool(tool):
        print(
            f"error: '{tool_name}' is a built-in core tool and is not exposed as a CLI subcommand",
            file=sys.stderr,
        )
        return 2

    from .command import DynamicToolCommand

    cmd = DynamicToolCommand(tool)
    return cmd.invoke_from_argv(
        rest,
        tool_registry=registry,
        tool_context=_build_tool_context(),
    )


def _print_root_help() -> None:
    print(
        "usage: clawcodex-dev tool [--list|--json|<tool-name> [--args ...]]\n"
        "\n"
        "Auto-exposed tool dispatcher.\n"
        "\n"
        "Subcommands:\n"
        "  --list                 List discoverable tools (non-core only).\n"
        "  --json                 Same as --list but emit JSON.\n"
        "  <tool-name> --key val  Invoke a specific tool. Run with --help for usage.\n"
        "  -h, --help             Show this message.\n"
        "\n"
        "Examples:\n"
        "  clawcodex-dev tool --list\n"
        "  clawcodex-dev tool detect_modality --path /data/sample.mp4\n"
    )
