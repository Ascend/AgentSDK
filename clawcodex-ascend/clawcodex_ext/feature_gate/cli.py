#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

"""Feature Gate CLI subcommand.

Usage::

    clawcodex feature list                          # List all registered features
    clawcodex feature list --enabled                # Only enabled features
    clawcodex feature list --disabled               # Only disabled features
    clawcodex feature get <NAME>                    # Show effective state of one feature
    clawcodex feature set <NAME> --on               # Enable a feature (persist)
    clawcodex feature set <NAME> --off              # Disable a feature (persist)
    clawcodex feature reload                        # Reload config from disk
    clawcodex feature reset                         # Clear all overrides, reload defaults
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

# pylint: disable=no-name-in-module  # cli: pending patch migration
from clawcodex_ext.cli.subcommand_registry import register


def _get_registry():
    """Lazy import of the singleton to avoid circular imports."""
    from clawcodex_ext.feature_gate import get_registry as _gr

    return _gr()


# ------------------------------------------------------------------
# Parser builders
# ------------------------------------------------------------------


def _build_list_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clawcodex feature list")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--enabled", action="store_true", help="Show only enabled features")
    g.add_argument("--disabled", action="store_true", help="Show only disabled features")
    p.add_argument("--json", action="store_true", help="Output in JSON format")
    return p


def _build_set_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clawcodex feature set")
    p.add_argument("name", help="Feature flag name")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--on", action="store_true", help="Enable the feature")
    g.add_argument("--off", action="store_true", help="Disable the feature")
    return p


def _build_get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clawcodex feature get")
    p.add_argument("name", help="Feature flag name")
    return p


# ------------------------------------------------------------------
# Handlers
# ------------------------------------------------------------------


@register("feature")
def run_feature_command(args: Sequence[str]) -> int:
    """Dispatch the ``feature`` subcommand.

    Usage: ``clawcodex feature <list|get|set|reload|reset>``
    """
    if not args:
        print(
            "usage: clawcodex feature <list|get|set|reload|reset>\n"
            "\n"
            "Commands:\n"
            "  list          List all registered feature flags\n"
            "  get <NAME>    Show effective state of a feature\n"
            "  set <NAME> --on|--off  Toggle a feature (persists to config)\n"
            "  reload        Reload feature config from disk\n"
            "  reset         Clear all overrides and reload defaults",
            file=sys.stderr,
        )
        return 2

    subcmd = args[0]
    rest = args[1:]

    if subcmd == "list":
        return _handle_list(rest)
    if subcmd == "get":
        return _handle_get(rest)
    if subcmd == "set":
        return _handle_set(rest)
    if subcmd == "reload":
        return _handle_reload()
    if subcmd == "reset":
        return _handle_reset()

    print(f"Unknown feature subcommand: {subcmd}", file=sys.stderr)
    print(
        "usage: clawcodex feature <list|get|set|reload|reset>",
        file=sys.stderr,
    )
    return 2


# ------------------------------------------------------------------
# Sub-handlers
# ------------------------------------------------------------------


def _handle_list(args: Sequence[str]) -> int:
    parser = _build_list_parser()
    parsed = parser.parse_args(args)
    reg = _get_registry()
    features = reg.list_features()

    if parsed.json:
        states = reg.get_effective_states()
        output = []
        for name in features:
            flag = reg.get_flag(name)
            entry = {
                "name": name,
                "enabled": states.get(name, False),
                "default": flag.default if flag else False,
                "deps": flag.deps if flag else [],
                "mutex_with": flag.mutex_with if flag else [],
                "description": flag.description if flag else "",
            }
            output.append(entry)
        if parsed.enabled:
            output = [e for e in output if e["enabled"]]
        elif parsed.disabled:
            output = [e for e in output if not e["enabled"]]
        json.dump(output, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if not features:
        print("(no features registered)")
        return 0

    # Apply text-mode filters
    if parsed.enabled:
        features = [n for n in features if reg.is_enabled(n)]
    elif parsed.disabled:
        features = [n for n in features if not reg.is_enabled(n)]

    enabled_count = len([f for f in features if reg.is_enabled(f)])
    total_count = len(features)

    if parsed.enabled or parsed.disabled:
        print(f"Filtered features: {total_count} shown")
    else:
        print(f"Registered features: {total_count} ({enabled_count} enabled, {total_count - enabled_count} disabled)")
    print()

    for name in features:
        flag = reg.get_flag(name)
        state = reg.is_enabled(name)
        marker = "+" if state else "-"
        deps = f" deps=[{','.join(flag.deps)}]" if flag and flag.deps else ""
        mutex = f" mutex=[{','.join(flag.mutex_with)}]" if flag and flag.mutex_with else ""
        desc = f" -- {flag.description}" if flag and flag.description else ""
        print(f"  [{marker}] {name}{deps}{mutex}{desc}")

    return 0


def _handle_get(args: Sequence[str]) -> int:
    parser = _build_get_parser()
    parsed = parser.parse_args(args)
    reg = _get_registry()

    if parsed.name not in reg.list_features():
        print(f"Unknown feature: '{parsed.name}'", file=sys.stderr)
        return 1

    state = reg.is_enabled(parsed.name)
    flag = reg.get_flag(parsed.name)
    print(f"{parsed.name}: {'enabled' if state else 'disabled'}")
    if flag:
        print(f"  default: {flag.default}")
        if flag.deps:
            missing = reg.check_deps(parsed.name)
            print(f"  deps: {', '.join(flag.deps)}" + (f" (missing: {', '.join(missing)})" if missing else ""))
        if flag.mutex_with:
            conflicts = reg.check_mutex(parsed.name)
            print(
                f"  mutex: {', '.join(flag.mutex_with)}"
                + (f" (conflicts: {', '.join(conflicts)})" if conflicts else "")
            )
    return 0


def _handle_set(args: Sequence[str]) -> int:
    parser = _build_set_parser()
    parsed = parser.parse_args(args)
    reg = _get_registry()

    if parsed.name not in reg.list_features():
        print(f"Unknown feature: '{parsed.name}'", file=sys.stderr)
        return 1

    # Validate before setting
    if parsed.on:
        ok, errors = reg.validate_registration(parsed.name)
        if not ok:
            for err in errors:
                print(f"Error: {err}", file=sys.stderr)
            return 1
        reg.enable_feature(parsed.name)
        print(f"Feature '{parsed.name}' enabled (override set)")
    else:
        reg.disable_feature(parsed.name)
        print(f"Feature '{parsed.name}' disabled (override set)")

    # Persist to config file
    reg.save_config()
    return 0


def _handle_reload() -> int:
    reg = _get_registry()
    reg.reload_config()
    print("Feature config reloaded from disk")
    return 0


def _handle_reset() -> int:
    reg = _get_registry()
    reg.clear_all_overrides()
    reg.reload_config()
    print("All feature overrides cleared; reloaded defaults from config")
    return 0
