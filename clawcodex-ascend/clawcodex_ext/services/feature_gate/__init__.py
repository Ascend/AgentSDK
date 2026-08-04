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

"""Facade -- services/feature_gate/ re-exports from clawcodex_ext.

Canonical implementation lives at ``clawcodex_ext/feature_gate/``.
This module is kept so existing imports of
``from clawcodex_ext.services.feature_gate import ...`` continue to
work, and so ``src/services/feature_gate/__init__.py`` can re-export
from the canonical location.
"""

from __future__ import annotations

# Core types and registry
from clawcodex_ext.feature_gate import (  # pylint: disable=no-name-in-module
    ConfigStore,
    FeatureFlag,
    FeatureRegistry,
    feature_gated,
    feature_gated_class,
    get_registry,
    guarded_call,
    guarded_is_enabled,
    register_defaults,
    reset_registry,
    run_feature_command,
)

# Aliases used by src/services/feature_gate/__init__.py
FeatureConfigStore = ConfigStore

__all__ = [
    "ConfigStore",
    "FeatureFlag",
    "FeatureRegistry",
    "FeatureConfigStore",
    "feature_gated",
    "feature_gated_class",
    "get_registry",
    "guarded_call",
    "guarded_is_enabled",
    "register_defaults",
    "reset_registry",
    "run_feature_command",
    "conditional_register",
    "feature_gated_function",
    "add_feature_gate_args",
]


def conditional_register(name: str, cls):
    """Register *cls* only if *name* feature is enabled.

    Convenience wrapper around ``@feature_gated_class`` for imperative
    code paths that cannot use a decorator.
    """
    reg = get_registry()
    if reg.is_enabled(name):
        return cls
    return None


def feature_gated_function(feature_name: str, fallback=None):
    """Alias for ``@feature_gated`` for compatibility with older naming."""
    return feature_gated(feature_name, fallback=fallback)


def add_feature_gate_args(parser):
    """Add ``--enable`` / ``--disable`` arguments to an existing argparse parser.

    This is a convenience helper for integrating feature-gate CLI flags
    into the main ClawCodex argument parser.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--enable",
        nargs="*",
        metavar="FEATURE",
        help="Enable the given feature flag(s)",
    )
    group.add_argument(
        "--disable",
        nargs="*",
        metavar="FEATURE",
        help="Disable the given feature flag(s)",
    )


def apply_feature_gate_args(args):
    """Apply ``--enable`` / ``--disable`` values from parsed args to the registry.

    Call this early in the CLI bootstrap pipeline (after argparse) so
    that ``@feature_gated`` decorators see the right state.
    """
    reg = get_registry()
    if getattr(args, "enable", None):
        for name in args.enable:
            reg.enable_feature(name)
    if getattr(args, "disable", None):
        for name in args.disable:
            reg.disable_feature(name)


def handle_list_features(enabled_only=False, disabled_only=False, as_json=False):
    """Programmatic helper to list features (used by CLI and tests).

    Returns a list of dicts when *as_json* is True, otherwise prints
    to stdout and returns 0.
    """
    import json as _json
    import sys as _sys

    reg = get_registry()
    features = reg.list_features()
    states = reg.get_effective_states()

    if as_json:
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
        if enabled_only:
            output = [e for e in output if e["enabled"]]
        elif disabled_only:
            output = [e for e in output if not e["enabled"]]
        _json.dump(output, _sys.stdout, indent=2)
        _sys.stdout.write("\n")
        return 0

    if not features:
        print("(no features registered)")
        return 0

    enabled_count = sum(1 for s in states.values() if s)
    print(f"Registered features: {len(features)} ({enabled_count} enabled, {len(features) - enabled_count} disabled)")
    print()
    for name in features:
        state = states.get(name, False)
        marker = "+" if state else "-"
        flag = reg.get_flag(name)
        deps = f" deps=[{','.join(flag.deps)}]" if flag and flag.deps else ""
        mutex = f" mutex=[{','.join(flag.mutex_with)}]" if flag and flag.mutex_with else ""
        desc = f" -- {flag.description}" if flag and flag.description else ""
        print(f"  [{marker}] {name}{deps}{mutex}{desc}")

    if enabled_only:
        print("\n  (filtered: enabled only)")
    elif disabled_only:
        print("\n  (filtered: disabled only)")
    return 0
