#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""``clawcodex multimodel`` model-group management command."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from clawcodex_ext.cli.subcommand_registry import register  # pylint: disable=no-name-in-module
from .config import (
    GroupConfig,
    MultiModelConfig,
    MultiModelConfigError,
    RouteConfig,
    load_config,
    parse_slot,
    save_config,
    validate_group,
)
from .preset import PRESETS, get_preset
from .feature import disabled_message, is_multimodel_enabled


def _build_preset_choices() -> dict[str, str]:
    """Return a mapping of preset name to a short description."""
    descriptions: dict[str, str] = {}
    for name, group in sorted(PRESETS.items()):
        slots_desc = ", ".join(s.name for s in group.slots)
        agg_desc = group.aggregator or "none"
        descriptions[name] = f"{group.strategy}/{agg_desc} [{slots_desc}]"
    return descriptions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clawcodex multimodel")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- group subcommand ---
    group = sub.add_parser(
        "group",
        help="Manage model groups (create, list, show, delete, update)",
    )
    group_sub = group.add_subparsers(dest="group_command", required=True)
    create = group_sub.add_parser("create", help="Create a new model group")
    create.add_argument("name", help="Group name")
    create.add_argument(
        "--slot",
        action="append",
        required=True,
        metavar="SPEC",
        help="Slot definition: name:model@provider[,weight=N][,timeout_ms=N] (repeatable)",
    )
    _group_options(create)
    group_sub.add_parser("list", help="List all configured model groups")
    show = group_sub.add_parser("show", help="Show details of a model group")
    show.add_argument("name", help="Group name")
    delete = group_sub.add_parser("delete", help="Delete a model group")
    delete.add_argument("name", help="Group name")
    update = group_sub.add_parser("update", help="Update an existing model group")
    update.add_argument("name", help="Group name")
    update.add_argument(
        "--add-slot",
        action="append",
        default=[],
        metavar="SPEC",
        help="Slot to add: name:model@provider[,weight=N][,timeout_ms=N] (repeatable)",
    )
    update.add_argument(
        "--remove-slot",
        action="append",
        default=[],
        metavar="NAME",
        help="Name of slot to remove (repeatable)",
    )
    _group_options(update, required=False)

    # --- top-level subcommands ---
    use = sub.add_parser("use", help="Switch to a model group (runtime + config)")
    use.add_argument("name", help="Group name")
    sub.add_parser("off", help="Switch back to single-model mode")
    sub.add_parser("status", help="Show current multi-model status")

    preset_help = _build_preset_choices()
    preset = sub.add_parser(
        "preset",
        help="Apply a built-in preset as a new model group",
        description=(
            "Apply a built-in preset. Each preset creates a named group\n"
            "with preconfigured strategy, slots, and aggregator:\n"
        )
        + "\n".join(f"  {k:<22s} {v}" for k, v in preset_help.items()),
    )
    preset.add_argument(
        "name",
        choices=sorted(PRESETS),
        metavar="NAME",
        help="Preset name: " + ", ".join(sorted(PRESETS)),
    )
    return parser


def _group_options(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument(
        "--strategy",
        choices=("parallel", "voting", "routing", "fallback"),
        required=required,
        default=None,
        help="Execution strategy for the model group (default: parallel)",
    )
    parser.add_argument(
        "--aggregator",
        choices=("passthrough", "first_success", "majority", "rank", "scoring", "fusion"),
        default=None,
        help="Response aggregation method (default: passthrough)",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=None,
        help="Minimum votes required for consensus (voting/majority only)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum concurrent slot invocations (default: 5)",
    )
    parser.add_argument(
        "--scorer-provider",
        default=None,
        help="Provider for scoring/fusion (default: openai)",
    )
    parser.add_argument(
        "--scorer-model",
        default=None,
        help="Model for scoring/fusion (default: gpt-4o)",
    )
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        metavar="PATTERN:SLOT",
        help="Routing rule: keyword pattern maps to a slot name (repeatable)",
    )


def _format_group(name: str, group: GroupConfig) -> str:
    lines = [f"Group: {name}", f"Strategy: {group.strategy}"]
    if group.aggregator:
        lines.append(f"Aggregator: {group.aggregator}")
    lines.append(f"Max concurrent: {group.max_concurrent}")
    if group.min_votes is not None:
        lines.append(f"Min votes: {group.min_votes}")
    if group.aggregator in {"scoring", "rank"}:
        lines.append(f"Scorer: {group.scorer_model} ({group.scorer_provider})")
    if group.aggregator == "fusion":
        lines.append(f"Fusion model: {group.scorer_model} ({group.scorer_provider})")
    lines.append("Slots:")
    lines.extend(
        f"  - {s.name}: {s.model} ({s.provider}), weight={s.weight:g}, timeout={s.timeout_ms}ms" for s in group.slots
    )
    lines.extend(f"  route {route.pattern!r} -> {route.slot}" for route in group.routes)
    return "\n".join(lines)


@register("multimodel")
def run_multimodel_command(argv: list[str]) -> int:
    if not is_multimodel_enabled():
        return _error(disabled_message())
    try:
        args = _parser().parse_args(argv)
        config = load_config()
    except (MultiModelConfigError, SystemExit) as exc:
        return int(exc.code) if isinstance(exc, SystemExit) else _error(str(exc))
    try:
        if args.command == "group":
            return _group_command(args, config)
        if args.command == "use":
            if args.name not in config.groups:
                return _error(f"unknown model group '{args.name}'")
            save_config(replace(config, default_group=args.name))
            print(f"✓ Switched to model group '{args.name}'")
            return 0
        if args.command == "off":
            save_config(replace(config, default_group=""))
            print("✓ Switched back to single-model mode")
            return 0
        if args.command == "status":
            if not config.default_group:
                print("Status: multi-model mode is off\nRun 'clawcodex multimodel use <name>' to enable.")
                return 0
            print(
                "Status: multi-model mode is on\n"
                + _format_group(config.default_group, config.groups[config.default_group])
            )
            return 0
        if args.command == "preset":
            groups = dict(config.groups)
            groups[args.name] = get_preset(args.name)
            save_config(MultiModelConfig(args.name, groups))
            print(f"✓ Applied preset '{args.name}' and activated")
            return 0
    except MultiModelConfigError as exc:
        return _error(str(exc))
    return _error("unknown multimodel command")


def _group_command(args: argparse.Namespace, config: MultiModelConfig) -> int:
    if args.group_command == "list":
        if not config.groups:
            print("No model groups configured.")
            return 0
        for name, group in config.groups.items():
            print(
                f"{'* ' if name == config.default_group else '  '}{name}\t{group.strategy}\t{', '.join(s.name for s in group.slots)}"
            )
        return 0
    if args.group_command == "show":
        if args.name not in config.groups:
            return _error(f"unknown model group '{args.name}'")
        print(_format_group(args.name, config.groups[args.name]))
        return 0
    if args.group_command == "delete":
        if args.name not in config.groups:
            return _error(f"unknown model group '{args.name}'")
        groups = dict(config.groups)
        del groups[args.name]
        save_config(MultiModelConfig("" if config.default_group == args.name else config.default_group, groups))
        print(f"✓ Deleted model group '{args.name}'")
        return 0
    groups = dict(config.groups)
    if args.group_command == "create":
        if args.name in groups:
            return _error(f"model group '{args.name}' already exists")
        groups[args.name] = validate_group(
            GroupConfig(
                args.strategy or "parallel",
                tuple(parse_slot(v) for v in args.slot),
                args.aggregator,
                args.max_concurrent or 5,
                args.min_votes,
                args.scorer_provider or "openai",
                args.scorer_model or "gpt-4o",
                _parse_routes(args.route),
            )
        )
        save_config(MultiModelConfig(config.default_group, groups))
        print(f"✓ Created model group '{args.name}'")
        return 0
    if args.name not in groups:
        return _error(f"unknown model group '{args.name}'")
    old = groups[args.name]
    slots = [slot for slot in old.slots if slot.name not in set(args.remove_slot)]
    slots.extend(parse_slot(v) for v in args.add_slot)
    routes = _parse_routes(args.route) if args.route else old.routes
    groups[args.name] = validate_group(
        GroupConfig(
            args.strategy or old.strategy,
            tuple(slots),
            args.aggregator if args.aggregator is not None else old.aggregator,
            args.max_concurrent or old.max_concurrent,
            args.min_votes if args.min_votes is not None else old.min_votes,
            args.scorer_provider or old.scorer_provider,
            args.scorer_model or old.scorer_model,
            routes,
        )
    )
    save_config(MultiModelConfig(config.default_group, groups))
    print(f"✓ Updated model group '{args.name}'")
    return 0


def _parse_routes(values: list[str]) -> tuple[RouteConfig, ...]:
    routes = []
    for value in values:
        try:
            pattern, slot = value.rsplit(":", 1)
        except ValueError as exc:
            raise MultiModelConfigError("route must have the form PATTERN:SLOT") from exc
        routes.append(RouteConfig(pattern.strip(), slot.strip()))
    return tuple(routes)


def _error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2
