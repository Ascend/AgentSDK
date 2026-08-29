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

"""The ``/multimodel`` REPL/TUI runtime command."""

from __future__ import annotations

import shlex
from argparse import ArgumentParser, ArgumentError
from typing import Any

from clawcodex_ext.command_system.types import LocalCommand, LocalCommandResult

from .config import (
    GroupConfig,
    MultiModelConfig,
    MultiModelConfigError,
    RouteConfig,
    load_config,
    parse_slot,
    resolve_active_group,
    save_config,
    validate_group,
)
from .feature import disabled_message, is_multimodel_enabled
from .preset import get_preset

# ── Registration ──────────────────────────────────────────────────────


def register_multimodel_runtime_command(registry: Any | None = None) -> None:
    from clawcodex_ext.command_system.registry import get_command_registry

    reg = registry or get_command_registry()
    command = LocalCommand(
        name="multimodel",
        description=("Manage multi-model groups — status, use, off, preset, group (list/show/delete/create/update)"),
        argument_hint=(
            "[status|use NAME|off|preset NAME|group list|group show NAME|"
            "group delete NAME|group create NAME ...|group update NAME ...]"
        ),
    )
    command.set_call(_call)
    reg.register(command)


# ── Main dispatch ─────────────────────────────────────────────────────


def _call(args: str, context: Any) -> LocalCommandResult:
    if not is_multimodel_enabled():
        return _text(disabled_message())

    try:
        tokens = shlex.split(args)
        config = load_config()
    except (ValueError, MultiModelConfigError) as exc:
        return _text(f"error: {exc}")

    if not tokens:
        return _status(config, context)

    cmd = tokens[0]

    if cmd == "status":
        return _status(config, context)
    if cmd == "use" and len(tokens) == 2:
        return _use(tokens[1], config, context)
    if cmd == "off":
        return _off(config, context)
    if cmd == "preset" and len(tokens) == 2:
        return _preset(tokens[1], config)
    if cmd == "group" and len(tokens) >= 2:
        return _group(tokens[1:], config, context)

    return _text(
        "Usage: /multimodel [status|use NAME|off|preset NAME|group list|"
        "group show NAME|group delete NAME|group create NAME ...|"
        "group update NAME ...]"
    )


# ── Top-level subcommand handlers ─────────────────────────────────────


def _status(config: MultiModelConfig, context: Any) -> LocalCommandResult:
    runtime = _get_runtime(context)
    selected = (
        getattr(runtime, "multimodel_group", None)
        if runtime is not None
        else getattr(context, "multimodel_group", None)
    )
    active = resolve_active_group(runtime_group=selected, config=config)
    if not active:
        groups = ", ".join(config.groups) if config.groups else "(none)"
        return _text(
            f"Status: multi-model mode is off\nAvailable model groups: {groups}\nType /multimodel use <name> to enable."
        )
    group = config.groups.get(active)
    if group is None:
        return _text(f"Active model group '{active}' not found.")
    return _text(_format_group(active, group, enabled=True))


def _use(name: str, config: MultiModelConfig, context: Any) -> LocalCommandResult:
    if name not in config.groups:
        return _text(f"error: unknown model group '{name}'")
    runtime = _get_runtime(context)
    if runtime is not None:
        try:
            runtime.swap_multimodel(name)
        except Exception as exc:
            return _text(f"error: cannot enable model group '{name}': {exc}")
    else:
        setattr(context, "multimodel_group", name)
    save_config(MultiModelConfig(name, config.groups))
    group = config.groups[name]
    return _text(
        f"✓ Switched to model group '{name}'\n"
        f"Strategy: {group.strategy} | "
        f"Slots: {', '.join(slot.name for slot in group.slots)}"
    )


def _off(config: MultiModelConfig, context: Any) -> LocalCommandResult:
    runtime = _get_runtime(context)
    if runtime is not None:
        try:
            runtime.disable_multimodel()
        except Exception as exc:
            return _text(f"error: cannot disable multi-model mode: {exc}")
    else:
        setattr(context, "multimodel_group", "")
    save_config(MultiModelConfig("", dict(config.groups)))
    return _text("✓ Switched back to single-model mode")


def _preset(name: str, config: MultiModelConfig) -> LocalCommandResult:
    try:
        group = get_preset(name)
    except KeyError:
        return _text(f"error: unknown preset '{name}'. Available: " + ", ".join(sorted(_available_presets())))
    groups = dict(config.groups)
    groups[name] = group
    save_config(MultiModelConfig(name, groups))
    return _text(
        f"✓ Applied preset '{name}' and activated\n"
        f"Strategy: {group.strategy} | "
        f"Slots: {', '.join(slot.name for slot in group.slots)}"
    )


# ── /multimodel group subcommand ──────────────────────────────────────


def _group(tokens: list[str], config: MultiModelConfig, context: Any) -> LocalCommandResult:
    cmd = tokens[0]
    if cmd == "list":
        return _group_list(config)
    if cmd == "show" and len(tokens) == 2:
        return _group_show(tokens[1], config)
    if cmd == "delete" and len(tokens) == 2:
        return _group_delete(tokens[1], config)
    if cmd in ("create", "update") and len(tokens) >= 2:
        return _group_create_or_update(cmd, tokens[1], tokens[2:], config)
    return _text(
        "Usage:\n"
        "  /multimodel group list\n"
        "  /multimodel group show <name>\n"
        "  /multimodel group delete <name>\n"
        "  /multimodel group create <name> [--key value ...]\n"
        "  /multimodel group update <name> [--key value ...]\n\n"
        "Structured mode (--key value):\n"
        "  --slot name:model@provider[,weight=N][,timeout_ms=N]  (repeatable)\n"
        "  --strategy parallel|voting|routing|fallback\n"
        "  --aggregator passthrough|first_success|majority|rank|scoring|fusion\n"
        "  --max-concurrent N    --min-votes N\n"
        "  --scorer-provider P   --scorer-model M\n"
        "  --route PATTERN:SLOT  (repeatable)\n"
        "For update also: --add-slot SPEC  --remove-slot NAME"
    )


def _group_list(config: MultiModelConfig) -> LocalCommandResult:
    if not config.groups:
        return _text("No model groups configured.")
    lines: list[str] = []
    for name, group in config.groups.items():
        marker = "* " if name == config.default_group else "  "
        slots_summary = ", ".join(s.name for s in group.slots)
        lines.append(f"{marker}{name}  {group.strategy}  [{slots_summary}]")
    return _text("\n".join(lines))


def _group_show(name: str, config: MultiModelConfig) -> LocalCommandResult:
    if name not in config.groups:
        return _text(f"error: unknown model group '{name}'")
    return _text(_format_group(name, config.groups[name], enabled=(name == config.default_group)))


def _group_delete(name: str, config: MultiModelConfig) -> LocalCommandResult:
    if name not in config.groups:
        return _text(f"error: unknown model group '{name}'")
    groups = dict(config.groups)
    del groups[name]
    new_default = "" if config.default_group == name else config.default_group
    save_config(MultiModelConfig(new_default, groups))
    return _text(f"✓ Deleted model group '{name}'")


# ── group create / update (dual-mode) ─────────────────────────────────


def _group_create_or_update(
    action: str,  # "create" | "update"
    name: str,
    tokens: list[str],  # tokens after group-name
    config: MultiModelConfig,
) -> LocalCommandResult:
    if action == "create" and name in config.groups:
        return _text(f"error: model group '{name}' already exists")
    if action == "update" and name not in config.groups:
        return _text(f"error: unknown model group '{name}'")
    return _group_structured(action, name, tokens, config)


# ── Structured mode ──────────────────────────────────────────────────


def _create_parser() -> ArgumentParser:
    """Build a fresh parser for ``group create``."""
    p = ArgumentParser(exit_on_error=False, add_help=False)
    p.add_argument("--slot", action="append", required=True, metavar="SPEC")
    p.add_argument(
        "--strategy",
        choices=("parallel", "voting", "routing", "fallback"),
        default="parallel",
    )
    p.add_argument(
        "--aggregator",
        choices=(
            "passthrough",
            "first_success",
            "majority",
            "rank",
            "scoring",
            "fusion",
        ),
    )
    p.add_argument("--min-votes", type=int)
    p.add_argument("--max-concurrent", type=int, default=5)
    p.add_argument("--scorer-provider", default="openai")
    p.add_argument("--scorer-model", default="gpt-4o")
    p.add_argument("--route", action="append", default=[], metavar="PATTERN:SLOT")
    return p


def _update_parser() -> ArgumentParser:
    """Build a fresh parser for ``group update``."""
    p = ArgumentParser(exit_on_error=False, add_help=False)
    p.add_argument("--add-slot", action="append", default=[], metavar="SPEC")
    p.add_argument("--remove-slot", action="append", default=[], metavar="NAME")
    p.add_argument("--slot", action="append", default=[], metavar="SPEC")
    p.add_argument("--strategy", choices=("parallel", "voting", "routing", "fallback"))
    p.add_argument(
        "--aggregator",
        choices=(
            "passthrough",
            "first_success",
            "majority",
            "rank",
            "scoring",
            "fusion",
        ),
    )
    p.add_argument("--min-votes", type=int)
    p.add_argument("--max-concurrent", type=int)
    p.add_argument("--scorer-provider")
    p.add_argument("--scorer-model")
    p.add_argument("--route", action="append", default=[], metavar="PATTERN:SLOT")
    return p


def _group_structured(action: str, name: str, tokens: list[str], config: MultiModelConfig) -> LocalCommandResult:
    """Parse ``--key value`` tokens into group config and persist."""
    try:
        if action == "create":
            args = _create_parser().parse_args(tokens)
            slots = tuple(parse_slot(s) for s in args.slot)
            routes = _parse_routes(args.route)
            group = validate_group(
                GroupConfig(
                    strategy=args.strategy,
                    slots=slots,
                    aggregator=args.aggregator,
                    max_concurrent=args.max_concurrent,
                    min_votes=args.min_votes,
                    scorer_provider=args.scorer_provider,
                    scorer_model=args.scorer_model,
                    routes=routes,
                )
            )
            groups = dict(config.groups)
            groups[name] = group
            save_config(MultiModelConfig(config.default_group, groups))
            return _text(f"✓ Created model group '{name}'\n{_format_group(name, group)}")

        else:  # update
            args = _update_parser().parse_args(tokens)
            old = config.groups[name]

            # Slot mutations: remove → add → replace (if --slot given)
            remove_slots = set(args.remove_slot)
            slots = [s for s in old.slots if s.name not in remove_slots]
            slots.extend(parse_slot(s) for s in args.add_slot)
            if args.slot:
                slots = [parse_slot(s) for s in args.slot]

            # Scalar overrides — explicit user values take priority
            routes = _parse_routes(args.route) if args.route else old.routes
            strategy = args.strategy if args.strategy is not None else old.strategy
            aggregator = args.aggregator if args.aggregator is not None else old.aggregator
            max_concurrent = args.max_concurrent if args.max_concurrent is not None else old.max_concurrent
            min_votes = args.min_votes if args.min_votes is not None else old.min_votes
            scorer_provider = args.scorer_provider if args.scorer_provider is not None else old.scorer_provider
            scorer_model = args.scorer_model if args.scorer_model is not None else old.scorer_model

            group = validate_group(
                GroupConfig(
                    strategy=strategy,
                    slots=tuple(slots),
                    aggregator=aggregator,
                    max_concurrent=max_concurrent,
                    min_votes=min_votes,
                    scorer_provider=scorer_provider,
                    scorer_model=scorer_model,
                    routes=routes,
                )
            )
            groups = dict(config.groups)
            groups[name] = group
            save_config(MultiModelConfig(config.default_group, groups))
            return _text(f"✓ Updated model group '{name}'\n{_format_group(name, group)}")

    except (MultiModelConfigError, ArgumentError, SystemExit) as exc:
        return _text(f"error: {exc}")


def _parse_routes(values: list[str]) -> tuple[RouteConfig, ...]:
    """Split ``PATTERN:SLOT`` strings into route configs."""
    routes: list[RouteConfig] = []
    for value in values:
        try:
            pattern, slot = value.rsplit(":", 1)
        except ValueError:
            raise MultiModelConfigError("route must have the form PATTERN:SLOT")
        routes.append(RouteConfig(pattern.strip(), slot.strip()))
    return tuple(routes)


# ── Helpers ────────────────────────────────────────────────────────────


def _get_runtime(context: Any) -> Any | None:
    return getattr(context, "runtime_context", None) or getattr(context, "runtime", None)


def _available_presets() -> list[str]:
    from .preset import PRESETS

    return list(PRESETS)


def _format_group(name: str, group: GroupConfig, *, enabled: bool = False) -> str:
    status = "enabled" if enabled else "disabled"
    lines = [
        f"Group: {name}",
        f"Status: {status}",
        f"Strategy: {group.strategy}",
    ]
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
    for slot in group.slots:
        extra = []
        if slot.weight != 1.0:
            extra.append(f"weight: {slot.weight:g}")
        if slot.timeout_ms != 120_000:
            extra.append(f"timeout: {slot.timeout_ms}ms")
        tag = f"  ({', '.join(extra)})" if extra else ""
        lines.append(f"  • {slot.name}: {slot.model} ({slot.provider}){tag}")
    for route in group.routes:
        lines.append(f"  route {route.pattern!r} → {route.slot}")
    return "\n".join(lines)


def _text(value: str) -> LocalCommandResult:
    return LocalCommandResult(type="text", value=value)
