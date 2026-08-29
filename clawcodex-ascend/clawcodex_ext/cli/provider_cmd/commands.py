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

"""Fast-path provider CLI commands."""

from __future__ import annotations

import sys

from clawcodex_ext.cli.model_cmd.registry import ModelRegistry
from clawcodex_ext.cli.model_cmd.resolver import resolve
from clawcodex_ext.cli.model_cmd.store import ModelStore
from clawcodex_ext.cli.provider_cmd.errors import ProviderCommandError
from clawcodex_ext.cli.subcommand_registry import register


_USAGE = (
    "usage: clawcodex provider [list | show [NAME] | current | use NAME | unset]\n\n"
    "Subcommands:\n"
    "  (no args)          Show current provider and model.\n"
    "  list               List all known providers with auth + default-model info.\n"
    "  show [NAME]        Show a provider. NAME omitted => current.\n"
    "  current            Same as no-args (alias).\n"
    "  use NAME           Persist a new default provider.\n"
    "  unset              Reset to the built-in default (anthropic).\n"
    "  help, --help, -h   Print this help.\n"
    "  --all, ls          Same as no-args + list (REPL-equivalent view).\n"
)


@register("provider")
def run_provider_command(args: list[str]) -> int:
    from clawcodex_ext.providers import ensure_provider_extensions_installed

    ensure_provider_extensions_installed()
    # No args => current state (mirrors `/provider` and `git config --get` zero-arg idiom).
    if not args:
        print(format_provider_current())
        return 0

    command = args[0]
    rest = args[1:]

    # Top-level help / list flags — handled before subcommand dispatch.
    if command in ("help", "--help", "-h"):
        print(_USAGE)
        return 0
    if command in ("--all", "ls"):
        print(_format_provider_explore())
        return 0

    try:
        if command == "list":
            _reject_unknown_args(rest)
            print(format_provider_list())
            return 0
        if command == "show":
            _reject_unknown_args(rest)
            name = rest[0] if rest else None
            print(format_provider_show(name))
            return 0
        if command == "current":
            _reject_unknown_args(rest)
            print(format_provider_current())
            return 0
        if command == "use":
            provider = _parse_use_args(rest)
            provider = ModelRegistry().validate_provider(provider)
            ModelStore().set_default_provider(provider)
            print(f"Default provider set to: {provider}")
            print(
                "(persisted to config; takes effect on next REPL launch — "
                "use /provider inside the REPL to switch immediately)"
            )
            return 0
        if command == "unset":
            _reject_unknown_args(rest)
            provider = ModelStore().unset_default_provider()
            print(f"Default provider reset to: {provider}")
            return 0
    except ProviderCommandError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(_USAGE, file=sys.stderr)
    return 2


def _format_provider_explore() -> str:
    """Combined ``current + list`` view, mirroring what ``/provider`` shows in the REPL."""
    return format_provider_current() + "\n\n" + format_provider_list()


def format_provider_list() -> str:
    lines = ["Providers:"]
    for status in ModelRegistry().provider_statuses():
        configured = "yes" if status.authenticated else "no"
        model = status.configured_model or status.default_model
        lines.append(f"  {status.name}\t{status.label}\tmodel={model}\tconfigured={configured}")
    return "\n".join(lines)


def format_provider_show(name: str | None = None) -> str:
    registry = ModelRegistry()
    if name is None:
        name = resolve(registry=registry).provider
    name = registry.validate_provider(name)
    info = registry.provider_info[name]
    models = ", ".join(registry.available_models(name))
    return "\n".join(
        [
            f"Provider: {name}",
            f"Label: {info['label']}",
            f"Default Base URL: {info['default_base_url']}",
            f"Default Model: {info['default_model']}",
            f"Available Models: {models}",
        ]
    )


def format_provider_current() -> str:
    resolution = resolve()
    # Mirror ``format_model_current``: clean ``provider: …\nmodel: …``,
    # no source labels.  Debugging source info is available via
    # ``run_model_command``'s debug log, not in the user-facing output.
    return "\n".join(
        [
            f"provider: {resolution.provider}",
            f"model: {resolution.model}",
        ]
    )


def _parse_use_args(args: list[str]) -> str:
    if not args:
        raise ProviderCommandError("provider NAME is required. Example: clawcodex provider use anthropic")
    provider = args[0]
    _reject_unknown_args(args[1:])
    if provider.startswith("--"):
        raise ProviderCommandError(f"Unknown argument: {provider}")
    return provider


def _reject_unknown_args(args: list[str]) -> None:
    for token in args:
        if token.startswith("--"):
            raise ProviderCommandError(f"Unknown argument: {token}")
        raise ProviderCommandError(f"Unexpected positional argument: {token}")
