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

"""Fast-path model CLI commands."""

from __future__ import annotations

import sys

from clawcodex_ext.cli.model_cmd.errors import ModelCommandError, UnknownModelError
from clawcodex_ext.cli.model_cmd.registry import ModelRegistry
from clawcodex_ext.cli.model_cmd.resolver import resolve
from clawcodex_ext.cli.model_cmd.store import ModelStore
from clawcodex_ext.cli.subcommand_registry import register


_USAGE = (
    "usage: clawcodex model [list [--provider NAME] | show [NAME] "
    "[--provider NAME] | current | use NAME [--provider NAME]]\n\n"
    "Subcommands:\n"
    "  (no args)              Show current provider and model.\n"
    "  list [--provider P]    List available models (optionally for one provider).\n"
    "  show [NAME] [--provider P]\n"
    "                         Show a model. NAME omitted => current.\n"
    "  current                Same as no-args (alias).\n"
    "  use NAME [--provider P]\n"
    "                         Persist a new default model.\n"
    "  help, --help, -h       Print this help.\n"
    "  --list, ls             Same as no-args + list (REPL-equivalent view).\n"
)


@register("model")
def run_model_command(args: list[str]) -> int:
    from clawcodex_ext.providers import ensure_provider_extensions_installed

    ensure_provider_extensions_installed()
    # No args => current state (mirrors `/model` and `git status` zero-arg idiom).
    if not args:
        print(format_model_current())
        return 0

    command = args[0]
    rest = args[1:]

    # Top-level help / list flags — handled before subcommand dispatch.
    if command in ("help", "--help", "-h"):
        print(_USAGE)
        return 0
    if command in ("--list", "ls"):
        print(_format_model_explore(wait_for_refresh=True))
        return 0

    try:
        if command == "list":
            provider = _parse_provider_flag(rest)
            print(format_model_list(provider, wait_for_refresh=True))
            return 0
        if command == "show":
            model, provider = _parse_show_args(rest)
            print(format_model_show(model, provider))
            return 0
        if command == "current":
            print(format_model_current())
            return 0
        if command == "use":
            model, provider = _parse_use_args(rest)
            messages = use_model(model, provider=provider)
            print("\n".join(messages))
            return 0
    except ModelCommandError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(_USAGE, file=sys.stderr)
    return 2


def _format_model_explore(*, wait_for_refresh: bool = False) -> str:
    """Combined ``current + list`` view, mirroring what ``/model`` shows in the REPL."""
    return format_model_current() + "\n\n" + format_model_list(wait_for_refresh=wait_for_refresh)


def format_model_list(provider: str | None = None, *, wait_for_refresh: bool = False) -> str:
    """Render the selected provider's model catalog.

    Args:
        provider: Provider to display, or the currently resolved provider.
        wait_for_refresh: Whether to wait briefly for live catalog discovery.

    Returns:
        A human-readable model catalog with current-model and refresh status.
    """

    registry = ModelRegistry()
    resolution = resolve() if provider is None else None
    provider = provider or resolution.provider
    from src.providers import canonical_provider_name

    provider = canonical_provider_name(provider)
    resolved_current = resolution.model if resolution is not None else None
    # When no provider is specified (or the fallback is "unknown" because
    # the caller has no runtime context), iterate over every configured
    # provider so the fallback still surfaces useful catalog information.
    # Otherwise restrict to the single requested provider.
    if not provider or provider == "unknown":
        providers = registry.provider_names()
    else:
        providers = [provider]
    lines = ["Models:"]
    for provider_name in providers:
        registry.validate_provider(provider_name)
        models = registry.configured_models(provider_name)
        current_model = resolved_current or registry.provider_default_model(provider_name)
        discovery_warning: str | None = None
        try:
            from src.config import get_provider_config
            from src.providers.runtime import build_provider_from_config

            provider_config = get_provider_config(provider_name) or {}
            if resolved_current is None:
                current_model = provider_config.get("default_model") or current_model
            runtime_provider = build_provider_from_config(provider_name, current_model)
            from clawcodex_ext.providers.model_catalog_cache import (
                get_model_catalog,
                refresh_model_catalog,
            )

            if wait_for_refresh:
                snapshot = refresh_model_catalog(provider_name, runtime_provider)
            else:
                snapshot = get_model_catalog(provider_name, runtime_provider)
            if snapshot.models:
                models = list(snapshot.models)
            if snapshot.error:
                shown = "cached" if snapshot.source == "stale-cache" else "configured fallback"
                discovery_warning = (
                    f"Last model catalog refresh failed for {provider_name}: {snapshot.error}; showing {shown} models."
                )
            elif snapshot.refreshing:
                shown = "cached" if snapshot.source == "stale-cache" else "configured fallback"
                discovery_warning = (
                    f"Model catalog refresh is running in the background for {provider_name}; showing {shown} models."
                )
        except Exception as exc:
            discovery_warning = f"Model discovery failed for {provider_name}: {exc}; showing configured fallback."
        if current_model and current_model not in models:
            models = [current_model, *models]
        lines.append(f"  {provider_name}:")
        if discovery_warning:
            lines.append(f"    ! {discovery_warning}")
        for model in models:
            marker = " *" if model == current_model else ""
            lines.append(f"    {model}{marker}")
    return "\n".join(lines)


def format_model_show(model: str | None = None, provider: str | None = None) -> str:
    registry = ModelRegistry()
    if model is None:
        current = resolve(registry=registry)
        model = current.model
        provider = provider or current.provider
    elif provider is None:
        try:
            provider = registry.infer_provider_for_model(model)
        except UnknownModelError:
            raise UnknownModelError(
                f"{model} (use 'clawcodex model list' to see available models, "
                f"or pass --provider NAME to specify a provider)"
            ) from None
    try:
        registry.validate_model(model, provider)
    except UnknownModelError:
        raise UnknownModelError(
            f"Model {model} is not available for provider {provider}. "
            f"Use 'clawcodex model list' to see available models, "
            f"or pass --provider NAME to try a different provider."
        ) from None
    return format_current_pair(model, provider)


def format_model_current() -> str:
    resolution = resolve()
    return format_current_pair(resolution.model, resolution.provider)


def format_current_pair(model: str, provider: str) -> str:
    """Single canonical ``provider: …\\nmodel: …`` rendering.

    Used by both ``format_model_current`` and ``format_model_show`` so the
    CLI has exactly one shape for "what's the active model?" output.  No
    source labels — those are debugging info, not for end users.
    """
    return "\n".join([f"provider: {provider}", f"model: {model}"])


def use_model(model: str, *, provider: str | None = None) -> list[str]:
    from clawcodex_ext.providers import ensure_provider_extensions_installed

    ensure_provider_extensions_installed()
    registry = ModelRegistry()
    if provider is None:
        provider = registry.infer_provider_for_model(model, discover=False)
    else:
        provider = registry.validate_provider(provider)

    store = ModelStore(registry)
    store.set_default_provider(provider)
    store.set_default_model_persist_unknown(provider, model)
    return [
        f"Default provider set to: {provider}",
        f"Default model for {provider} set to: {model}",
        "(persisted to config; takes effect on next REPL launch — use /model inside the REPL to switch immediately)",
    ]


def _parse_provider_flag(args: list[str]) -> str | None:
    provider = None
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == "--provider" and idx + 1 < len(args):  # nosec B105
            provider = args[idx + 1]
            idx += 2
            continue
        raise ModelCommandError(f"Unknown argument: {token}")
    return provider


def _parse_show_args(args: list[str]) -> tuple[str | None, str | None]:
    model = None
    provider = None
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == "--provider":  # nosec B105
            if idx + 1 >= len(args):
                raise ModelCommandError("--provider requires a value")
            provider = args[idx + 1]
            idx += 2
            continue
        if token.startswith("--"):
            raise ModelCommandError(f"Unknown argument: {token}")
        if model is None:
            model = token
            idx += 1
            continue
        raise ModelCommandError(f"Unexpected positional argument: {token}")
    return model, provider


def _parse_use_args(args: list[str]) -> tuple[str, str | None]:
    provider = None
    model: str | None = None
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == "--provider":  # nosec B105
            if idx + 1 >= len(args):
                raise ModelCommandError("--provider requires a value")
            provider = args[idx + 1]
            idx += 2
            continue
        if token.startswith("--"):
            raise ModelCommandError(f"Unknown argument: {token}")
        if model is None:
            model = token
            idx += 1
            continue
        raise ModelCommandError(f"Unexpected positional argument: {token}")
    if model is None:
        raise ModelCommandError("model NAME is required. Example: clawcodex model use claude-sonnet-4-6")
    return model, provider
