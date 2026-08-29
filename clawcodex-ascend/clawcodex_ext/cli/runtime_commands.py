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

"""Runtime slash commands for provider/model switching.

Both ``/provider`` and ``/model`` share a unified surface that mirrors the
CLI subcommands in :mod:`clawcodex_ext.cli.provider_cmd` and
:mod:`clawcodex_ext.cli.model_cmd`:

* ``/provider`` (no args)            — show current provider + list all
* ``/provider <NAME>``               — switch to ``<NAME>``
* ``/model``    (no args)            — show current provider/model + list all
* ``/model <NAME> [--provider P]``   — switch to ``<NAME>`` (inferred or
  explicit provider)

The legacy ``list`` / ``current`` / ``use <NAME>`` subcommand spellings are
no longer recognised — they were folded into the unified form so the
slash command behaves identically in REPL and TUI.
"""

from __future__ import annotations

import re
from typing import Any

from clawcodex_ext.cli.model_cmd.registry import ModelRegistry
from clawcodex_ext.cli.model_cmd.store import ModelStore
from clawcodex_ext.cli.model_cmd.errors import UnknownModelError, ProviderMismatchError
from clawcodex_ext.cli.provider_cmd.commands import format_provider_list
from clawcodex_ext.cli.provider_cmd.errors import UnknownProviderError
from src.command_system.types import LocalCommand, LocalCommandResult

# ── OKLCH-based markup helpers for REPL provider/model output ──────────
# Semantic names registered in clawcodex_ext/repl/color_scheme.py's
# build_rich_theme() — key_label (amber), value_text (purple),
# version_num (sky-blue), primary (blue).

_VERSION_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?)")


def _colorize_model_name(model: str | None) -> str:
    """Wrap version numbers in ``[version_num]…[/version_num]`` markup.

    Handles *None* gracefully (unknown provider case) so callers don't
    need to guard against ``re.sub`` ``TypeError``.
    """
    if model is None:
        return "[dim]unknown[/dim]"
    return _VERSION_RE.sub(r"[version_num]\1[/version_num]", model)


def _colorize_provider_list_text(plain: str) -> str:
    """Add Rich markup to ``format_provider_list()`` output.

    Transforms ``  name\\tlabel\\tmodel=X\\tconfigured=Y`` lines into
    OKLCH-coloured Rich markup using the theme from ``build_rich_theme()``.
    """
    lines = plain.split("\n")
    out: list[str] = []
    for line in lines:
        # "Providers:" header — keep as-is
        if not line.startswith(" "):
            out.append(line)
            continue
        # Data lines: "  name\\tlabel\\tmodel=value\\tconfigured=value"
        parts = line.split("\t")
        if len(parts) >= 4:
            # Provider name → primary (blue)
            name_col = f"[primary]{parts[0].strip()}[/primary]"
            label_col = parts[1]  # untouched
            # model=… key-label + value
            model_str = _colorize_model_name(parts[2])
            model_kv = re.sub(
                r"^(\w+)=(.+)$",
                r"[key_label]\1=[/key_label][value_text]\2[/value_text]",
                model_str,
            )
            # configured=yes/no
            configured_kv = re.sub(
                r"^(\w+)=(.+)$",
                r"[key_label]\1=[/key_label][value_text]\2[/value_text]",
                parts[3],
            )
            out.append(f"  {name_col}\t{label_col}\t{model_kv}\t{configured_kv}")
        else:
            out.append(line)
    return "\n".join(out)


def register_runtime_commands(registry: Any | None = None) -> None:
    from src.command_system.registry import get_command_registry

    reg = registry or get_command_registry()
    for command in (_provider_command(), _model_command()):
        reg.register(command)


def format_model_list(provider: str | None = None) -> str:
    return _format_configured_model_list(provider)


def _format_runtime_model_list(context: Any) -> str:
    """Render models reported by the active provider instance.

    ``/model`` is a session command, so its source of truth must be the
    provider that will handle the next request rather than the process-wide
    static registry.  Registry/config data remains an explicit fallback when
    the provider cannot refresh its catalog.
    """
    runtime = _runtime(context)
    provider = getattr(runtime, "provider", None) if runtime is not None else None
    provider = provider or getattr(context, "provider", None)
    provider_name = _current_provider_name(context) or "unknown"
    current = getattr(provider, "model", None)

    try:
        from clawcodex_ext.providers.model_catalog_cache import get_model_catalog

        snapshot = get_model_catalog(provider_name, provider)
        models = list(snapshot.models)
    except Exception as exc:
        fallback = _mark_current_model(format_model_list(provider=provider_name), current=current)
        return f"Model discovery failed for {provider_name}: {exc}\nShowing configured fallback models.\n\n{fallback}"

    # When the caller has no runtime context, the snapshot only contains the
    # placeholder "unknown" provider with the SyntheticProvider's mock models.
    # Fall back to the full registry listing so the user sees every
    # configured provider instead of a single bogus "unknown:" section.
    # Checked BEFORE the empty-models early-return so both paths are covered.
    if provider_name == "unknown":
        return format_model_list(provider=None)

    if current and current not in models:
        models.insert(0, current)
    if not models:
        fallback = _mark_current_model(format_model_list(provider=provider_name), current=current)
        return (
            f"Model discovery returned no models for {provider_name}.\n"
            "Showing configured fallback models.\n\n"
            f"{fallback}"
        )

    lines = []
    if snapshot.error:
        shown = "cached" if snapshot.source == "stale-cache" else "configured fallback"
        lines.extend(
            [
                f"Last model catalog refresh failed for {provider_name}: {snapshot.error}; showing {shown} models.",
                "",
            ]
        )
    elif snapshot.refreshing:
        shown = "cached" if snapshot.source == "stale-cache" else "configured fallback"
        lines.extend(
            [
                f"Model catalog refresh is running in the background; showing {shown} models.",
                "",
            ]
        )
    lines.extend(["Models:", f"  {provider_name}:"])
    for model in models:
        marker = " *" if model == current else ""
        lines.append(f"    {model}{marker}")
    return "\n".join(lines)


def _mark_current_model(model_list: str, *, current: str | None) -> str:
    if not current:
        return model_list
    lines = model_list.splitlines()
    found = False
    for index, line in enumerate(lines):
        if not line.startswith("    "):
            continue
        model = line.strip().removesuffix(" *")
        if model == current:
            lines[index] = f"    {model} *"
            found = True
        elif line.endswith(" *"):
            lines[index] = line[:-2]
    if not found:
        lines.append(f"    {current} *")
    return "\n".join(lines)


def _format_configured_model_list(provider: str | None = None) -> str:
    """Show models only for providers with API keys in config.

    Reads ``~/.clawcodex/config.json`` and filters the full model list
    to only those providers the user has actually configured.  Falls
    back to all known providers when no config is available (e.g. in
    tests or CI).

    For each configured provider, the displayed model list is the *union*
    of:
    1. Models known to the built-in ``ModelRegistry`` (``available_models``)
    2. Models recorded in the config's ``models`` list (user's custom /
       previously-used models outside the built-in registry)
    """
    registry = ModelRegistry(discovery_hooks={})
    try:
        from src.config import get_provider_config

        configured = [
            name
            for name in registry.provider_names()
            if (provider is None or name == provider) and get_provider_config(name)
        ]
        # Also include providers that exist only in config but aren't
        # known to ModelRegistry (e.g. completely custom providers).
        import json
        from src.config import get_global_config_path

        gp = get_global_config_path()
        if gp and gp.exists():
            raw = json.loads(gp.read_text())
            for p in raw.get("providers") or {}:
                if (provider is None or p == provider) and p not in configured and p not in registry.provider_names():
                    configured.append(p)
    except Exception:
        configured = []

    if not configured:
        # No config or all empty — fall back to showing all known providers
        # so the user always sees something.
        configured = [provider] if provider is not None else list(registry.provider_names())

    lines = ["Models:"]
    for provider_name in configured:
        try:
            registry.validate_provider(provider_name)
        except Exception:  # nosec B110
            pass  # Custom provider not in registry — still show it.

        reg_models: list[str] = []
        try:
            reg_models = list(registry.configured_models(provider_name) or [])
        except Exception:  # nosec B110
            pass  # This probe is optional; preserve the existing conservative fallback.

        # Merge in models from config's ``models`` list
        cfg_models: list[str] = []
        default_model: str | None = None
        try:
            from src.config import get_provider_config

            pc = get_provider_config(provider_name)
            if pc:
                cfg_models = list(pc.get("models", []) or [])
                default_model = pc.get("default_model")
        except Exception:  # nosec B110
            pass  # This probe is optional; preserve the existing conservative fallback.

        if default_model is None:
            try:
                default_model = registry.provider_default_model(provider_name)
            except Exception:  # nosec B110
                pass  # This probe is optional; preserve the existing conservative fallback.

        all_models = list(dict.fromkeys(reg_models + cfg_models))  # dedup, preserve order

        lines.append(f"  {provider_name}:")
        for model in all_models:
            marker = " *" if model == default_model else ""
            lines.append(f"    {model}{marker}")
    return "\n".join(lines)


_MODEL_USAGE = (
    "Usage: /model [NAME [--provider NAME]]\n\n"
    "Modes:\n"
    "  (no args)              Show current provider/model + available models.\n"
    "  NAME                   Switch to model NAME. Unique short prefixes auto-\n"
    "                         match (e.g. 'gpt-4o-m' -> 'gpt-4o-mini'); ambiguous\n"
    "                         prefixes list candidates; on typo, suggests\n"
    "                         'Did you mean ...?'.\n"
    "  NAME --provider P      Switch to NAME under provider P (skip inference).\n"
    "  help, --help, -h       Print this help.\n\n"
    "Persistence:\n"
    "  Known model     runtime only - config unchanged.\n"
    "  Unknown model   saved to config.json; will survive restart.\n"
    "  If save fails   session only; not persisted.\n"
)


def _provider_command() -> LocalCommand:
    command = LocalCommand(
        name="provider",
        description="Show current provider (and available list), or switch to a named provider",
        argument_hint="[NAME]",
    )
    command.set_call(_provider_call)
    return command


def _model_command() -> LocalCommand:
    command = LocalCommand(
        name="model",
        description="Show current model (and available list), or switch to a named model",
        argument_hint="[NAME [--provider NAME]]",
    )
    command.set_call(_model_call)
    return command


def _provider_call(args: str, context: Any) -> LocalCommandResult:
    tokens = args.split()

    if not tokens:
        current = _format_runtime_current(context)
        plain_list = format_provider_list()
        colored_list = _colorize_provider_list_text(plain_list)
        lines = [current, "", colored_list] if current else [colored_list]
        return _text("\n".join(lines))

    provider = tokens[0]
    registry = ModelRegistry(discovery_hooks={})
    try:
        provider = registry.validate_provider(provider)
    except UnknownProviderError:
        pass  # Registry validation is optional here; preserve the configured-name fallback.
    runtime = _runtime(context)
    if runtime is None:
        return _text("Runtime context is not available — cannot switch provider.")

    warnings: list[str] = []
    try:
        ModelStore().set_default_provider(provider)
    except UnknownProviderError:
        warnings.append(f"Warning: unknown provider '{provider}' — proceeding anyway")
        from src.config import set_default_provider as _set_dp

        _set_dp(provider)

    try:
        runtime.swap_provider(provider)
    except Exception as exc:
        return _text(
            f"Failed to switch to provider '{provider}': {exc}\n"
            "Check that the provider is configured (api_key, base_url) "
            "and try again."
        )

    _sync_context(context, runtime)

    lines = [f"Provider switched to: {provider}"]
    lines.extend(warnings)
    lines.append("")
    current = _format_runtime_current(context)
    if current is not None:
        lines.append(current)
    return _text("\n".join(lines))


def _model_call(args: str, context: Any) -> LocalCommandResult:
    tokens = args.split()

    # Help subcommand - first-token check so /model --help still shows help.
    if tokens and tokens[0] in ("help", "--help", "-h"):
        return _text(_MODEL_USAGE)

    if not tokens:
        current = _format_runtime_current(context)
        model_list = _format_runtime_model_list(context)
        lines = [current, "", model_list] if current else [model_list]
        return _text("\n".join(lines))

    try:
        model, provider = _parse_model_args(tokens)
    except ValueError as exc:
        return _text(f"usage: /model [NAME [--provider NAME]]\n{exc}")

    warnings: list[str] = []
    # Switching is session-scoped and the provider API is authoritative.
    # Avoid discovery hooks here: they would turn `/model <name>` into a
    # blocking network operation before the runtime can even try the model.
    registry = ModelRegistry(discovery_hooks={})

    # ---- Resolve provider ----
    if provider is None:
        # Use current runtime provider instead of infer_provider_for_model,
        # so /model switches models under the user's current provider rather
        # than silently jumping to whichever provider "owns" the model name.
        provider = _current_provider_name(context) or "anthropic"
    try:
        provider = registry.validate_provider(provider)
    except UnknownProviderError:
        pass  # Registry validation is optional here; preserve the configured-name fallback.

    # ---- Try prefix matching / spelling suggestions on validation failure ----
    # Unique prefix matches are auto-corrected with a Note; multiple matches
    # are listed for the user to choose from; zero matches fall through to
    # ``Did you mean ...?`` suggestions from difflib.
    try:
        registry.validate_model(model, provider)
    except (UnknownModelError, ProviderMismatchError):
        prefix_matches = registry.find_prefix_matches(model, provider)
        if len(prefix_matches) == 1:
            resolved_model, resolved_provider = prefix_matches[0]
            warnings.append(f"Note: matched '{model}' by prefix to '{resolved_model}'")
            model = resolved_model
            provider = resolved_provider
        elif len(prefix_matches) > 1:
            options = ", ".join(m[0] for m in prefix_matches)
            warnings.append(f"Multiple models start with '{model}': {options} - please be more specific")
        else:
            suggestions = registry.suggest_models(model, provider)
            if suggestions:
                warnings.append(f"Did you mean: {', '.join(suggestions)}?")

    # ---- Validate and warn about unknown models ----
    # /model is session-scoped only — it never modifies the persisted config.
    # For persistent changes, use ``clawcodex model use <name>`` from the CLI.
    try:
        registry.validate_model(model, provider)
    except (UnknownModelError, ProviderMismatchError):
        warnings.append(
            f"Unknown model '{model}' — proceeding anyway (session only; use 'clawcodex model use' to persist)"
        )

    # ---- Runtime switch (session-scoped only) ----
    runtime = _runtime(context)
    if runtime is None:
        return _text("Runtime context is not available — cannot switch model.")
    runtime.swap_provider(provider, model)  # type: ignore[union-attr]
    _sync_context(context, runtime)
    lines = [f"Model switched to: {model} (provider: {provider})"]
    lines.extend(warnings)
    lines.append("")
    current = _format_runtime_current(context)
    if current is not None:
        lines.append(current)
    return _text("\n".join(lines))


def _parse_provider_flag(tokens: list[str]) -> str | None:
    provider = None
    idx = 0
    while idx < len(tokens):
        if tokens[idx] == "--provider" and idx + 1 < len(tokens):
            provider = tokens[idx + 1]
            idx += 2
            continue
        raise ValueError(f"Unknown argument: {tokens[idx]}")
    return provider


def _parse_model_args(tokens: list[str]) -> tuple[str, str | None]:
    model = tokens[0]
    provider = _parse_provider_flag(tokens[1:])
    return model, provider


def _runtime(context: Any) -> Any | None:
    """Return the runtime context, or *None* if not available.

    Callers that only *display* current state (no-arg ``/model`` /
    ``/provider``) should tolerate *None* and omit the current-state
    line.  Callers that *mutate* state (``/model <name>``) raise
    ``ValueError`` themselves when they need ``swap_provider``.
    """
    return getattr(context, "runtime_context", None)


def _current_provider_name(context: Any) -> str | None:
    """Return the current provider name from the runtime context, or *None*.

    Used as the fallback provider when ``infer_provider_for_model`` cannot
    identify a model — avoids resetting to a hardcoded default that may
    differ from the user's actual provider.
    """
    runtime = _runtime(context)
    if runtime is not None:
        return getattr(runtime, "provider_name", None)
    # Fallback: try the command context's own provider name.
    return getattr(context, "provider_name", None) or getattr(getattr(context, "provider", None), "provider_name", None)


def _sync_context(context: Any, runtime: Any) -> None:
    context.provider = runtime.provider
    context.tool_registry = runtime.tool_registry
    context.tool_context = runtime.tool_context


def _format_runtime_current(context: Any, *, prefix: str | None = None) -> str | None:
    """Return an OKLCH-coloured ``"provider: …\nmodel: …"`` snippet with
    Rich markup, or *None* when the runtime context is missing.
    """
    runtime = _runtime(context)
    if runtime is None:
        return None
    lines = []
    if prefix:
        lines.append(prefix)
    raw_model = getattr(runtime.provider, "model", runtime.options.model)
    colored_model = _colorize_model_name(raw_model)
    lines.extend(
        [
            f"[primary]provider[/primary]: [value_text]{runtime.provider_name}[/value_text]",
            f"[primary]model[/primary]: {colored_model}",
        ]
    )
    return "\n".join(lines)


def _text(value: str) -> LocalCommandResult:
    return LocalCommandResult(type="text", value=value)
