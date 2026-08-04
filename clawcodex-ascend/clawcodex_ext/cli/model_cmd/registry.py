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

"""Provider/model metadata helpers for F-43 commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from clawcodex_ext.cli.model_cmd.errors import (
    AmbiguousModelError,
    ProviderMismatchError,
    UnknownModelError,
)
from clawcodex_ext.cli.provider_cmd.errors import UnknownProviderError


# ---- Dynamic model discovery hooks ----
# Extension code can register discovery callables at import time.
# `ModelRegistry.available_models()` merges the static list with
# all registered hook results (best-effort, de-duplicated).
_DISCOVERY_HOOKS: dict[str, list[Callable[[], list[str]]]] = {}


def register_discovery_hook(provider: str, hook: Callable[[], list[str]]) -> None:
    """Register a callable that returns extra models for *provider* at runtime.

    The hook is called each time ``available_models()`` is invoked for this
    provider.  Exceptions are silently swallowed (best-effort discovery).
    Hooks registered after a ``ModelRegistry`` instance is created affect
    **all** future calls — the instance reads the global registry lazily.
    """
    _DISCOVERY_HOOKS.setdefault(provider, []).append(hook)


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    label: str
    default_model: str
    configured_model: str | None
    authenticated: bool | None
    auth_detail: str | None = None


class ModelRegistry:
    """Wrap built-in provider metadata with validation helpers."""

    def __init__(
        self,
        provider_info: dict[str, Any] | None = None,
        discovery_hooks: dict[str, list[Callable[[], list[str]]]] | None = None,
    ) -> None:
        if provider_info is None:
            from clawcodex_ext.providers import ensure_provider_extensions_installed

            ensure_provider_extensions_installed()
            from src.providers import PROVIDER_INFO

            provider_info = PROVIDER_INFO
        self.provider_info = provider_info
        # If no custom hooks dict, reference the global registry so any
        # import-time registration is visible to every ModelRegistry instance.
        self._discovery_hooks = _DISCOVERY_HOOKS if discovery_hooks is None else discovery_hooks

    def _provider_key(self, provider: str) -> str:
        """Resolve aliases while preserving names in custom test registries."""
        from src.providers import canonical_provider_name

        canonical = canonical_provider_name(provider)
        if canonical in self.provider_info:
            return canonical
        return provider

    def provider_names(self) -> list[str]:
        return list(self.provider_info.keys())

    def validate_provider(self, provider: str) -> str:
        provider_key = self._provider_key(provider)
        if provider_key not in self.provider_info:
            raise UnknownProviderError(provider)
        return provider_key

    def provider_default_model(self, provider: str) -> str:
        provider = self.validate_provider(provider)
        return self.provider_info[provider]["default_model"]

    def configured_models(self, provider: str) -> list[str]:
        """Return the provider's local catalog without running discovery hooks."""
        provider = self.validate_provider(provider)
        return list(self.provider_info[provider].get("available_models", []))

    def available_models(self, provider: str) -> list[str]:
        provider = self.validate_provider(provider)
        baseline = self.configured_models(provider)
        for hook in self._discovery_hooks.get(provider, []):
            try:
                extra = hook()
                for m in extra:
                    if m not in baseline:
                        baseline.append(m)
            except Exception:  # nosec B110
                pass  # best-effort discovery — never fails the caller
        return baseline

    def validate_model(self, model: str, provider: str) -> str:
        provider = self.validate_provider(provider)
        if model in self.available_models(provider):
            return model
        if any(model in self.available_models(name) for name in self.provider_names()):
            raise ProviderMismatchError(model, provider)
        raise UnknownModelError(model)

    def infer_provider_for_model(self, model: str, *, discover: bool = True) -> str:
        """Return the unique provider whose catalog contains a model.

        Args:
            model: Model identifier to locate.
            discover: Whether to include live discovery hooks. Disable this
                on startup and persistence paths that must remain local.

        Returns:
            The unique matching provider identifier.
        """

        models_for = self.available_models if discover else self.configured_models
        matches = [provider for provider in self.provider_names() if model in models_for(provider)]
        if not matches:
            raise UnknownModelError(model)
        if len(matches) > 1:
            raise AmbiguousModelError(model, matches)
        return matches[0]

    def find_prefix_matches(self, prefix: str, provider: str | None = None) -> list[tuple[str, str]]:
        """Return ``(model, provider)`` tuples whose model name starts with ``prefix``.

        When ``provider`` is given, only that provider is searched; otherwise
        every known provider is searched.  The empty prefix yields no matches.
        Used by ``/model`` to auto-correct short names like ``sonnet`` →
        ``claude-sonnet-4-6`` when no exact match exists.
        """
        if not prefix:
            return []
        providers = [provider] if provider else self.provider_names()
        matches: list[tuple[str, str]] = []
        for prov in providers:
            try:
                self.validate_provider(prov)
            except UnknownProviderError:
                continue
            for model in self.available_models(prov):
                if model.startswith(prefix) and model != prefix:
                    matches.append((model, prov))
        return matches

    def suggest_models(self, name: str, provider: str | None = None, n: int = 3) -> list[str]:
        """Return up to *n* close-matching model names for "Did you mean ...?".

        Compares *name* against each model's first dash-separated segment
        ("family name" — e.g. ``claude-sonnet-4-6`` -> ``claude``) so single-
        word typos like ``cluade`` -> ``claude-...`` surface with high
        SequenceMatcher ratios.  Full model names are then expanded in
        registry order, deduplicated, and truncated to *n*.

        When ``provider`` is given, only that provider's models are searched;
        otherwise the entire registry is searched.
        """
        import difflib

        if not name:
            return []
        providers = [provider] if provider else self.provider_names()

        # Build short-name -> full-model list mapping.
        short_to_fulls: dict[str, list[str]] = {}
        for prov in providers:
            try:
                self.validate_provider(prov)
            except UnknownProviderError:
                continue
            for model in self.available_models(prov):
                short = model.split("-", 1)[0] if "-" in model else model
                short_to_fulls.setdefault(short, []).append(model)

        short_candidates = list(short_to_fulls.keys())
        short_matches = difflib.get_close_matches(name, short_candidates, n=n, cutoff=0.6)

        # Expand short matches back to full model names (preserve registry order).
        full_suggestions: list[str] = []
        for short in short_matches:
            for full in short_to_fulls[short]:
                if full not in full_suggestions:
                    full_suggestions.append(full)
                    if len(full_suggestions) >= n:
                        return full_suggestions
        return full_suggestions

    def provider_statuses(self) -> list[ProviderStatus]:
        from src.config import get_provider_config

        statuses: list[ProviderStatus] = []
        for name, info in self.provider_info.items():
            configured_model = None
            authenticated: bool | None = None
            auth_detail = None
            try:
                cfg = get_provider_config(name)
                configured_model = cfg.get("default_model")
                if name == "openai-codex":
                    try:
                        from src.auth.codex_oauth import get_codex_auth_status

                        status = get_codex_auth_status()
                        authenticated = status.is_authenticated
                        auth_detail = status.error or status.source
                    except Exception as exc:
                        authenticated = False
                        auth_detail = str(exc)
                else:
                    authenticated = bool(cfg.get("api_key"))
            except Exception:
                authenticated = False

            statuses.append(
                ProviderStatus(
                    name=name,
                    label=info["label"],
                    default_model=info["default_model"],
                    configured_model=configured_model,
                    authenticated=authenticated,
                    auth_detail=auth_detail,
                )
            )
        return statuses
