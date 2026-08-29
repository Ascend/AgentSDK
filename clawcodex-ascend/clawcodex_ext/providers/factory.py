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

# pylint: disable=cyclic-import
"""Extension provider construction and registration.

Moved from ``src/providers/__init__.py`` so that the upstream package
remains a clean registry of built-in providers.  All call sites that
need ``create_provider``, ``register_provider``, or
``register_provider_info`` should import from this module (or via the
facade re-exports in ``src/providers/__init__.py``).

Architecture::

    src/providers/__init__.py                       ← upstream built-in registry (get_provider_class, PROVIDER_INFO)
        ↑ import                                    ← _EXTRA_PROVIDER_CLASSES dict lives here
    clawcodex_ext/providers/factory.py              ← extension factory and registration
        ↑ import
    clawcodex_ext/providers/_litellm_adapter.py     ← LiteLLM fallback provider (canonical)
    extensions/providers_ext/                       ← deprecated re-export shim (backward compat)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from clawcodex_ext.providers.base import BaseProvider

if TYPE_CHECKING:
    from src.providers import ProviderInfo


# See the ``__getattr__`` and ``__dir__`` implementation below.
# ``_EXTRA_PROVIDER_CLASSES``, ``PROVIDER_INFO``, and ``get_provider_class``
# are imported lazily by callers to avoid this cycle:
# ``src.providers`` (through the openai_compatible facade) →
# ``clawcodex_ext.providers`` → ``factory`` →
# ``from src.providers import _EXTRA_PROVIDER_CLASSES``.


# ---------------------------------------------------------------------------
# LiteLLM switch
# ---------------------------------------------------------------------------


def should_use_litellm() -> bool:
    """Return whether runtime provider creation should use LiteLLM."""
    from os import getenv

    return getenv("CLAW_USE_LITELLM", "").lower() in {"1", "true", "yes", "on"}


def get_provider_class(provider_name: str):
    """Compatibility indirection retained for callers/tests that patch it."""
    from src.providers import get_provider_class as resolve_provider_class

    return resolve_provider_class(provider_name)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def create_provider(provider_name: str, *args, **kwargs) -> BaseProvider:
    """Create a provider instance for runtime use."""
    if should_use_litellm():
        from clawcodex_ext.providers._litellm_adapter import create_litellm_provider

        return create_litellm_provider(provider_name, *args, **kwargs)

    try:
        provider_cls = get_provider_class(provider_name)
    except ValueError:
        # Unknown provider — fallback to LiteLLM
        from clawcodex_ext.providers._litellm_adapter import create_litellm_provider

        return create_litellm_provider(provider_name, *args, **kwargs)

    return provider_cls(*args, **kwargs)


# ---------------------------------------------------------------------------
# Extension registration API
# ---------------------------------------------------------------------------


def register_provider(name: str, info: "ProviderInfo", cls: type | callable) -> None:
    """Register a new provider at runtime.

    Adds *info* to ``PROVIDER_INFO`` and registers *cls* so that
    ``get_provider_class(name)`` returns it.

    *cls* may be a ``BaseProvider`` subclass **or** a zero-arg callable
    that returns one.  The callable form supports lazy imports — it is
    invoked the first time ``get_provider_class(name)`` is called,
    and the result replaces the callable in the registry so subsequent
    lookups are direct.

    Idempotent: calling twice with the same *name* is a no-op
    (first registration wins).
    """
    from src.providers import EXTRA_PROVIDER_CLASSES as _EXTRA_PROVIDER_CLASSES

    register_provider_info(name, info)
    if name not in _EXTRA_PROVIDER_CLASSES:
        _EXTRA_PROVIDER_CLASSES[name] = cls


def register_provider_info(name: str, info: "ProviderInfo") -> None:
    """Add or update *info* in ``PROVIDER_INFO`` without a class mapping.

    Useful when the provider is served by LiteLLM or another generic
    backend that doesn't have a dedicated ``BaseProvider`` subclass.

    Also refreshes ``AVAILABLE_PROVIDERS`` so the new provider shows up
    in UI/CLI listings.
    """
    from src.providers import PROVIDER_INFO

    if name not in PROVIDER_INFO:
        PROVIDER_INFO[name] = info
        # Refresh the display dict so it reflects the new provider.
        # Use in-place update so that existing ``from src.providers import
        # AVAILABLE_PROVIDERS`` references (which bound at import time)
        # see the change.
        import src.providers as _pkg

        _pkg.AVAILABLE_PROVIDERS[name] = info["label"]


__all__ = [
    "create_provider",
    "get_provider_class",
    "should_use_litellm",
    "register_provider",
    "register_provider_info",
]


# fix (defense-in-depth): trigger the cancel-latency provider
# registration as a side-effect of importing this module. We place the
# import at the bottom (after ``register_provider`` /
# ``register_provider_info`` are defined) so that
# ``clawcodex_ext/providers/__init__.py`` can import them without a
# circular-import failure. Every code path that builds a provider
# (REPL, headless, TUI, agent background runner, CLI subcommands)
# imports this module either directly or transitively via
# ``src.providers.runtime``, so the side-effect import guarantees the
# ``ClawcodexAnthropicProvider`` / ``ClawcodexMinimaxProvider``
# overrides are registered in ``_EXTRA_PROVIDER_CLASSES`` before any
# ``get_provider_class(...)`` lookup.
#
# Without this, the bare upstream ``AnthropicProvider`` is used and
# a Ctrl+C waits the full platform socket timeout (~60s) because its
# only cancellation mechanism is the advisory
# ``response.close()`` plus our ``_close_transport_safely`` helper
# (which is a silent no-op on ``httpx.Response`` since the
# ``_transport`` attribute lives on the client, not the response).
import clawcodex_ext.providers as _providers_ext  # noqa: E402

_providers_ext._init_provider_extensions()
