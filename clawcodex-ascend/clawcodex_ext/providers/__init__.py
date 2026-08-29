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
"""Downstream provider extensions — model discovery hooks and provider overrides.

All side-effect code (``_extend_builtin_models()``, ``register_provider()``
calls, media registry wiring) is deferred to :func:`_init_provider_extensions`,
called from ``clawcodex_ext/__init__.py:ensure_eager_extensions_installed()``
after all ``src/`` modules are fully loaded.  This avoids a circular-import
chain::

    src.providers  [mid-load]
      → openai_compatible (facade)
        → clawcodex_ext.providers.openai_compatible
          → clawcodex_ext/providers/__init__.py  [loaded here]
            → from src.providers import …  ← src.providers partially initialized!
"""

from __future__ import annotations


def _extend_builtin_models(provider_info: dict) -> None:
    """Extend built-in provider model lists with downstream additions.

    Idempotent: on second call the ``extend`` would be a no-op because
    the upstream models are already present (the extended lists are
    supersets of the upstream baseline).
    """
    # Anthropic — add OpenRouter-style paths + newer variants
    anthropic_models = provider_info.setdefault("anthropic", {}).setdefault("available_models", [])
    _anthropic_extras = [
        "anthropic/claude-3.5-haiku",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-opus-4.1",
        "anthropic/claude-sonnet-4.5",
    ]
    for m in _anthropic_extras:
        if m not in anthropic_models:
            anthropic_models.append(m)

    # OpenAI — add OpenRouter-style paths + newer variants
    openai_models = provider_info.setdefault("openai", {}).setdefault("available_models", [])
    _openai_extras = [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-5",
        "openai/gpt-5-mini",
    ]
    for m in _openai_extras:
        if m not in openai_models:
            openai_models.append(m)

    # Z.ai / GLM — add OpenRouter-style paths. Upstream renamed the ``glm``
    # provider to ``zai`` (``glm`` is now only a ``PROVIDER_ALIASES`` spelling),
    # so extend the canonical ``zai`` entry. Guard on existence so we never
    # fabricate a label-less stub entry that would break ``AVAILABLE_PROVIDERS``.
    glm_models = provider_info.get("zai", provider_info.get("glm", {})).setdefault("available_models", [])
    _glm_extras = [
        "zai/glm-3-turbo",
        "zai/glm-4",
        "zai/glm-4-air",
        "zai/glm-4-flash",
        "zai/glm-4-plus",
        "zai/glm-4.5",
        "zai/glm-4.6",
        "zai/glm-4.7",
        "zai/glm-5",
        "zai/glm-5-turbo",
    ]
    for m in _glm_extras:
        if m not in glm_models:
            glm_models.append(m)

    # DeepSeek — add OpenRouter-style paths + downstream variants
    deepseek_models = provider_info.setdefault("deepseek", {}).setdefault("available_models", [])
    _deepseek_extras = [
        "deepseek/deepseek-chat-v3.1",
        "deepseek/deepseek-r1-0528",
        "deepseek/deepseek-v3.1-terminus",
        "deepseek/deepseek-v3.2",
        "deepseek/deepseek-v3.2-speciale",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
    ]
    for m in _deepseek_extras:
        if m not in deepseek_models:
            deepseek_models.append(m)

    # Gemini — add OpenRouter-style paths
    gemini_models = provider_info.setdefault("gemini", {}).setdefault("available_models", [])
    _gemini_extras = [
        "google/gemini-2.0-flash",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
    ]
    for m in _gemini_extras:
        if m not in gemini_models:
            gemini_models.append(m)

    # Minimax — add M3 series (current default in ~/.clawcodex/config.json).
    # The upstream PROVIDER_INFO entry only lists the M2.x line; M3 ships as a
    # downstream-only variant served by the same Anthropic-compatible endpoint.
    minimax_models = provider_info.setdefault("minimax", {}).setdefault("available_models", [])
    _minimax_extras = [
        "MiniMax-M3",
        "MiniMax-M3-highspeed",
    ]
    for m in _minimax_extras:
        if m not in minimax_models:
            minimax_models.append(m)


# ---------------------------------------------------------------------------
# Lazy-accessor functions (defined at module level, called from
# ``_init_provider_extensions`` below).  Each returns a provider class
# without importing it until the accessor is first invoked.
# ---------------------------------------------------------------------------


def _OpenAICodexProvider_lazy():
    from clawcodex_ext.providers.openai_codex_provider import OpenAICodexProvider

    return OpenAICodexProvider


def _KimiProvider_lazy():
    from clawcodex_ext.providers.kimi_provider import KimiProvider

    return KimiProvider


def _KimiCodingProvider_lazy():
    from clawcodex_ext.providers.kimi_coding_provider import KimiCodingProvider

    return KimiCodingProvider


def _GLMProvider_lazy():
    from src.providers.glm_provider import GLMProvider

    return GLMProvider


def _ClawcodexAnthropicProvider_lazy():
    from clawcodex_ext.providers.anthropic_provider import ClawcodexAnthropicProvider

    return ClawcodexAnthropicProvider


def _ClawcodexMinimaxProvider_lazy():
    from clawcodex_ext.providers.minimax_provider import ClawcodexMinimaxProvider

    return ClawcodexMinimaxProvider


def _AgnesImageProvider_lazy():
    from clawcodex_ext.providers.media.image.agnes import AgnesImageProvider

    return AgnesImageProvider


def _AgnesVideoProvider_lazy():
    from clawcodex_ext.providers.media.video.agnes import AgnesVideoProvider

    return AgnesVideoProvider


# ---------------------------------------------------------------------------
# Lazy init — deferred from package-import to ``ensure_eager_extensions``
# ---------------------------------------------------------------------------

_provider_extensions_initialized: bool = False


def ensure_provider_extensions_installed() -> None:
    """Install only provider metadata, factories, and discovery hooks.

    Provider/model CLI commands use this lightweight bootstrap so listing or
    selecting a model does not import unrelated orchestrator and tool-system
    extensions.
    """
    _init_provider_extensions()


def _init_provider_extensions() -> None:
    """Register downstream providers, model extensions, and media providers.

    Idempotent.  Must be called after ``src.providers`` is fully loaded
    (i.e. from ``ensure_eager_extensions_installed()`` in
    ``clawcodex_ext/__init__.py``, NOT at package import time) to avoid
    the ``src.providers`` partial-init circular import.
    """
    global _provider_extensions_initialized
    if _provider_extensions_initialized:
        return
    _provider_extensions_initialized = True

    from src.providers import PROVIDER_INFO

    _extend_builtin_models(PROVIDER_INFO)

    from clawcodex_ext.providers.hooks import _codex_api_discovery
    from clawcodex_ext.cli.model_cmd.registry import register_discovery_hook  # pylint: disable=no-name-in-module

    register_discovery_hook("openai-codex", _codex_api_discovery)

    from clawcodex_ext.providers.factory import register_provider, register_provider_info

    # Register openai-codex
    register_provider(
        "openai-codex",
        {
            "label": "OpenAI Codex (ChatGPT OAuth)",
            "default_base_url": "https://chatgpt.com/backend-api/codex",
            "default_model": "gpt-5.5",
            "available_models": [
                "gpt-5.6-sol",
                "gpt-5.6-terra",
                "gpt-5.6-luna",
                "gpt-5.5",
                "gpt-5.4",
                "gpt-5.4-mini",
                "gpt-5.3-codex-spark",
                "codex-auto-review",
            ],
        },
        _OpenAICodexProvider_lazy,  # type: ignore[arg-type]
    )

    register_provider(
        "kimi",
        {
            "label": "Kimi (Moonshot AI)",
            "default_base_url": "https://api.moonshot.ai/v1",
            "default_model": "kimi-k2.6",
            "available_models": [
                "kimi-k2.6",
                "kimi-k2.5",
                "kimi-k2.7-code",
                "kimi-k2.7-code-highspeed",
                "moonshot-v1-8k",
                "moonshot-v1-32k",
                "moonshot-v1-128k",
                "moonshot-v1-8k-vision-preview",
                "moonshot-v1-32k-vision-preview",
                "moonshot-v1-128k-vision-preview",
            ],
        },
        _KimiProvider_lazy,  # type: ignore[arg-type]
    )

    register_provider(
        "kimi-coding",
        {
            "label": "Kimi Coding (Moonshot AI)",
            "default_base_url": "https://api.kimi.com/coding",
            "default_model": "kimi-code",
            "available_models": [
                "kimi-code",
            ],
        },
        _KimiCodingProvider_lazy,  # type: ignore[arg-type]
    )

    # Keep the historical native class route for direct callers without
    # registering ``glm`` as a second provider identity.  Provider/model CLI,
    # config persistence, and catalog lookup must all normalize ``glm`` to the
    # canonical upstream ``zai`` entry; ``src.providers.get_provider_class``
    # deliberately checks this literal compatibility route before aliasing.
    from src.providers import EXTRA_PROVIDER_CLASSES as _EXTRA_PROVIDER_CLASSES

    _EXTRA_PROVIDER_CLASSES.setdefault("glm", _GLMProvider_lazy)  # type: ignore[arg-type]

    # Cancel-latency overrides
    register_provider(
        "anthropic",
        PROVIDER_INFO["anthropic"],  # type: ignore[arg-type]
        _ClawcodexAnthropicProvider_lazy,  # type: ignore[arg-type]
    )
    register_provider(
        "minimax",
        PROVIDER_INFO["minimax"],  # type: ignore[arg-type]
        _ClawcodexMinimaxProvider_lazy,  # type: ignore[arg-type]
    )

    # Agnes provider info (for config system)
    register_provider_info(
        "agnes",
        {
            "label": "Agnes AI (Image + Video)",
            "default_base_url": "https://apihub.agnes-ai.com/v1",
            "default_model": "agnes-image-2.1-flash",
            "available_models": [
                "agnes-image-2.1-flash",
                "agnes-image-2.0-flash",
                "agnes-video-v2.0",
                "agnes-2.0-flash",
            ],
        },
    )

    # Media registry wiring
    from clawcodex_ext.providers.media.registry import media_registry

    media_registry.register_image("agnes", _AgnesImageProvider_lazy)
    media_registry.register_video("agnes", _AgnesVideoProvider_lazy)


def initialize_provider_extensions() -> None:
    """Install the downstream provider extensions through a public boundary."""
    _init_provider_extensions()
