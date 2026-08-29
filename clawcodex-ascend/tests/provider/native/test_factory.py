#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

"""Tests for the native provider factory (P72-D)."""

from __future__ import annotations

from typing import Any

import pytest

from src.providers.native import (
    NativeGrokProvider,
    NativeOpenAIProvider,
    create_native_provider,
    get_native_provider_class,
    registered_native_providers,
)
from src.providers.native.gemini_adapter import NativeGeminiProvider


def test_registry_includes_known_providers() -> None:
    """The native registry must include OpenAI, Gemini and Grok."""
    registry = registered_native_providers()
    assert "openai" in registry
    assert "gemini" in registry
    assert "grok" in registry
    assert registry["openai"] is NativeOpenAIProvider
    assert registry["grok"] is NativeGrokProvider
    assert registry["gemini"] is NativeGeminiProvider


def test_get_native_provider_class_unknown_returns_none() -> None:
    assert get_native_provider_class("does-not-exist") is None
    assert get_native_provider_class("") is None


def test_get_native_provider_class_returns_known_classes() -> None:
    assert get_native_provider_class("openai") is NativeOpenAIProvider
    assert get_native_provider_class("grok") is NativeGrokProvider
    assert get_native_provider_class("gemini") is NativeGeminiProvider


def test_create_native_provider_unknown_returns_none() -> None:
    """An unknown provider name yields ``None`` so the caller can
    fall back to LiteLLM without a try/except.
    """
    assert create_native_provider("not-a-provider", {}) is None


def test_create_native_provider_passes_config() -> None:
    """The factory must forward ``api_key``/``base_url``/``default_model``
    from the config dict into the adapter constructor.
    """
    captured: dict[str, Any] = {}

    class _Spy:
        def __init__(self, api_key, base_url, model):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["model"] = model
            self.api_key = api_key
            self.base_url = base_url
            self.model = model

    from src.providers.native import _NATIVE_REGISTRY

    _NATIVE_REGISTRY["spy"] = _Spy
    try:
        instance = create_native_provider(
            "spy",
            {
                "api_key": "sk-test",
                "base_url": "https://example.test/v1",
                "default_model": "spy-1",
            },
        )
        assert isinstance(instance, _Spy)
        assert captured == {
            "api_key": "sk-test",
            "base_url": "https://example.test/v1",
            "model": "spy-1",
        }
    finally:
        _NATIVE_REGISTRY.pop("spy", None)


def test_create_native_provider_handles_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the adapter's SDK is missing, the factory returns ``None``
    (the documented soft-fallback contract).
    """

    class _BoomOnInit:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("simulated missing SDK")

    from src.providers.native import _NATIVE_REGISTRY

    _NATIVE_REGISTRY["boom"] = _BoomOnInit
    try:
        assert create_native_provider("boom", {"api_key": "x"}) is None
    finally:
        _NATIVE_REGISTRY.pop("boom", None)


def test_create_native_provider_handles_init_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construction errors that aren't ``ModuleNotFoundError`` are
    still swallowed — the factory never raises to its caller.
    """

    class _BoomOnInit:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("kaboom")

    from src.providers.native import _NATIVE_REGISTRY

    _NATIVE_REGISTRY["boom2"] = _BoomOnInit
    try:
        assert create_native_provider("boom2", {"api_key": "x"}) is None
    finally:
        _NATIVE_REGISTRY.pop("boom2", None)


def test_create_native_provider_default_config_is_empty_dict() -> None:
    """``config`` is optional — omitting it must not raise."""
    from src.providers.native import _NATIVE_REGISTRY

    captured: dict[str, Any] = {}

    class _DefaultConfig:
        def __init__(self, api_key, base_url, model):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["model"] = model

    _NATIVE_REGISTRY["default"] = _DefaultConfig
    try:
        instance = create_native_provider("default")
        assert isinstance(instance, _DefaultConfig)
        # Empty string is the documented "no key" sentinel — the
        # factory should not synthesise a default.
        assert captured["api_key"] == ""
        assert captured["base_url"] is None
        assert captured["model"] is None
    finally:
        _NATIVE_REGISTRY.pop("default", None)
