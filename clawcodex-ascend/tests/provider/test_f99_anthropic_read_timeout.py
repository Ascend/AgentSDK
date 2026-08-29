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

"""Tests for the Anthropic read timeout."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.providers.anthropic_provider import (
    _F99_READ_TIMEOUT,
    AnthropicProvider,
)


@pytest.fixture
def fresh_provider():
    """Build a fresh AnthropicProvider for each test (no cached client)."""
    return AnthropicProvider(api_key="test-key", base_url="https://example.invalid")


@pytest.fixture
def fake_anthropic_module():
    """A MagicMock standing in for the ``anthropic`` module."""
    mod = MagicMock(name="fake_anthropic_module")
    mod.Anthropic = MagicMock(name="Anthropic class")
    sentinel = MagicMock(name="Anthropic instance")
    mod.Anthropic.return_value = sentinel
    return mod


@pytest.fixture
def patched_anthropic(fake_anthropic_module):
    """Inject the fake ``anthropic`` module into the provider's namespace.

    Patches the module-level ``anthropic`` attribute that
    ``_ensure_client`` looks up via ``sys.modules[__name__].anthropic``.
    The PEP 562 lazy ``__getattr__`` only fires when the attribute is
    MISSING; once we set it, direct attribute access wins. After the
    test we restore the original (or delete the attribute so the
    lazy loader takes over again).
    """
    import src.providers.anthropic_provider as mod

    original = getattr(mod, "anthropic", None)
    mod.anthropic = fake_anthropic_module
    yield mod, fake_anthropic_module
    if original is None:
        del mod.anthropic
    else:
        mod.anthropic = original


def test_read_timeout_constant_is_five_seconds() -> None:
    """the bound is 5s — short enough to feel instant, long enough to
    tolerate real network jitter on slow chunks.

    Pinning the constant prevents accidental drift (e.g. someone
    bumping it to 60s "to be safe" — which would defeat the whole
    fix).
    """
    assert _F99_READ_TIMEOUT == 5.0


def test_ensure_client_passes_timeout_kwarg(fresh_provider, patched_anthropic) -> None:
    """Verify ensure client passes timeout kwarg."""
    mod, fake_anthropic_module = patched_anthropic
    fresh_provider._ensure_client()
    fake_anthropic_module.Anthropic.assert_called_once()
    call_kwargs = fake_anthropic_module.Anthropic.call_args.kwargs
    assert call_kwargs.get("timeout") == _F99_READ_TIMEOUT
    # api_key still forwarded (the existing contract).
    assert call_kwargs.get("api_key") == "test-key"


def test_ensure_client_preserves_explicit_timeout(fresh_provider, patched_anthropic) -> None:
    """caller-supplied ``timeout`` overrides the default.

    If a future caller threads an ``http_client`` or custom
    ``timeout`` through ``_client_kwargs``, must not stomp on
    it. The ``if 'timeout' not in kwargs`` guard makes the
    override opt-in: callers that need the old behaviour can
    request it explicitly.
    """
    mod, fake_anthropic_module = patched_anthropic
    # Inject a custom timeout via _client_kwargs as a future caller
    # might do (e.g. for SSE streaming with longer chunks).
    fresh_provider._client_kwargs["timeout"] = 30.0
    fresh_provider._ensure_client()
    call_kwargs = fake_anthropic_module.Anthropic.call_args.kwargs
    assert call_kwargs.get("timeout") == 30.0


def test_ensure_client_preserves_explicit_http_client(fresh_provider, patched_anthropic) -> None:
    """caller-supplied ``http_client`` wins over the timeout.

    A caller that builds their own httpx client (e.g. with proxy,
    SSL context, or telemetry hooks) wants to stay out of the
    way. The ``if 'http_client' not in kwargs`` guard ensures the
    timeout is only applied when the SDK is responsible for
    building its own httpx client.
    """
    mod, fake_anthropic_module = patched_anthropic
    custom_http = MagicMock(name="custom httpx client")
    fresh_provider._client_kwargs["http_client"] = custom_http
    fresh_provider._ensure_client()
    call_kwargs = fake_anthropic_module.Anthropic.call_args.kwargs
    # When http_client is supplied, must NOT also supply
    # timeout — the user's client owns its own timeout config.
    assert "timeout" not in call_kwargs
    assert call_kwargs.get("http_client") is custom_http


def test_ensure_client_caches_client(fresh_provider, patched_anthropic) -> None:
    """subsequent calls return the cached client.

    The existing cache contract (set ``self.client`` once, return
    the same instance) must be preserved by the fix. This pins
    that we don't accidentally rebuild the client per request.
    """
    mod, fake_anthropic_module = patched_anthropic
    c1 = fresh_provider._ensure_client()
    c2 = fresh_provider._ensure_client()
    assert c1 is c2
    # Anthropic() constructor called exactly once.
    assert fake_anthropic_module.Anthropic.call_count == 1


def test_ensure_client_forwards_base_url(fresh_provider, patched_anthropic) -> None:
    """``base_url`` (and any other ``_client_kwargs`` keys) still forwarded.

    Regression guard — the fix only adds a default ``timeout``
    kwarg; existing keys must still reach the constructor so the
    proxy / custom-endpoint flow keeps working.
    """
    mod, fake_anthropic_module = patched_anthropic
    fresh_provider._ensure_client()
    call_kwargs = fake_anthropic_module.Anthropic.call_args.kwargs
    assert call_kwargs.get("base_url") == "https://example.invalid"
