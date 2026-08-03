"""F-99 方案1 tests — ``AnthropicProvider._ensure_client`` read_timeout bound.

The fix caps the blocking httpx socket read at 5s so a Ctrl+C on
platforms where ``response.close()`` is advisory (LiteLLM proxy,
some Win32 / Linux kernels) surfaces as ``httpx.ReadTimeout`` within
~5s instead of the upstream default 60s.

These tests pin the contract by patching ``anthropic`` on the
provider module and asserting the ``timeout`` kwarg is forwarded.
We don't exercise the real SDK — that would require a live API key.
"""

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
    """F-99: the bound is 5s — short enough to feel instant, long enough to
    tolerate real network jitter on slow chunks.

    Pinning the constant prevents accidental drift (e.g. someone
    bumping it to 60s "to be safe" — which would defeat the whole
    fix).
    """
    assert _F99_READ_TIMEOUT == 5.0


def test_ensure_client_passes_timeout_kwarg(fresh_provider, patched_anthropic) -> None:
    """F-99 方案1: ``_ensure_client`` forwards a ``timeout=5.0`` kwarg.

    The Anthropic SDK accepts ``timeout`` as either an ``httpx.Timeout``
    or a float (defaults applied). Pinning the kwarg rather than the
    concrete ``httpx.Timeout`` instance keeps the test stable across
    SDK upgrades where the constructor signature might rewrap the
    value.
    """
    mod, fake_anthropic_module = patched_anthropic
    fresh_provider._ensure_client()
    fake_anthropic_module.Anthropic.assert_called_once()
    call_kwargs = fake_anthropic_module.Anthropic.call_args.kwargs
    assert call_kwargs.get("timeout") == _F99_READ_TIMEOUT
    # api_key still forwarded (the existing contract).
    assert call_kwargs.get("api_key") == "test-key"


def test_ensure_client_preserves_explicit_timeout(fresh_provider, patched_anthropic) -> None:
    """F-99: caller-supplied ``timeout`` overrides the F-99 default.

    If a future caller threads an ``http_client`` or custom
    ``timeout`` through ``_client_kwargs``, F-99 must not stomp on
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
    """F-99: caller-supplied ``http_client`` wins over the F-99 timeout.

    A caller that builds their own httpx client (e.g. with proxy,
    SSL context, or telemetry hooks) wants F-99 to stay out of the
    way. The ``if 'http_client' not in kwargs`` guard ensures the
    F-99 timeout is only applied when the SDK is responsible for
    building its own httpx client.
    """
    mod, fake_anthropic_module = patched_anthropic
    custom_http = MagicMock(name="custom httpx client")
    fresh_provider._client_kwargs["http_client"] = custom_http
    fresh_provider._ensure_client()
    call_kwargs = fake_anthropic_module.Anthropic.call_args.kwargs
    # When http_client is supplied, F-99 must NOT also supply
    # timeout — the user's client owns its own timeout config.
    assert "timeout" not in call_kwargs
    assert call_kwargs.get("http_client") is custom_http


def test_ensure_client_caches_client(fresh_provider, patched_anthropic) -> None:
    """F-99: subsequent calls return the cached client.

    The existing cache contract (set ``self.client`` once, return
    the same instance) must be preserved by the F-99 fix. This pins
    that we don't accidentally rebuild the client per request.
    """
    mod, fake_anthropic_module = patched_anthropic
    c1 = fresh_provider._ensure_client()
    c2 = fresh_provider._ensure_client()
    assert c1 is c2
    # Anthropic() constructor called exactly once.
    assert fake_anthropic_module.Anthropic.call_count == 1


def test_ensure_client_forwards_base_url(fresh_provider, patched_anthropic) -> None:
    """F-99: ``base_url`` (and any other ``_client_kwargs`` keys) still forwarded.

    Regression guard — the F-99 fix only adds a default ``timeout``
    kwarg; existing keys must still reach the constructor so the
    proxy / custom-endpoint flow keeps working.
    """
    mod, fake_anthropic_module = patched_anthropic
    fresh_provider._ensure_client()
    call_kwargs = fake_anthropic_module.Anthropic.call_args.kwargs
    assert call_kwargs.get("base_url") == "https://example.invalid"
