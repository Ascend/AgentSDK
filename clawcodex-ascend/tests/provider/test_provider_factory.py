"""Tests for provider runtime factory selection."""

from __future__ import annotations

import pytest

from clawcodex_ext.providers.kimi_coding_provider import KimiCodingProvider
from clawcodex_ext.providers.kimi_provider import KimiProvider
from src import providers


class FakeProvider:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_should_use_litellm_truthy_values(monkeypatch, value):
    monkeypatch.setenv("CLAW_USE_LITELLM", value)

    assert providers.should_use_litellm() is True


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off"])
def test_should_use_litellm_false_values(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CLAW_USE_LITELLM", raising=False)
    else:
        monkeypatch.setenv("CLAW_USE_LITELLM", value)

    assert providers.should_use_litellm() is False


def test_create_provider_uses_original_provider_by_default(monkeypatch):
    monkeypatch.delenv("CLAW_USE_LITELLM", raising=False)
    monkeypatch.setattr(
        "clawcodex_ext.providers.factory.get_provider_class",
        lambda provider_name: FakeProvider,
    )

    provider = providers.create_provider(
        "openai",
        api_key="test-key",
        base_url="https://api.example.com/v1",
        model="gpt-4o",
    )

    assert isinstance(provider, FakeProvider)
    assert provider.args == ()
    assert provider.kwargs == {
        "api_key": "test-key",
        "base_url": "https://api.example.com/v1",
        "model": "gpt-4o",
    }


def test_create_provider_uses_litellm_when_enabled(monkeypatch):
    calls = []

    class FakeLiteLLMProvider:
        pass

    expected_provider = FakeLiteLLMProvider()

    def fake_create_litellm_provider(provider_name, *args, **kwargs):
        calls.append((provider_name, args, kwargs))
        return expected_provider

    monkeypatch.setenv("CLAW_USE_LITELLM", "1")
    monkeypatch.setattr(
        "clawcodex_ext.providers._litellm_adapter.create_litellm_provider",
        fake_create_litellm_provider,
    )

    provider = providers.create_provider(
        "anthropic",
        api_key="sk-ant-test",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-6",
    )

    assert provider is expected_provider
    assert calls == [
        (
            "anthropic",
            (),
            {
                "api_key": "sk-ant-test",
                "base_url": "https://api.anthropic.com",
                "model": "claude-sonnet-4-6",
            },
        )
    ]


def test_create_provider_kimi_native(monkeypatch):
    """Kimi resolves to the native KimiProvider when LiteLLM is disabled."""
    monkeypatch.delenv("CLAW_USE_LITELLM", raising=False)

    provider = providers.create_provider(
        "kimi",
        api_key="sk-kimi-test",
        model="kimi-k2.5",
    )

    assert isinstance(provider, KimiProvider)
    assert provider.api_key == "sk-kimi-test"
    assert provider.model == "kimi-k2.5"


def test_create_provider_kimi_litellm_fallback(monkeypatch):
    """Kimi routes through LiteLLM when CLAW_USE_LITELLM=1."""
    calls = []

    class FakeLiteLLMProvider:
        pass

    expected_provider = FakeLiteLLMProvider()

    def fake_create_litellm_provider(provider_name, *args, **kwargs):
        calls.append((provider_name, args, kwargs))
        return expected_provider

    monkeypatch.setenv("CLAW_USE_LITELLM", "1")
    monkeypatch.setattr(
        "clawcodex_ext.providers._litellm_adapter.create_litellm_provider",
        fake_create_litellm_provider,
    )

    provider = providers.create_provider(
        "kimi",
        api_key="sk-kimi-test",
        base_url="https://api.moonshot.ai/v1",
        model="kimi-k2.6",
    )

    assert provider is expected_provider
    assert calls == [
        (
            "kimi",
            (),
            {
                "api_key": "sk-kimi-test",
                "base_url": "https://api.moonshot.ai/v1",
                "model": "kimi-k2.6",
            },
        )
    ]


def test_create_provider_kimi_coding_native(monkeypatch):
    """Kimi Coding resolves to the native KimiCodingProvider when LiteLLM is disabled."""
    monkeypatch.delenv("CLAW_USE_LITELLM", raising=False)

    provider = providers.create_provider(
        "kimi-coding",
        api_key="sk-kimi-test",
        model="kimi-code",
    )

    assert isinstance(provider, KimiCodingProvider)
    assert provider.api_key == "sk-kimi-test"
    assert provider.model == "kimi-code"
    assert provider.base_url == "https://api.kimi.com/coding"


def test_create_provider_kimi_coding_litellm_fallback(monkeypatch):
    """Kimi Coding routes through LiteLLM when CLAW_USE_LITELLM=1."""
    calls = []

    class FakeLiteLLMProvider:
        pass

    expected_provider = FakeLiteLLMProvider()

    def fake_create_litellm_provider(provider_name, *args, **kwargs):
        calls.append((provider_name, args, kwargs))
        return expected_provider

    monkeypatch.setenv("CLAW_USE_LITELLM", "1")
    monkeypatch.setattr(
        "clawcodex_ext.providers._litellm_adapter.create_litellm_provider",
        fake_create_litellm_provider,
    )

    provider = providers.create_provider(
        "kimi-coding",
        api_key="sk-kimi-test",
        base_url="https://api.kimi.com/coding/",
        model="kimi-code",
    )

    assert provider is expected_provider
    assert calls == [
        (
            "kimi-coding",
            (),
            {
                "api_key": "sk-kimi-test",
                "base_url": "https://api.kimi.com/coding/",
                "model": "kimi-code",
            },
        )
    ]
