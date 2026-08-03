"""Tests for the native Gemini adapter wrapper (F-72 P72-B)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.providers.native.capabilities import (
    CAP_AUDIO_INPUT,
    CAP_GROUNDING,
    CAP_SAFETY_SETTINGS,
    CAP_TTS,
    CAP_VISION,
)
from src.providers.native.gemini_adapter import NativeGeminiProvider


def test_capabilities_match_f72_plan() -> None:
    """Gemini's exclusive capabilities: safety settings, grounding,
    TTS, audio input, vision. Notably *not* structured_output
    (Gemini's response_schema is distinct from OpenAI's response_format
    and warrants its own capability tag in a future iteration).
    """
    expected = {
        CAP_VISION,
        CAP_SAFETY_SETTINGS,
        CAP_GROUNDING,
        CAP_TTS,
        CAP_AUDIO_INPUT,
    }
    assert NativeGeminiProvider.capabilities == expected


def test_get_provider_name() -> None:
    assert NativeGeminiProvider(api_key="k").get_provider_name() == "gemini"


def test_construction_does_not_eagerly_import_inner() -> None:
    """The wrapper must be constructible even when ``google-genai``
    is broken — the inner provider is built lazily inside
    ``_ensure_inner``. This is the central design point of the
    composition refactor (see the ``gemini_adapter`` docstring).
    """
    # ``NativeGeminiProvider.__init__`` should not touch the SDK.
    wrapper = NativeGeminiProvider(api_key="k", model="gemini-2.5-pro")
    assert wrapper._inner is None
    assert wrapper._native_safety_settings is None
    assert wrapper._native_grounding_enabled is False


def test_get_available_models_falls_back_when_sdk_broken(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_available_models`` must return the static fallback
    list when the SDK can't be imported — callers like the
    ``/provider`` picker must not crash on a dev environment
    that doesn't have ``google-genai`` installed.
    """
    wrapper = NativeGeminiProvider(api_key="k")
    # Bypass the lazy-import path: pretend the inner provider
    # construction would fail. We do this by pre-setting
    # ``_inner_error`` so ``_ensure_inner`` short-circuits to
    # the raise branch.
    wrapper._inner = None
    wrapper._inner_error = ImportError("simulated SDK failure")

    models = wrapper.get_available_models()
    assert "gemini-2.5-pro" in models
    assert "gemini-2.5-flash" in models


def test_with_safety_settings_validates_type() -> None:
    with pytest.raises(TypeError):
        NativeGeminiProvider.with_safety_settings(
            api_key="k",
            safety_settings="not-a-list",  # type: ignore[arg-type]
        )


def test_with_safety_settings_attaches_payload() -> None:
    payload = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]
    wrapper = NativeGeminiProvider.with_safety_settings(api_key="k", safety_settings=payload)
    assert wrapper._native_safety_settings == payload
    assert wrapper._native_grounding_enabled is False


def test_with_grounding_enables_flag() -> None:
    wrapper = NativeGeminiProvider.with_grounding(api_key="k")
    assert wrapper._native_grounding_enabled is True
    assert wrapper._native_safety_settings is None


def test_chat_delegates_to_inner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper's ``chat`` must call into the underlying
    ``GeminiProvider`` and forward arguments.
    """
    sentinel_response = MagicMock(content="hello", model="gemini-2.5-pro")
    inner = MagicMock()
    inner.chat.return_value = sentinel_response

    wrapper = NativeGeminiProvider(api_key="k", model="gemini-2.5-pro")
    wrapper._inner = inner  # bypass the lazy-import path

    result = wrapper.chat([{"role": "user", "content": "hi"}])
    assert result is sentinel_response
    inner.chat.assert_called_once()


def test_chat_splices_safety_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """``safety_settings`` is a F-72 native-only kwarg that the
    wrapper attaches to the call so it's observable in logs and
    trace events.
    """
    inner = MagicMock()
    inner.chat.return_value = MagicMock()

    wrapper = NativeGeminiProvider.with_safety_settings(
        api_key="k",
        safety_settings=[{"category": "X", "threshold": "BLOCK_NONE"}],
    )
    wrapper._inner = inner
    wrapper.chat([{"role": "user", "content": "hi"}])
    kwargs = inner.chat.call_args.kwargs
    assert kwargs.get("safety_settings") == [{"category": "X", "threshold": "BLOCK_NONE"}]
    # Grounding is opt-in via a separate factory method; this
    # wrapper did not enable it, so the kwarg must be absent.
    assert "grounding" not in kwargs


def test_chat_splices_grounding_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The grounding opt-in is a boolean flag the wrapper
    forwards as ``grounding=True`` to the inner provider.
    """
    inner = MagicMock()
    inner.chat.return_value = MagicMock()

    wrapper = NativeGeminiProvider.with_grounding(api_key="k")
    wrapper._inner = inner
    wrapper.chat([{"role": "user", "content": "hi"}])
    kwargs = inner.chat.call_args.kwargs
    assert kwargs.get("grounding") is True
    # Safety settings were not requested on this wrapper.
    assert "safety_settings" not in kwargs
