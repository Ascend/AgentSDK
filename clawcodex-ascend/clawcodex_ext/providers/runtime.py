"""Runtime provider construction helpers."""

from __future__ import annotations

import os

# F-99 fix: importing the package (not just ``factory``) triggers
# ``clawcodex_ext/providers/__init__.py``, which registers the
# cancel-latency-fixed ``ClawcodexAnthropicProvider`` /
# ``ClawcodexMinimaxProvider`` overrides via
# ``_EXTRA_PROVIDER_CLASSES``. Without this side-effect import,
# ``get_provider_class("anthropic")`` falls through to the bare
# upstream ``AnthropicProvider``, which iterates
# ``stream.text_stream`` on the main thread — its only cancellation
# mechanism is ``response.close()`` (advisory on Linux/Winsock and
# silently a no-op for the ``_transport`` access in our
# ``_close_transport_safely`` helper because ``httpx.Response``
# doesn't expose ``_transport``), so a Ctrl+C waits the full platform
# socket timeout (~60s) instead of the ~100ms the worker-thread
# poll in ``drain_text_stream_with_abort_poll`` provides.
#
# Importing here is safe because every provider-build call site goes
# through ``build_provider_from_config`` / ``create_provider``, and
# this module sits at the top of that dependency chain. ``factory``
# already imports cleanly from this module (no circular reference),
# so the chain is: caller -> ``runtime`` -> factory + package
# ``__init__`` -> ``register_provider`` populates
# ``_EXTRA_PROVIDER_CLASSES``.
import clawcodex_ext.providers  # noqa: F401  -- side-effect import

from src.auth.codex_oauth import CodexAuthError, resolve_codex_runtime_credentials
from src.config import get_provider_config
from clawcodex_ext.providers.factory import create_provider

from clawcodex_ext.providers.base import BaseProvider

OAUTH_PROVIDERS = {"openai-codex"}


def build_provider_from_config(provider_name: str, model: str | None = None) -> BaseProvider:
    try:
        provider_cfg = get_provider_config(provider_name)
    except ValueError:
        provider_cfg = {}
    selected_model = model or provider_cfg.get("default_model")

    if provider_name == "openai-codex":
        try:
            credentials = resolve_codex_runtime_credentials()
        except CodexAuthError as exc:
            raise RuntimeError(
                f"OpenAI Codex is not authenticated. Run `clawcodex login` and select openai-codex. ({exc})"
            ) from exc
        return create_provider(
            provider_name,
            api_key=credentials.api_key,
            base_url=provider_cfg.get("base_url") or credentials.base_url,
            model=selected_model,
        )

    # Resolve API key: config first, then env var / keychain fallback.
    if not provider_cfg.get("api_key"):
        from src.auth.auth import load_api_key

        api_key = load_api_key(provider_name)
        if not api_key and provider_cfg:
            raise RuntimeError(
                f"API key for provider '{provider_name}' is not configured. "
                "Run `clawcodex login` to set it up, or set the "
                f"{provider_name.upper()}_API_KEY environment variable."
            )
    else:
        api_key = provider_cfg["api_key"]

    # Resolve base_url: config first, then env var fallback.
    base_url = provider_cfg.get("base_url")
    if not base_url:
        env_base = os.environ.get(f"{provider_name.upper()}_BASE_URL")
        if env_base:
            base_url = env_base

    return create_provider(
        provider_name,
        api_key=api_key,
        base_url=base_url,
        model=selected_model,
    )
