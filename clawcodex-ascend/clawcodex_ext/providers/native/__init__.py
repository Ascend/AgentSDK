"""Native SDK adapter factory for F-72 (P72-D).

This package re-implements a slim set of provider classes that talk
directly to vendor SDKs (``openai``, ``google-genai``) rather than
going through LiteLLM. The motivation is to expose platform-specific
features (Gemini's ``SafetySetting``, OpenAI's ``response_format``
JSON-Schema mode, Grok's tool-calling variant) that the universal
LiteLLM facade tends to flatten.

The factory follows the resolution order spelled out in the F-72
plan:

1. If the caller passed an explicit provider class, build it.
2. Otherwise, look up the provider name in the native registry
   (``openai`` → :class:`NativeOpenAIProvider`, ``gemini`` →
   :class:`NativeGeminiProvider`, ``grok`` → :class:`NativeGrokProvider`).
3. If the SDK for the requested provider is not installed, return
   ``None`` so the caller can fall back to LiteLLM. This is the
   "soft fallback" — we never raise for a missing optional dep, we
   simply don't claim to support the provider.

The :func:`create_native_provider` helper is the public entry point;
:func:`get_native_provider_class` is a thin wrapper around the
internal registry that the F-72 wiring code in
``clawcodex_ext.providers.factory`` uses when a caller asks for a
native adapter by name.
"""

from __future__ import annotations

from typing import Any, Optional

from clawcodex_ext.providers.base import BaseProvider
from .base import NativeProvider
from .grok_adapter import NativeGrokProvider
from .openai_adapter import NativeOpenAIProvider

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps a provider name (matching the keys in
#: ``src.providers.PROVIDER_INFO``) to its native adapter class.
#: ``gemini`` is inserted lazily on first access so that the
#: ``google-genai`` import does not block startup when the user only
#: uses OpenAI / Grok.
_NATIVE_REGISTRY: dict[str, type[NativeProvider]] = {
    "openai": NativeOpenAIProvider,
    "grok": NativeGrokProvider,
}


def _register_gemini() -> Optional[type[NativeProvider]]:
    """Late-bind the Gemini adapter to avoid an unconditional
    ``google-genai`` import.

    Most dev environments only have ``openai`` installed; if we
    imported ``google-genai`` at module top, the entire
    ``src.providers`` package would fail to import on those
    machines. The ``NativeGeminiProvider`` class itself swallows
    the ``ModuleNotFoundError`` at instantiation time, so this
    registry lookup has to do the same.

    We catch both ``ModuleNotFoundError`` (the SDK isn't installed
    at all) and ``ImportError`` (the ``google`` namespace package
    exists but the ``genai`` submodule is missing or broken). On
    some platforms the ``google-genai`` distribution fails to
    register its submodule in the parent namespace, and the only
    visible signal at this layer is a bare ``ImportError``.
    """
    try:
        from .gemini_adapter import NativeGeminiProvider
    except (ImportError, ModuleNotFoundError):
        return None
    _NATIVE_REGISTRY["gemini"] = NativeGeminiProvider
    return NativeGeminiProvider


def get_native_provider_class(name: str) -> Optional[type[NativeProvider]]:
    """Look up a native provider class by registry name.

    Returns ``None`` when the name is unknown or when the
    corresponding SDK is not installed.
    """
    if name in _NATIVE_REGISTRY:
        return _NATIVE_REGISTRY[name]
    if name == "gemini":
        cls = _register_gemini()
        return cls
    return None


def registered_native_providers() -> dict[str, type[NativeProvider]]:
    """Return a *copy* of the current native registry.

    Copying the dict prevents callers from mutating the registry
    out from under the factory. ``gemini`` is bound lazily on the
    first call to this function, so the snapshot reflects whatever
    is currently importable.
    """
    # Touch the gemini slot so a single call to this function
    # yields a complete view — useful for the CLI's ``/provider``
    # listing.
    get_native_provider_class("gemini")
    return dict(_NATIVE_REGISTRY)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_native_provider(
    provider_name: str,
    config: Optional[dict[str, Any]] = None,
) -> Optional[BaseProvider]:
    """Try to build a native adapter for *provider_name*.

    The factory is deliberately permissive: any failure (unknown
    name, missing SDK, bad config) returns ``None`` so the caller
    can fall back to LiteLLM without a try/except. The F-72 plan
    calls this "soft fallback" — the native path is an
    *optimisation*, not a requirement.

    Args:
        provider_name: The provider key, e.g. ``"openai"``.
        config: A dict with ``api_key``, ``base_url``, and
            ``default_model`` keys. Missing keys are tolerated and
            fall back to the adapter's class-level defaults.

    Returns:
        A configured :class:`NativeProvider` instance, or ``None``
        when the native path is unavailable for the given name.
    """
    cls = get_native_provider_class(provider_name)
    if cls is None:
        return None
    cfg = config or {}
    try:
        return cls(
            api_key=cfg.get("api_key", ""),
            base_url=cfg.get("base_url"),
            model=cfg.get("default_model"),
        )
    except (ModuleNotFoundError, ImportError):
        # SDK is missing — fall back to LiteLLM rather than raising.
        return None
    except Exception:
        # Construction failed for any other reason (e.g. bad
        # credentials at the network layer, though that wouldn't
        # normally surface here). Returning ``None`` is the
        # documented contract; the caller logs the failure.
        return None


__all__ = [
    "NativeProvider",
    "NativeOpenAIProvider",
    "NativeGrokProvider",
    "create_native_provider",
    "get_native_provider_class",
    "registered_native_providers",
]


def __getattr__(name: str):  # pragma: no cover - lazy import hook
    """Late-bind :class:`NativeGeminiProvider` for
    ``from clawcodex_ext.providers.native import NativeGeminiProvider``.
    """
    if name == "NativeGeminiProvider":
        cls = _register_gemini()
        if cls is None:
            raise AttributeError(
                "NativeGeminiProvider is not available — install `google-genai` to use the native Gemini adapter."
            )
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
