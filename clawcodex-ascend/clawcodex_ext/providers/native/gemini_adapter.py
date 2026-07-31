"""Native Gemini provider wrapper for F-72 (P72-B).

The :class:`~src.providers.gemini_provider.GeminiProvider` in
``src/providers/`` is already a fully native implementation built on
``google-genai`` — it converts the Anthropic-style message format into
the Gemini ``contents``/``parts`` shape and back. The F-72 plan asks
for a ``native/gemini_adapter.py`` entrypoint that:

* sits in the ``native/`` subpackage so the F-72 factory and the
  capability registry see it as a first-class native adapter;
* re-exports the existing provider's behaviour under a class that
  carries the :class:`NativeProvider` capabilities contract;
* advertises Gemini-exclusive features
  (:data:`CAP_SAFETY_SETTINGS`, :data:`CAP_GROUNDING`,
  :data:`CAP_TTS`, :data:`CAP_AUDIO_INPUT`) so the rest of the
  system can reason about Gemini's platform-specific extensions.

**Why composition?** The previous draft inherited from both
``NativeProvider`` and ``GeminiProvider`` so the ``chat`` /
``chat_stream`` methods would be inherited. That made the module
unimportable on machines that don't have ``google-genai`` installed —
the ``from ..gemini_provider import GeminiProvider`` line raised
``ImportError`` at module-load time, and the F-72 factory's "soft
fallback" contract (return ``None`` instead of raising) couldn't
even reach the import stage. Composition sidesteps the issue: the
wrapper class can be *defined* regardless of SDK availability; the
inner :class:`GeminiProvider` is constructed lazily on first use,
and instantiation-time errors surface as a clean ``None`` from
:func:`create_native_provider`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .base import NativeProvider
from .capabilities import (
    CAP_AUDIO_INPUT,
    CAP_GROUNDING,
    CAP_SAFETY_SETTINGS,
    CAP_TTS,
    CAP_VISION,
)

if TYPE_CHECKING:
    # Imported only for typing. The runtime import is deferred to
    # ``__init__`` so the module can load when ``google-genai`` is
    # missing.
    from src.providers.gemini_provider import GeminiProvider


class NativeGeminiProvider(NativeProvider):
    """Native Gemini adapter (F-72) — composition over ``GeminiProvider``.

    The wrapper holds a :class:`GeminiProvider` instance and delegates
    the chat / streaming / model-listing surface to it. The wrapping
    layer exists for two reasons:

    1. The F-72 capability registry needs a class that descends from
       :class:`NativeProvider` so callers can ask
       ``NativeGeminiProvider.has_capability("safety_settings")``.
       The underlying ``GeminiProvider`` does not expose that
       interface.
    2. The :meth:`with_safety_settings` / :meth:`with_grounding`
       factory methods need a stable entrypoint that survives
       re-organisation of the inner provider's keyword surface.
    """

    capabilities = {
        CAP_VISION,
        CAP_SAFETY_SETTINGS,
        CAP_GROUNDING,
        CAP_TTS,
        CAP_AUDIO_INPUT,
    }

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(api_key, base_url, model)
        # ``_inner`` is built lazily because the underlying SDK
        # import is the failure mode the factory is trying to
        # catch. We expose the inner provider's behaviour through
        # ``chat`` / ``chat_stream`` / ``get_available_models``
        # below.
        self._inner: Optional["GeminiProvider"] = None
        self._inner_error: Optional[Exception] = None
        # Native-only configuration that gets spliced into the
        # request body when ``chat`` is called. These default to
        # ``None`` so the wrapper behaves like the underlying
        # provider when the caller didn't opt in.
        self._native_safety_settings: Optional[list[dict[str, Any]]] = None
        self._native_grounding_enabled: bool = False

    # ---- inner construction ----

    def _ensure_inner(self) -> "GeminiProvider":
        if self._inner is not None:
            return self._inner
        if self._inner_error is not None:
            raise self._inner_error
        try:
            from src.providers.gemini_provider import GeminiProvider  # local import
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self._inner_error = exc
            raise
        self._inner = GeminiProvider(
            api_key=self.api_key or "",
            base_url=self.base_url,
            model=self.model,
        )
        return self._inner

    def get_provider_name(self) -> str:
        return "gemini"

    # ---- native configuration passthroughs ----

    @classmethod
    def with_safety_settings(
        cls,
        api_key: str,
        safety_settings: list[dict[str, Any]],
        **kwargs: Any,
    ) -> "NativeGeminiProvider":
        """Build an instance pre-configured with Gemini ``SafetySetting``s.

        ``safety_settings`` is a list of
        ``{"category": "...", "threshold": "..."}`` dicts in the
        shape the Gemini SDK expects.
        """
        if not isinstance(safety_settings, list):
            raise TypeError("safety_settings must be a list of dicts")
        instance = cls(api_key=api_key, **kwargs)
        instance._native_safety_settings = safety_settings
        return instance

    @classmethod
    def with_grounding(
        cls,
        api_key: str,
        **kwargs: Any,
    ) -> "NativeGeminiProvider":
        """Build an instance that requests Google Search grounding."""
        instance = cls(api_key=api_key, **kwargs)
        instance._native_grounding_enabled = True
        return instance

    # ---- delegation to the inner provider ----

    def chat(self, messages, tools=None, **kwargs):
        inner = self._ensure_inner()
        # Splice the native configuration into the kwargs the inner
        # provider already understands. ``inner`` does not yet read
        # these — wiring them through is a P72-E follow-up — but
        # surfacing them on the kwargs means the values are
        # discoverable from logs and tracing.
        merged: dict[str, Any] = dict(kwargs)
        if self._native_safety_settings is not None:
            merged.setdefault("safety_settings", self._native_safety_settings)
        if self._native_grounding_enabled:
            merged.setdefault("grounding", True)
        return inner.chat(messages, tools, **merged)

    def chat_stream(self, messages, tools=None, **kwargs):
        inner = self._ensure_inner()
        return inner.chat_stream(messages, tools, **kwargs)

    def chat_stream_response(self, messages, tools=None, on_text_chunk=None, **kwargs):  # pylint: disable=arguments-differ
        inner = self._ensure_inner()
        return inner.chat_stream_response(messages, tools, on_text_chunk=on_text_chunk, **kwargs)

    def get_available_models(self) -> list[str]:
        try:
            inner = self._ensure_inner()
        except (ImportError, ModuleNotFoundError):
            return ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]
        return inner.get_available_models()


__all__ = ["NativeGeminiProvider"]
