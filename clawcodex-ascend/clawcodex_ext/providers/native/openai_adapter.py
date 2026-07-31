"""Native OpenAI provider using the ``openai`` SDK directly (F-72 P72-A).

Differences from the existing
:class:`~src.providers.openai_provider.OpenAIProvider`:

* Does **not** go through :class:`OpenAICompatibleProvider` — the
  Anthropic-to-OpenAI message translation is performed here using the
  same converter the compat path uses, but the SDK call is the bare
  OpenAI ``client.chat.completions.create`` call. This makes the
  capability surface (structured output, vision, streaming tools)
  discoverable through the ``capabilities`` registry.

* Advertises :data:`CAP_STRUCTURED_OUTPUT`,
  :data:`CAP_STREAMING_TOOLS`, :data:`CAP_VISION`, and
  :data:`CAP_REASONING` (the ``o1`` / ``o3`` families support
  ``reasoning_effort``).

* Falls back to a clear ``ModuleNotFoundError`` if the ``openai``
  package is missing, matching the existing pattern in
  :mod:`src.providers.gemini_provider`.
"""

from __future__ import annotations

from typing import Any, Generator, Optional

try:
    from openai import OpenAI  # type: ignore
    from openai import (
        APIError,  # type: ignore
        APITimeoutError,  # type: ignore
        AuthenticationError as OpenAIAuthError,  # type: ignore
        BadRequestError,  # type: ignore
        RateLimitError,  # type: ignore
    )
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None
    APIError = None
    APITimeoutError = None
    OpenAIAuthError = None
    BadRequestError = None
    RateLimitError = None

from src.providers.openai_compatible import _convert_anthropic_messages_to_openai

from .base import NativeProvider
from .capabilities import (
    CAP_REASONING,
    CAP_STREAMING_TOOLS,
    CAP_STRUCTURED_OUTPUT,
    CAP_VISION,
)


def _ensure_sdk() -> None:
    if OpenAI is None:
        raise ModuleNotFoundError(
            "openai package is not installed. Run `pip install openai` to use the native OpenAI adapter."
        )


class NativeOpenAIProvider(NativeProvider):
    """Native OpenAI provider via the ``openai`` Python SDK.

    The adapter uses the SDK directly rather than the OpenAI-compat
    facade so that platform-specific features (``response_format``,
    ``reasoning_effort``) flow through the SDK unchanged. The message
    converter is shared with the compat path — that translation is a
    pure data rewrite, not an API call, so sharing it does not
    compromise the "native" claim.
    """

    capabilities = {
        CAP_STRUCTURED_OUTPUT,
        CAP_STREAMING_TOOLS,
        CAP_VISION,
        CAP_REASONING,
    }

    DEFAULT_MODEL = "gpt-5.4"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(api_key, base_url, model or self.DEFAULT_MODEL)
        _ensure_sdk()
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 60.0}
        if base_url:
            kwargs["base_url"] = base_url
        # Honour the same SSL-bypass flag as the legacy OpenAIProvider
        # so that corporate-proxy users do not regress when they
        # switch to the native adapter.
        import os

        if os.environ.get("CLAWCODEX_SSL_VERIFY", "").lower() in ("0", "false", "no"):
            import httpx  # local import to keep the top-level dep minimal

            kwargs["http_client"] = httpx.Client(verify=False, timeout=60.0)
        self.client = OpenAI(**kwargs)

    def get_provider_name(self) -> str:
        return "openai"

    # ---- request shape ----

    def _check_supports_vision(self, **kwargs: Any) -> bool | None:
        """Check whether the resolved model supports vision input."""
        model = self._get_model(**kwargs)
        if model:
            from src.models.capabilities import supports_vision as _supports_vision

            return _supports_vision(model)
        return None

    def _build_request(
        self,
        messages: list[Any],
        tools: Optional[list[dict[str, Any]]],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Translate the Anthropic-style call into OpenAI kwargs.

        Note: we keep the converter local to this module (rather than
        reusing a class-level cache) so that the native adapter's
        behaviour is observable from a single read.
        """
        request: dict[str, Any] = {
            "model": self._get_model(**kwargs),
            "messages": _convert_anthropic_messages_to_openai(
                self._prepare_messages(messages),
                supports_vision=self._check_supports_vision(**kwargs),
            ),
        }
        if tools:
            request["tools"] = tools
        # Forward a small whitelist of OpenAI-native parameters so the
        # caller can opt into structured output / reasoning effort
        # without bypassing the adapter.
        for key in (
            "temperature",
            "top_p",
            "max_tokens",
            "response_format",
            "reasoning_effort",
            "stop",
            "user",
        ):
            if key in kwargs and kwargs[key] is not None:
                request[key] = kwargs[key]
        return request

    # ---- public API ----

    def chat(
        self,
        messages: list[Any],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        request = self._build_request(messages, tools, kwargs)
        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        message = choice.message
        tool_uses: Optional[list[dict[str, Any]]] = None
        if getattr(message, "tool_calls", None):
            tool_uses = []
            for tc in message.tool_calls:
                fn = getattr(tc, "function", None)
                tool_uses.append(
                    {
                        "id": str(getattr(tc, "id", "") or ""),
                        "name": str(getattr(fn, "name", "") or ""),
                        "input": _safe_json_loads(getattr(fn, "arguments", "") or "{}"),
                    }
                )
        usage = getattr(response, "usage", None)
        usage_dict: dict[str, Any] = {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
        }
        from src.providers.base import ChatResponse

        return ChatResponse(
            content=str(getattr(message, "content", "") or ""),
            model=str(getattr(response, "model", request["model"])),
            usage=usage_dict,
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            tool_uses=tool_uses,
        )

    def chat_stream(
        self,
        messages: list[Any],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        request = self._build_request(messages, tools, kwargs)
        request["stream"] = True
        stream = self.client.chat.completions.create(**request)
        for chunk in stream:
            try:
                choices = chunk.choices
            except AttributeError:
                continue
            if not choices:
                continue
            delta = choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece

    def chat_stream_response(  # pylint: disable=arguments-differ
        self,
        messages: list[Any],
        tools: Optional[list[dict[str, Any]]] = None,
        on_text_chunk=None,
        **kwargs: Any,
    ):
        # The native SDK supports incremental tool-call deltas; for
        # now we follow the existing convention of returning a single
        # concatenated ``ChatResponse`` from the non-streaming path.
        # Callers needing the tool_use mid-stream can switch to
        # ``chat_stream`` once we wire the tool_call deltas.
        response = self.chat(messages, tools, **kwargs)
        if on_text_chunk is not None and response.content:
            on_text_chunk(response.content)
        return response

    def get_available_models(self) -> list[str]:
        return [
            "gpt-5.4",
            "gpt-5.4-pro",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.2",
            "gpt-5.2-pro",
            "gpt-5.2-mini",
            "gpt-5.2-nano",
            "gpt-5.3-codex",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ]


def _safe_json_loads(raw: str) -> dict[str, Any]:
    """Parse a JSON object string, falling back to ``{"_raw": raw}``.

    The OpenAI SDK exposes ``tool_call.function.arguments`` as a raw
    string. Some providers (and older completions) emit malformed
    JSON; rather than raising inside the adapter, we surface the raw
    text under a ``_raw`` key so the upstream query loop can decide
    how to handle the parse failure.
    """
    import json

    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except (ValueError, TypeError):
        return {"_raw": raw}
    if isinstance(loaded, dict):
        return loaded
    return {"_raw": loaded}


__all__ = ["NativeOpenAIProvider"]
