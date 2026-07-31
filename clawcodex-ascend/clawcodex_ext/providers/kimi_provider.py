"""Kimi (Moonshot AI) provider implementation.

Kimi exposes two API flavors:

1. OpenAI-compatible chat completions at https://api.moonshot.ai/v1
   (and the China endpoint https://api.moonshot.cn/v1).  This is the
   default path for ``KimiProvider``.
2. Anthropic Messages API at https://api.kimi.com/coding for
   ``sk-kimi-*`` keys.  This is handled by ``KimiCodingProvider`` in
   ``kimi_coding_provider.py``.

See https://platform.kimi.ai/docs/api/overview for details.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from openai import OpenAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None

from clawcodex_ext.providers._moonshot_schema import sanitize_moonshot_tools
from clawcodex_ext.providers.openai_compatible import OpenAICompatibleProvider


class KimiProvider(OpenAICompatibleProvider):
    """Kimi provider using the OpenAI SDK against the Moonshot base URL."""

    DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize Kimi provider.

        Args:
            api_key: Moonshot API key.
            base_url: Base URL (optional, defaults to https://api.moonshot.ai/v1).
            model: Default model (default: kimi-k2.6).
        """
        super().__init__(
            api_key,
            base_url or self.DEFAULT_BASE_URL,
            model or "kimi-k2.6",
        )

    def _create_client(self) -> Any:
        """Create OpenAI SDK client pointed at Moonshot."""
        if OpenAI is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "openai package is not installed. Install optional dependencies to use KimiProvider."
            )
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url or self.DEFAULT_BASE_URL,
            "timeout": 60.0,
        }
        # Support SSL verification bypass for corporate/internal endpoints.
        import os

        if os.environ.get("CLAWCODEX_SSL_VERIFY", "").lower() in ("0", "false", "no"):
            import httpx

            kwargs["http_client"] = httpx.Client(verify=False, timeout=60.0)
        return OpenAI(**kwargs)

    def _prepare_tools(self, tools: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
        """Convert tools to OpenAI format and sanitize for Moonshot quirks.

        Moonshot rejects standard JSON-Schema shapes that OpenAI accepts,
        such as missing ``type`` on properties or ``type`` on the parent of
        an ``anyOf``.  The sanitizer normalizes these without losing
        semantics.
        """
        prepared = super()._prepare_tools(tools)  # pylint: disable=no-member
        return sanitize_moonshot_tools(prepared) if prepared else None

    @staticmethod
    def _apply_kimi_request_quirks(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Apply Moonshot-specific request quirks.

        - Drop ``temperature`` — Moonshot manages sampling internally and
          some model families reject the parameter.
        - Enable thinking via ``extra_body.thinking`` by default, unless an
          explicit ``reasoning_config`` requests a recognized effort level
          (then use ``reasoning_effort``) or disables thinking.

        Mirrors the behavior of hermes-agent's ``KimiProfile``.
        """
        kwargs = {k: v for k, v in kwargs.items() if k not in ("temperature",)}

        reasoning_config = kwargs.pop("reasoning_config", None)
        extra_body = kwargs.pop("extra_body", {}) or {}

        _thinking_off = bool(
            reasoning_config and isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False
        )

        if _thinking_off:
            extra_body["thinking"] = {"type": "disabled"}
        elif reasoning_config and isinstance(reasoning_config, dict):
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort in {"low", "medium", "high"}:
                kwargs["reasoning_effort"] = effort
            else:
                extra_body["thinking"] = {"type": "enabled"}
        else:
            # Default: let the server manage thinking depth.
            extra_body["thinking"] = {"type": "enabled"}

        if extra_body:
            kwargs["extra_body"] = extra_body

        return kwargs

    def chat(
        self,
        messages: list[Any],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        kwargs = self._apply_kimi_request_quirks(kwargs)
        return super().chat(messages, tools=tools, **kwargs)

    def chat_stream(
        self,
        messages: list[Any],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        kwargs = self._apply_kimi_request_quirks(kwargs)
        return super().chat_stream(messages, tools=tools, **kwargs)

    def chat_stream_response(  # pylint: disable=arguments-renamed
        self,
        messages: list[Any],
        tools: Optional[list[dict[str, Any]]] = None,
        on_text_chunk: Any = None,
        on_thinking_chunk: Any = None,
        abort_signal: Any = None,
        **kwargs: Any,
    ) -> Any:
        kwargs = self._apply_kimi_request_quirks(kwargs)
        return super().chat_stream_response(
            messages,
            tools=tools,
            on_text_chunk=on_text_chunk,
            on_thinking_chunk=on_thinking_chunk,
            abort_signal=abort_signal,
            **kwargs,
        )

    def get_available_models(self) -> list[str]:
        """Return Kimi's current production models.

        Model IDs are sourced from https://platform.kimi.ai/docs/models.md.
        Users can always pass arbitrary model IDs via config or CLI.
        """
        return [
            # K2.x flagship / multi-modal models
            "kimi-k2.6",
            "kimi-k2.5",
            "kimi-k2.7-code",
            "kimi-k2.7-code-highspeed",
            # Moonshot V1 generation models
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
            # Vision-capable V1 previews
            "moonshot-v1-8k-vision-preview",
            "moonshot-v1-32k-vision-preview",
            "moonshot-v1-128k-vision-preview",
        ]
