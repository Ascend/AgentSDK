"""Agnes AI image generation provider.

Uses the **OpenAI-compatible** ``POST /v1/images/generations`` endpoint,
which means it can reuse the same wire format as DALL-E.

API reference: https://agnes-ai.com/doc/agnes-image-21-flash

Supported models:
    - ``agnes-image-2.1-flash`` — text-to-image (high info density)
    - ``agnes-image-2.0-flash`` — image-to-image / multi-image
"""

from __future__ import annotations

from typing import Any

import httpx

from clawcodex_ext.providers.media.base import ImageProvider, ImageResult

# Default API base — matches the OpenAPI-compatible endpoint.
_DEFAULT_BASE_URL = "https://apihub.agnes-ai.com/v1"

# Supported image sizes for agnes-image-2.1-flash.
_SUPPORTED_SIZES = frozenset({"1024x1024", "1792x1024", "1024x1792"})

# Default fallback.
_DEFAULT_SIZE = "1024x1024"


class AgnesImageProvider(ImageProvider):
    """Image generation via Agnes AI (OpenAI-compatible interface).

    Args:
        api_key: Agnes AI API key.
        base_url: Base URL for the API (defaults to
            ``https://apihub.agnes-ai.com/v1``).
        default_model: Override the default model
            (``"agnes-image-2.1-flash"``).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        default_model: str = "agnes-image-2.1-flash",
        **kwargs: Any,
    ):
        super().__init__(api_key, base_url or _DEFAULT_BASE_URL, **kwargs)
        self.default_model = default_model
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=kwargs.get("timeout", 120.0),
        )

    def get_provider_name(self) -> str:
        return "agnes"

    def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        n: int = 1,
        **kwargs: Any,
    ) -> ImageResult:
        """Generate an image via ``POST /v1/images/generations``.

        Args:
            prompt: Text description of the desired image.
            model: Model name (``"agnes-image-2.1-flash"`` or
                ``"agnes-image-2.0-flash"``).
            size: Output size, e.g. ``"1024x1024"``.  Supported values
                depend on the model; ``agnes-image-2.1-flash`` supports
                ``1024x1024``, ``1792x1024``, ``1024x1792``.
            n: Number of images (default 1).
            **kwargs: Additional parameters passed through to the API.
                For image-to-image (``image`` key in ``extra_body``,
                ``tags=["img2img"]``).

        Returns:
            An :class:`ImageResult` with the generated image URL.

        Raises:
            httpx.HTTPStatusError: On API error (4xx/5xx).
        """
        resolved_model = model or self.default_model
        resolved_size = size or _DEFAULT_SIZE

        body: dict[str, Any] = {
            "model": resolved_model,
            "prompt": prompt,
            "size": resolved_size,
            "n": n,
        }

        # Image-to-image support: ``image`` is placed inside
        # ``extra_body`` per Agnes API convention.
        image_input = kwargs.get("image")
        if image_input:
            extra = dict(kwargs.get("extra_body", {}))
            extra.setdefault("image", image_input)
            if isinstance(image_input, list) and len(image_input) > 1:
                extra.setdefault("response_format", "url")
            body["extra_body"] = extra

        # Override response format if explicitly requested.
        response_format = kwargs.get("response_format")
        if response_format:
            if "extra_body" not in body:
                body["extra_body"] = {}
            body["extra_body"]["response_format"] = response_format

        resp = self._client.post("/images/generations", json=body)
        resp.raise_for_status()
        data = resp.json()

        # Parse the OpenAI-compatible response shape:
        #   { "data": [{ "url": "...", "revised_prompt": "..." }] }
        items = data.get("data", [])
        if not items:
            raise RuntimeError(f"Image generation returned empty data: {data}")

        first = items[0]
        return ImageResult(
            url=first.get("url", ""),
            revised_prompt=first.get("revised_prompt"),
            b64_json=first.get("b64_json"),
            extra={"model": resolved_model, "raw_response": data},
        )


__all__ = ["AgnesImageProvider"]
