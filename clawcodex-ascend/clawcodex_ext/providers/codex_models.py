"""Model discovery for OpenAI Codex ChatGPT OAuth."""

from __future__ import annotations

from typing import Any

import httpx

CODEX_FALLBACK_MODELS = [
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex-spark",
    "codex-auto-review",
]
CODEX_MODELS_URL = "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0"
CODEX_MODEL_DISCOVERY_ATTEMPTS = 2


def get_codex_model_ids(
    access_token: str,
    *,
    timeout_seconds: float = 3.0,
    raise_on_error: bool = False,
) -> list[str]:
    """Discover model identifiers available to a Codex OAuth session.

    Args:
        access_token: ChatGPT OAuth access token.
        timeout_seconds: Per-request HTTP timeout.
        raise_on_error: Raise instead of returning the fallback catalog.

    Returns:
        Discovered model identifiers, or the configured fallback catalog.
    """

    if not access_token.strip():
        return list(CODEX_FALLBACK_MODELS)
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            for attempt in range(CODEX_MODEL_DISCOVERY_ATTEMPTS):
                try:
                    response = client.get(
                        CODEX_MODELS_URL,
                        headers={"Authorization": f"Bearer {access_token}"},
                    )
                except httpx.TransportError:
                    if attempt + 1 < CODEX_MODEL_DISCOVERY_ATTEMPTS:
                        continue
                    raise
                if response.status_code >= 500 and attempt + 1 < CODEX_MODEL_DISCOVERY_ATTEMPTS:
                    continue
                if response.status_code != 200:
                    return _fallback_or_raise(
                        f"Codex model discovery failed with status {response.status_code}.",
                        raise_on_error=raise_on_error,
                    )
                models = _extract_models(response.json())
                if models:
                    return models
                return _fallback_or_raise(
                    "Codex model discovery returned no models.",
                    raise_on_error=raise_on_error,
                )
    except Exception as exc:
        if raise_on_error:
            raise RuntimeError(f"Codex model discovery failed: {exc}") from exc
        return list(CODEX_FALLBACK_MODELS)
    return _fallback_or_raise(
        "Codex model discovery failed after all retry attempts.",
        raise_on_error=raise_on_error,
    )


def _fallback_or_raise(message: str, *, raise_on_error: bool) -> list[str]:
    if raise_on_error:
        raise RuntimeError(message)
    return list(CODEX_FALLBACK_MODELS)


def _extract_models(payload: Any) -> list[str]:
    candidates: Any = payload
    if isinstance(payload, dict):
        candidates = payload.get("models") or payload.get("data") or []
    if not isinstance(candidates, list):
        return []
    model_ids: list[str] = []
    for item in candidates:
        if isinstance(item, str):
            model_id = item
            hidden = False
        elif isinstance(item, dict):
            model_id = item.get("id") or item.get("name") or item.get("slug")
            hidden = bool(item.get("hidden") or item.get("hide"))
        else:
            continue
        if isinstance(model_id, str) and model_id and not hidden:
            model_ids.append(model_id)
    return list(dict.fromkeys(model_ids))
