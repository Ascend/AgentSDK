from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from clawcodex_ext.providers.codex_models import get_codex_model_ids
from clawcodex_ext.providers.hooks import _codex_api_discovery


def test_codex_registry_discovery_uses_runtime_credentials(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.auth.codex_oauth.resolve_codex_runtime_credentials",
        lambda **kwargs: SimpleNamespace(api_key="live-access-token"),
    )
    seen_tokens: list[str] = []
    monkeypatch.setattr(
        "clawcodex_ext.providers.codex_models.get_codex_model_ids",
        lambda token: seen_tokens.append(token) or ["gpt-live-account-model"],
    )

    assert _codex_api_discovery() == ["gpt-live-account-model"]
    assert seen_tokens == ["live-access-token"]


def test_codex_fallback_catalog_omits_account_unsupported_model() -> None:
    models = get_codex_model_ids("")

    assert "gpt-5.6-sol" in models
    assert "gpt-5.3-codex" not in models


def test_codex_model_discovery_retries_with_one_persistent_client(monkeypatch) -> None:
    calls: list[int] = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"models": [{"slug": "gpt-live-after-retry"}]}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, *args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ConnectError("transient TLS EOF")
            return Response()

    clients: list[Client] = []

    def make_client(**kwargs):
        client = Client()
        clients.append(client)
        return client

    monkeypatch.setattr("clawcodex_ext.providers.codex_models.httpx.Client", make_client)
    monkeypatch.setattr(
        "clawcodex_ext.providers.codex_models.httpx.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fresh client used")),
    )

    assert get_codex_model_ids("access-token") == ["gpt-live-after-retry"]
    assert len(clients) == 1
    assert len(calls) == 2


def test_codex_model_discovery_can_surface_exhausted_transport_errors(monkeypatch) -> None:
    calls = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, *args, **kwargs):
            calls.append(1)
            raise httpx.ConnectError("TLS EOF")

    timeouts = []
    monkeypatch.setattr(
        "clawcodex_ext.providers.codex_models.httpx.Client",
        lambda **kwargs: timeouts.append(kwargs["timeout"]) or Client(),
    )

    with pytest.raises(RuntimeError, match="Codex model discovery failed"):
        get_codex_model_ids("access-token", raise_on_error=True)

    assert timeouts == [3.0]
    assert len(calls) == 2
