from __future__ import annotations

from types import SimpleNamespace

from clawcodex_ext.providers.base import BaseProvider, ChatResponse


class CatalogProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test", model="configured-model")
        self.client = SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: SimpleNamespace(
                    data=[
                        SimpleNamespace(id="account-model-a"),
                        {"id": "account-model-b"},
                    ]
                )
            )
        )

    def chat(self, messages, tools=None, **kwargs) -> ChatResponse:
        raise NotImplementedError

    def chat_stream(self, messages, tools=None, **kwargs):
        raise NotImplementedError

    def get_available_models(self) -> list[str]:
        return ["configured-model"]


def test_base_provider_discovers_account_models_from_sdk_catalog() -> None:
    assert CatalogProvider().discover_available_models() == [
        "account-model-a",
        "account-model-b",
    ]
