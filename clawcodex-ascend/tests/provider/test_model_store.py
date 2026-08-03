from __future__ import annotations

import pytest

from clawcodex_ext.cli.model_cmd.errors import UnknownModelError
from clawcodex_ext.cli.model_cmd.store import ModelStore
from clawcodex_ext.cli.provider_cmd.errors import (
    UnsupportedScopeError as ProviderUnsupportedScopeError,
)
from clawcodex_ext.cli.model_cmd.errors import UnsupportedScopeError as ModelUnsupportedScopeError


def test_model_store_sets_default_provider(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("src.config.set_default_provider", calls.append)

    ModelStore().set_default_provider("glm")

    assert calls == ["zai"]


def test_model_store_rejects_project_provider_scope() -> None:
    with pytest.raises(ProviderUnsupportedScopeError):
        ModelStore().set_default_provider("glm", scope="project")


def test_model_store_sets_default_model_without_losing_existing_config(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "src.config.get_provider_config",
        lambda provider: {"api_key": "secret", "base_url": "https://custom.example"},
    )

    def fake_set_api_key(provider: str, **kwargs) -> None:
        calls.append({"provider": provider, **kwargs})

    monkeypatch.setattr("src.config.set_api_key", fake_set_api_key)

    ModelStore().set_default_model("glm", "zai/glm-4")

    assert calls == [
        {
            "provider": "zai",
            "api_key": "secret",
            "base_url": "https://custom.example",
            "default_model": "zai/glm-4",
        }
    ]


def test_model_store_rejects_project_model_scope() -> None:
    with pytest.raises(ModelUnsupportedScopeError):
        ModelStore().set_default_model("glm", "zai/glm-4", scope="project")


# ---------------------------------------------------------------------------
# set_default_model: allow_unknown + persist_unknown convenience method
# ---------------------------------------------------------------------------


def test_set_default_model_unknown_still_raises_by_default() -> None:
    """Without ``allow_unknown``, an unknown model still raises."""
    with pytest.raises(UnknownModelError):
        ModelStore().set_default_model("glm", "totally-bogus-model-xyz")


def test_set_default_model_allow_unknown_persists(monkeypatch) -> None:
    """``allow_unknown=True`` skips registry validation and persists."""
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "src.config.get_provider_config",
        lambda provider: {"api_key": "secret", "base_url": "https://custom.example"},
    )

    def fake_set_api_key(provider: str, **kwargs) -> None:
        calls.append({"provider": provider, **kwargs})

    monkeypatch.setattr("src.config.set_api_key", fake_set_api_key)

    ModelStore().set_default_model("glm", "totally-bogus-model-xyz", allow_unknown=True)

    assert calls == [
        {
            "provider": "zai",
            "api_key": "secret",
            "base_url": "https://custom.example",
            "default_model": "totally-bogus-model-xyz",
        }
    ]


def test_set_default_model_persist_unknown_accepts_unknown_model(monkeypatch) -> None:
    """``set_default_model_persist_unknown`` is the convenience wrapper used by /model REPL."""
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "src.config.get_provider_config",
        lambda provider: {"api_key": "secret", "base_url": "https://custom.example"},
    )

    def fake_set_api_key(provider: str, **kwargs) -> None:
        calls.append({"provider": provider, **kwargs})

    monkeypatch.setattr("src.config.set_api_key", fake_set_api_key)

    # Should not raise — REPL needs to tolerate unknown model names.
    ModelStore().set_default_model_persist_unknown("glm", "totally-bogus-model-xyz")

    assert calls[0]["default_model"] == "totally-bogus-model-xyz"


def test_set_default_model_persist_unknown_falls_back_when_provider_config_missing(
    monkeypatch,
) -> None:
    """When get_provider_config raises ValueError (no provider entry), fall back to registry default base URL."""
    from src.providers import PROVIDER_INFO

    calls: list[dict[str, object]] = []

    def fake_get_provider_config(provider: str) -> dict[str, object]:
        raise ValueError(f"Unknown provider: {provider}")

    def fake_set_api_key(provider: str, **kwargs) -> None:
        calls.append({"provider": provider, **kwargs})

    monkeypatch.setattr("src.config.get_provider_config", fake_get_provider_config)
    monkeypatch.setattr("src.config.set_api_key", fake_set_api_key)

    ModelStore().set_default_model_persist_unknown("glm", "totally-bogus-model-xyz")

    assert calls[0]["base_url"] == PROVIDER_INFO["zai"]["default_base_url"]
    assert calls[0]["api_key"] == ""
    assert calls[0]["default_model"] == "totally-bogus-model-xyz"


def test_set_default_model_persist_unknown_rejects_project_scope() -> None:
    """``scope != "user"`` is still rejected for the convenience method."""
    with pytest.raises(ModelUnsupportedScopeError):
        ModelStore().set_default_model_persist_unknown("glm", "x", scope="project")
