#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for src/services/langfuse/client.py (F-65 P65-A).

The Langfuse SDK is an *optional* dependency. These tests must
work whether or not it is importable. We rely on
:func:`reset_langfuse_client` and a fake module injection in
:func:`_install_fake_sdk` to simulate the SDK without requiring
the real package.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

from src.services.langfuse import client as client_module
from src.services.langfuse.client import (
    LangfuseConfig,
    get_langfuse_client,
    init_langfuse,
    is_langfuse_available,
    reset_langfuse_client,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_client_state():
    """Each test starts with a clean singleton + warning flags."""
    reset_langfuse_client()
    client_module._warned_missing_dep = False  # type: ignore[attr-defined]
    client_module._warned_missing_creds = False  # type: ignore[attr-defined]
    yield
    reset_langfuse_client()
    client_module._warned_missing_dep = False  # type: ignore[attr-defined]
    client_module._warned_missing_creds = False  # type: ignore[attr-defined]


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install a fake ``langfuse`` module on ``sys.modules`` so
    :func:`_try_import_sdk` resolves successfully.

    Returns a dict the test can inspect to see what was constructed.
    """
    state: dict[str, Any] = {"calls": []}

    class _FakeLangfuse:
        def __init__(self, *, public_key: str, secret_key: str, host: str) -> None:
            state["calls"].append({"public_key": public_key, "secret_key": secret_key, "host": host})

    fake_module = type(sys)("langfuse")
    fake_module.Langfuse = _FakeLangfuse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    return state


# ---------------------------------------------------------------------------
# LangfuseConfig
# ---------------------------------------------------------------------------


def test_config_is_configured_when_both_keys_set() -> None:
    cfg = LangfuseConfig(public_key="pk", secret_key="sk")
    assert cfg.is_configured is True


def test_config_not_configured_when_public_key_missing() -> None:
    cfg = LangfuseConfig(public_key="", secret_key="sk")
    assert cfg.is_configured is False


def test_config_not_configured_when_secret_key_missing() -> None:
    cfg = LangfuseConfig(public_key="pk", secret_key="")
    assert cfg.is_configured is False


def test_config_from_env_reads_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "env-pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "env-sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://example.langfuse.com")
    cfg = LangfuseConfig.from_env()
    assert cfg.public_key == "env-pk"
    assert cfg.secret_key == "env-sk"
    assert cfg.host == "https://example.langfuse.com"


def test_config_from_env_injected_mapping() -> None:
    cfg = LangfuseConfig.from_env(
        env={
            "LANGFUSE_PUBLIC_KEY": "inj-pk",
            "LANGFUSE_SECRET_KEY": "inj-sk",
            "LANGFUSE_HOST": "https://inj.example.com",
        }
    )
    assert cfg.public_key == "inj-pk"
    assert cfg.secret_key == "inj-sk"
    assert cfg.host == "https://inj.example.com"


def test_config_from_env_default_host() -> None:
    cfg = LangfuseConfig.from_env(env={})
    assert cfg.host == "https://cloud.langfuse.com"


# ---------------------------------------------------------------------------
# init_langfuse
# ---------------------------------------------------------------------------


def test_init_langfuse_returns_none_when_creds_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert init_langfuse() is None


def test_init_langfuse_returns_none_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    # Simulate "SDK not installed" by patching the import helper.
    monkeypatch.setattr(client_module, "_try_import_sdk", lambda: None)
    assert init_langfuse() is None
    assert is_langfuse_available() is False


def test_init_langfuse_constructs_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sdk(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    client = init_langfuse()
    assert client is not None
    assert is_langfuse_available() is True


def test_init_langfuse_explicit_args_win(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_sdk(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "env-pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "env-sk")
    init_langfuse(public_key="arg-pk", secret_key="arg-sk")
    assert state["calls"][-1]["public_key"] == "arg-pk"
    assert state["calls"][-1]["secret_key"] == "arg-sk"


def test_init_langfuse_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_sdk(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    first = init_langfuse()
    second = init_langfuse()
    assert first is second
    # Only one constructor call.
    assert len(state["calls"]) == 1


def test_init_langfuse_warns_on_config_change(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_fake_sdk(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    init_langfuse(host="https://first.example.com")
    with caplog.at_level("WARNING"):
        same_client = init_langfuse(host="https://second.example.com")
    # The first client is returned, but a warning is logged.
    assert same_client is not None
    assert any("new config" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# get_langfuse_client
# ---------------------------------------------------------------------------


def test_get_langfuse_client_initializes_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sdk(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert get_langfuse_client() is not None


def test_get_langfuse_client_returns_none_without_creds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert get_langfuse_client() is None


# ---------------------------------------------------------------------------
# is_langfuse_available
# ---------------------------------------------------------------------------


def test_is_langfuse_available_false_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setattr(client_module, "_try_import_sdk", lambda: None)
    assert is_langfuse_available() is False


def test_is_langfuse_available_true_when_both_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sdk(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert is_langfuse_available() is True


# ---------------------------------------------------------------------------
# reset_langfuse_client
# ---------------------------------------------------------------------------


def test_reset_drops_cached_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sdk(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    first = init_langfuse()
    assert first is not None
    reset_langfuse_client()
    second = init_langfuse()
    assert second is not None
    # The fake client is a fresh instance.
    assert first is not second


def test_module_reimport_safe() -> None:
    """Smoke test: the module can be re-imported without error.

    This is a defence against accidental import-time state
    corruption when tests run in the same process.
    """
    reimported = importlib.reload(client_module)
    assert reimported is client_module
