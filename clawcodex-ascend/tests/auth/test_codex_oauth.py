#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations

import time

import httpx
import pytest

# ``codex_oauth`` must be the CANONICAL module: ``src.auth.codex_oauth`` is
# only a star-import facade, and functions here resolve their helpers through
# ``clawcodex_ext.auth.codex_oauth``'s own module globals — monkeypatching the
# facade (e.g. ``read_codex_tokens`` / ``get_auth_file``) silently misses and
# the real ~/.clawcodex auth.json is read instead.
from clawcodex_ext.auth import codex_oauth
from clawcodex_ext.auth.codex_oauth import (
    CODEX_DEVICE_TOKEN_URL,
    CODEX_DEVICE_USER_CODE_URL,
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_TOKEN_URL,
    CodexAuthError,
    CodexAuthRecord,
    CodexDeviceFlow,
    CodexOAuthTokens,
    exchange_codex_authorization,
    poll_codex_device_flow,
    refresh_codex_tokens,
    resolve_codex_runtime_credentials,
    start_codex_device_flow,
)


def test_start_codex_device_flow_posts_client_id(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return httpx.Response(
            200,
            json={
                "user_code": "ABCD-EFGH",
                "device_auth_id": "device-auth-id",
                "verification_uri": "https://example.com/device",
                "interval": 1,
            },
        )

    monkeypatch.setattr(codex_oauth.httpx, "post", fake_post)

    flow = start_codex_device_flow(timeout_seconds=7)

    assert flow.user_code == "ABCD-EFGH"
    assert flow.device_auth_id == "device-auth-id"
    assert flow.verification_uri == "https://example.com/device"
    assert flow.interval == 3
    assert calls == [
        {
            "url": CODEX_DEVICE_USER_CODE_URL,
            "json": {"client_id": CODEX_OAUTH_CLIENT_ID},
            "headers": {"Content-Type": "application/json"},
            "timeout": 7,
        }
    ]


@pytest.mark.parametrize("status_code", [403, 404])
def test_poll_codex_device_flow_returns_none_while_pending(monkeypatch, status_code: int) -> None:
    monkeypatch.setattr(
        codex_oauth.httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(status_code, json={"error": "authorization_pending"}),
    )

    assert poll_codex_device_flow(CodexDeviceFlow("CODE", "device-id")) is None


def test_poll_codex_device_flow_returns_authorization_payload(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return httpx.Response(200, json={"authorization_code": "auth-code", "code_verifier": "verifier"})

    monkeypatch.setattr(codex_oauth.httpx, "post", fake_post)

    result = poll_codex_device_flow(CodexDeviceFlow("CODE", "device-id"), timeout_seconds=9)

    assert result == {"authorization_code": "auth-code", "code_verifier": "verifier"}
    assert calls == [
        {
            "url": CODEX_DEVICE_TOKEN_URL,
            "json": {"device_auth_id": "device-id", "user_code": "CODE"},
            "headers": {"Content-Type": "application/json"},
            "timeout": 9,
        }
    ]


def test_exchange_codex_authorization_posts_form_payload(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return httpx.Response(
            200,
            json={
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
                "token_type": "Bearer",
                "scope": "codex",
            },
        )

    monkeypatch.setattr(codex_oauth.httpx, "post", fake_post)

    tokens = exchange_codex_authorization("auth-code", "verifier", timeout_seconds=11)

    assert tokens.access_token == "access-token"
    assert tokens.refresh_token == "refresh-token"
    assert tokens.expires_at is not None and tokens.expires_at > time.time()
    assert tokens.scope == "codex"
    assert calls[0]["url"] == CODEX_OAUTH_TOKEN_URL
    assert calls[0]["data"] == {
        "grant_type": "authorization_code",
        "code": "auth-code",
        "redirect_uri": codex_oauth.CODEX_DEVICE_REDIRECT_URI,
        "client_id": CODEX_OAUTH_CLIENT_ID,
        "code_verifier": "verifier",
    }
    assert calls[0]["headers"] == {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    assert calls[0]["timeout"] == 11


def test_refresh_codex_tokens_reuses_existing_refresh_token_when_omitted(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        codex_oauth.httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(200, json={"access_token": "new-access", "expires_in": 3600}),
    )

    tokens = refresh_codex_tokens("old-refresh")

    assert tokens.access_token == "new-access"
    assert tokens.refresh_token == "old-refresh"


def test_refresh_codex_tokens_marks_invalid_grant_as_relogin(monkeypatch) -> None:
    monkeypatch.setattr(
        codex_oauth.httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            400,
            json={
                "error": "invalid_grant",
                "error_description": "Refresh token expired",
            },
        ),
    )

    with pytest.raises(CodexAuthError) as exc_info:
        refresh_codex_tokens("expired-refresh")

    assert exc_info.value.code == "codex_refresh_failed"
    assert exc_info.value.relogin_required is True
    assert "Refresh token expired" in str(exc_info.value)


def test_resolve_codex_runtime_credentials_refreshes_expiring_tokens(
    monkeypatch,
) -> None:
    saved: list[tuple[CodexOAuthTokens, str]] = []
    expiring = CodexOAuthTokens("old-access", "old-refresh", expires_at=time.time() + 10)
    refreshed = CodexOAuthTokens("new-access", "new-refresh", expires_at=time.time() + 3600)

    monkeypatch.setattr(
        codex_oauth,
        "read_codex_tokens",
        lambda: CodexAuthRecord(tokens=expiring, source="test-source", auth_mode="chatgpt"),
    )
    monkeypatch.setattr(codex_oauth, "refresh_codex_tokens", lambda refresh_token: refreshed)
    monkeypatch.setattr(
        codex_oauth,
        "save_codex_tokens",
        lambda tokens, source: saved.append((tokens, source)),
    )

    credentials = resolve_codex_runtime_credentials()

    assert credentials.api_key == "new-access"
    assert credentials.provider == "openai-codex"
    assert credentials.source == "test-source"
    assert saved == [(refreshed, "test-source")]


def test_resolve_codex_runtime_credentials_imports_codex_cli_tokens(
    monkeypatch,
) -> None:
    imported = CodexOAuthTokens("cli-access", "cli-refresh", expires_at=time.time() + 3600)

    monkeypatch.setattr(codex_oauth, "read_codex_tokens", lambda: None)
    monkeypatch.setattr(codex_oauth, "import_codex_cli_tokens", lambda: imported)

    credentials = resolve_codex_runtime_credentials()

    assert credentials.api_key == "cli-access"
    assert credentials.source == "codex-cli"


def test_resolve_codex_runtime_credentials_reimports_newer_codex_cli_tokens(
    monkeypatch,
) -> None:
    stale = CodexOAuthTokens("stale-access", "stale-refresh", expires_at=time.time() + 3600)
    current = CodexOAuthTokens("current-access", "current-refresh", expires_at=time.time() + 3600)

    monkeypatch.setattr(
        codex_oauth,
        "read_codex_tokens",
        lambda: CodexAuthRecord(
            tokens=stale,
            source="codex-cli",
            last_refresh=time.time() - 3600,
        ),
    )
    monkeypatch.setattr(codex_oauth, "codex_cli_auth_is_newer", lambda _last_refresh: True)
    monkeypatch.setattr(codex_oauth, "import_codex_cli_tokens", lambda: current)

    credentials = resolve_codex_runtime_credentials()

    assert credentials.api_key == "current-access"
    assert credentials.source == "codex-cli"


def test_resolve_codex_runtime_credentials_raises_when_not_authenticated(
    monkeypatch,
) -> None:
    monkeypatch.setattr(codex_oauth, "read_codex_tokens", lambda: None)
    monkeypatch.setattr(codex_oauth, "import_codex_cli_tokens", lambda: None)

    with pytest.raises(CodexAuthError) as exc_info:
        resolve_codex_runtime_credentials()

    assert exc_info.value.code == "codex_auth_missing"
    assert exc_info.value.relogin_required is True


def test_get_codex_auth_status_reports_missing_auth(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(codex_oauth, "read_codex_tokens", lambda: None)
    monkeypatch.setattr(codex_oauth, "get_auth_file", lambda: tmp_path / "auth.json")

    status = codex_oauth.get_codex_auth_status()

    assert status.is_authenticated is False
    assert status.auth_file == str(tmp_path / "auth.json")
    assert status.error == "Not logged in"
