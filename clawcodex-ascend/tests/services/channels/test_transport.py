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

"""validate_webhook_url / redact_webhook_url / encode_json_body tests."""

from __future__ import annotations

import json

import pytest

from src.services.channels import (
    InvalidWebhookURLError,
    encode_json_body,
    redact_webhook_url,
    validate_webhook_url,
)


def test_validate_webhook_url_accepts_https_with_public_host() -> None:
    # Use a hostname that resolves to a public IP; the helper must accept it.
    url = "https://hooks.example.com/services/abcdef0123456789"
    # If DNS lookup fails in the sandbox we still want the unit test to
    # pass; that's why the helper exposes ``resolve_host=False``.
    assert validate_webhook_url(url, resolve_host=False) == "https://hooks.example.com/services/abcdef0123456789"


def test_validate_webhook_url_rejects_http_by_default() -> None:
    with pytest.raises(InvalidWebhookURLError):
        validate_webhook_url(
            "http://hooks.example.com/x",
            resolve_host=False,
        )


def test_validate_webhook_url_accepts_http_with_allow_http() -> None:
    url = "http://hooks.example.com/x"
    assert validate_webhook_url(url, allow_http=True, resolve_host=False) == url


def test_validate_webhook_url_rejects_empty() -> None:
    with pytest.raises(InvalidWebhookURLError):
        validate_webhook_url("")


def test_validate_webhook_url_rejects_non_string() -> None:
    with pytest.raises(InvalidWebhookURLError):
        validate_webhook_url(None)  # type: ignore[arg-type]


def test_validate_webhook_url_rejects_unknown_scheme() -> None:
    with pytest.raises(InvalidWebhookURLError):
        validate_webhook_url("ftp://hooks.example.com/x", resolve_host=False)


def test_validate_webhook_url_rejects_empty_hostname() -> None:
    with pytest.raises(InvalidWebhookURLError):
        validate_webhook_url("https:///path", resolve_host=False)


def test_validate_webhook_url_rejects_loopback_literal_by_default() -> None:
    with pytest.raises(InvalidWebhookURLError):
        validate_webhook_url("https://127.0.0.1/x", resolve_host=False)


def test_validate_webhook_url_rejects_loopback_hostname_by_default() -> None:
    with pytest.raises(InvalidWebhookURLError):
        validate_webhook_url("https://localhost/x", resolve_host=False)


def test_validate_webhook_url_accepts_loopback_with_allow_loopback() -> None:
    url = "https://localhost/x"
    assert validate_webhook_url(url, allow_loopback=True, resolve_host=False) == url


def test_validate_webhook_url_rejects_link_local_literal() -> None:
    with pytest.raises(InvalidWebhookURLError):
        validate_webhook_url("https://169.254.169.254/x", resolve_host=False)


def test_validate_webhook_url_rejects_multicast_literal() -> None:
    with pytest.raises(InvalidWebhookURLError):
        validate_webhook_url("https://224.0.0.1/x", resolve_host=False)


def test_validate_webhook_url_rejects_unspecified_literal() -> None:
    with pytest.raises(InvalidWebhookURLError):
        validate_webhook_url("https://0.0.0.0/x", resolve_host=False)


def test_validate_webhook_url_rejects_private_literal() -> None:
    with pytest.raises(InvalidWebhookURLError):
        validate_webhook_url("https://10.0.0.5/x", resolve_host=False)
    with pytest.raises(InvalidWebhookURLError):
        validate_webhook_url("https://192.168.1.1/x", resolve_host=False)


def test_validate_webhook_url_requires_separate_private_opt_in() -> None:
    url = "https://10.0.0.5/x"
    with pytest.raises(InvalidWebhookURLError):
        validate_webhook_url(url, allow_loopback=True, resolve_host=False)
    assert validate_webhook_url(url, allow_private=True, resolve_host=False) == url


def test_validate_webhook_url_rejects_unresolvable_host(monkeypatch) -> None:
    # Override the conftest's DNS stub so resolution raises gaierror.
    import socket as _socket

    def _raise(*args, **kwargs):
        raise _socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(_socket, "getaddrinfo", _raise)

    # A DNS name that does not resolve must raise InvalidWebhookURLError
    # (not socket.gaierror leaking through).
    with pytest.raises(InvalidWebhookURLError):
        validate_webhook_url("https://this-host-should-not-exist-clawcodex-12345.example/x")


def test_redact_webhook_url_strips_query_string() -> None:
    redacted = redact_webhook_url("https://hooks.example.com/services/T0/B0/abcdef0123456789?token=secret")
    assert "token=secret" not in redacted
    assert "?" not in redacted


def test_redact_webhook_url_masks_token_path_segment() -> None:
    redacted = redact_webhook_url("https://hooks.example.com/services/T0/B0/abcdef0123456789")
    assert redacted.endswith("/***")
    assert "abcdef0123456789" not in redacted


def test_redact_webhook_url_preserves_scheme_and_host() -> None:
    redacted = redact_webhook_url("https://hooks.example.com/api/webhooks/1/abcdef0123456789")
    assert redacted.startswith("https://hooks.example.com/api/webhooks/1/")


@pytest.mark.parametrize(
    ("url", "expected_authority"),
    [
        ("https://user:password@example.com/hooks/abcdef012345", "example.com"),
        ("https://user@example.com/hooks/abcdef012345", "example.com"),
        ("https://user:password@[2001:db8::1]:8443/hooks/abcdef012345", "[2001:db8::1]:8443"),
    ],
)
def test_redact_webhook_url_removes_userinfo(url: str, expected_authority: str) -> None:
    redacted = redact_webhook_url(url)

    assert "user" not in redacted
    assert "password" not in redacted
    assert f"//{expected_authority}/" in redacted


def test_redact_webhook_url_handles_non_token_path() -> None:
    # Path segments shorter than 6 alphanumeric chars are kept verbatim.
    url = "https://hooks.example.com/api/v1/run"
    assert redact_webhook_url(url) == url


def test_redact_webhook_url_empty_input() -> None:
    assert redact_webhook_url("") == ""
    assert redact_webhook_url(None) == ""  # type: ignore[arg-type]


def test_encode_json_body_compact_and_unicode_safe() -> None:
    payload = {"text": "你好", "level": "info"}
    body = encode_json_body(payload)
    # Compact separators: no spaces between separators.
    assert b", " not in body
    assert b": " not in body
    # Unicode preserved (ensure_ascii=False).
    assert "你好".encode("utf-8") in body
    # And it must round-trip back to the original dict.
    assert json.loads(body.decode("utf-8")) == payload
