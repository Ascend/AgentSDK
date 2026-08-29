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

"""Conditional import API, payload rewriting and SSRF address validation."""

from __future__ import annotations

import json
import pytest
from pydantic import ValidationError
from extensions.visualizer.import_router import (
    ImportRequest,
    _rewrite_imported_jsonl,
    _validate_import_url,
)


def test_import_request_rejects_unknown_format() -> None:
    with pytest.raises(ValidationError):
        ImportRequest(url="https://example.test/session", format="yaml")


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.test/session", "http:///x"],
)
def test_import_url_rejects_non_http_or_missing_host(url: str) -> None:
    with pytest.raises(ValueError):
        _validate_import_url(url)


def test_import_url_rejects_private_resolution(monkeypatch) -> None:
    monkeypatch.setattr(
        "extensions.visualizer.import_router._is_private_host",
        lambda hostname: hostname == "private.test",
    )

    with pytest.raises(ValueError, match="private/local"):
        _validate_import_url("https://private.test/session")


def test_jsonl_import_rewrites_legacy_tool_messages() -> None:
    source = "\n".join(
        [
            json.dumps({"role": "user", "content": "run"}),
            json.dumps(
                {
                    "role": "assistant",
                    "content": "working",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "Read", "arguments": {"path": "a"}},
                        }
                    ],
                }
            ),
            json.dumps({"role": "tool", "tool_call_id": "call-1", "content": "ok"}),
        ]
    )

    rewritten = [json.loads(line) for line in _rewrite_imported_jsonl(source).splitlines()]

    assert rewritten[0]["origin"] == "import"
    assert any(block["type"] == "tool_use" for block in rewritten[1]["content"])
    assert rewritten[2]["content"][0]["type"] == "tool_result"
    assert rewritten[2]["origin"] == "tool_result"


def test_jsonl_import_marks_later_user_messages_as_user_origin() -> None:
    source = "\n".join(
        [
            json.dumps({"role": "user", "content": "first"}),
            json.dumps({"role": "assistant", "content": "answer"}),
            json.dumps({"role": "user", "content": "follow-up"}),
        ]
    )

    rewritten = [json.loads(line) for line in _rewrite_imported_jsonl(source).splitlines()]

    assert [message["origin"] for message in rewritten] == [
        "import",
        "agent",
        "user",
    ]


@pytest.mark.parametrize(
    "url",
    [
        "http://0.0.0.0/data.json",
        "http://100.64.0.1/data.json",
        "http://224.0.0.1/data.json",
        "http://[::1]/data.json",
    ],
)
def test_import_url_rejects_non_global_literal_addresses(url: str) -> None:
    with pytest.raises(ValueError, match="private/local"):
        _validate_import_url(url)


def test_import_url_rejects_any_non_global_resolved_address(monkeypatch) -> None:
    import socket

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        ],
    )
    with pytest.raises(ValueError, match="private/local"):
        _validate_import_url("https://example.invalid/data.json")


def test_import_url_accepts_global_literal_address() -> None:
    url = "https://93.184.216.34/data.json"
    assert _validate_import_url(url) == url


class TestImportAPI:
    def test_import_disabled_by_default(self, tmp_path):
        from extensions.visualizer.server import create_app

        app = create_app(sessions_dir=tmp_path, allow_import=False)
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post(
            "/api/viz/import",
            json={
                "url": "https://example.com/data.json",
            },
        )
        assert resp.status_code == 404 or resp.status_code == 405

    def test_import_enabled(self, client):
        resp = client.post(
            "/api/viz/import",
            json={
                "url": "https://example.com/data.json",
            },
        )
        assert resp.status_code in (202, 400, 403)
