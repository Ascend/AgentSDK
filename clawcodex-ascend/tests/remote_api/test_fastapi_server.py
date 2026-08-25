#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Tests for the optional FastAPI transport and extension-local imports."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from extensions.remote_api import server
from extensions.remote_api.core import RemoteAPIConfig, RemoteAPIError


def test_fastapi_transport_has_clear_optional_dependency_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(server, "FastAPI", None)

    with pytest.raises(RuntimeError, match="FastAPI transport is unavailable"):
        server.create_app(RemoteAPIConfig(tmp_path))


def test_fastapi_transport_uses_extension_core_contract() -> None:
    from extensions.remote_api import core

    assert server.RemoteAPIConfig is core.RemoteAPIConfig
    assert server.RemoteAPIService is core.RemoteAPIService


def test_proactive_focus_import_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "extensions.remote_api.state_reporter",
        None,
    )

    with pytest.raises(RemoteAPIError) as error:
        server._set_proactive_focus("working")

    assert error.value.status_code == 500
    assert error.value.to_payload()["error"]["message"] == ("proactive focus is unavailable")


@pytest.mark.asyncio
async def test_json_reader_only_maps_decode_errors_to_invalid_request() -> None:
    class InvalidJSONRequest:
        async def json(self) -> object:
            raise ValueError("invalid json")

    class BrokenRequest:
        async def json(self) -> object:
            raise RuntimeError("client disconnected")

    with pytest.raises(RemoteAPIError) as invalid:
        await server._read_json_object(InvalidJSONRequest())
    assert invalid.value.status_code == 400

    with pytest.raises(RuntimeError, match="client disconnected"):
        await server._read_json_object(BrokenRequest())


@pytest.mark.skipif(server.FastAPI is None, reason="optional FastAPI dependency unavailable")
def test_fastapi_routes_auth_health_and_compatible_errors(tmp_path: Path) -> None:
    testclient = pytest.importorskip("fastapi.testclient")
    app = server.create_app(RemoteAPIConfig(tmp_path, api_key="secret", model="model"))
    client = testclient.TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["workspace"] == str(tmp_path)

    unauthorized = client.get("/v1/models")
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["code"] == "unauthorized"

    models = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "model"

    malformed = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
        content="not-json",
    )
    assert malformed.status_code == 400
    assert malformed.json()["error"]["code"] == "invalid_request"

    missing = client.get("/missing")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"
