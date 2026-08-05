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

"""Feishu SDK factory wiring tests."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import asyncio
import sys
from types import ModuleType

import pytest

from clawcodex_ext.services.channels.feishu_sdk import _ensure_private_ws_loop, build_feishu_channel
from clawcodex_ext.services.channels.feishu_settings import FeishuAppSettings


def _settings(**overrides) -> FeishuAppSettings:
    values = {
        "channel_id": "feishu",
        "connection_mode": "websocket",
        "app_id": "cli_app",
        "app_secret": "secret",
        "encrypt_key": "encrypt-key",
        "verification_token": "verification-token",
    }
    values.update(overrides)
    return FeishuAppSettings(**values)


def test_build_feishu_channel_passes_event_security_fields_to_sdk() -> None:
    channel = build_feishu_channel(_settings())

    assert channel.config.encrypt_key == "encrypt-key"
    assert channel.config.verification_token == "verification-token"


@pytest.mark.asyncio
async def test_build_feishu_channel_replaces_sdk_ws_loop_when_imported_on_running_loop(
    monkeypatch,
) -> None:
    from lark_oapi.ws import client as ws_client

    running_loop = asyncio.get_running_loop()
    original_loop = ws_client.loop
    monkeypatch.setattr(ws_client, "loop", running_loop)

    try:
        build_feishu_channel(_settings())

        assert ws_client.loop is not running_loop
        assert not ws_client.loop.is_running()
        assert not ws_client.loop.is_closed()
    finally:
        monkeypatch.setattr(ws_client, "loop", original_loop)


def test_feishu_sdk_ws_runtime_preserves_websocket_env_proxy(monkeypatch) -> None:
    root_module = ModuleType("lark_oapi")
    ws_module = ModuleType("lark_oapi.ws")
    client_module = ModuleType("lark_oapi.ws.client")
    loop = asyncio.new_event_loop()

    async def _connect(_uri, *, proxy=True):
        return proxy

    client_module.loop = loop
    client_module.websockets = type("Websockets", (), {"connect": _connect})
    client_module._ws_connect_kwargs = lambda: {"proxy": None}
    ws_module.client = client_module
    monkeypatch.setitem(sys.modules, "lark_oapi", root_module)
    monkeypatch.setitem(sys.modules, "lark_oapi.ws", ws_module)
    monkeypatch.setitem(sys.modules, "lark_oapi.ws.client", client_module)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")

    try:
        _ensure_private_ws_loop()

        assert client_module._ws_connect_kwargs() == {"proxy": True}  # pylint: disable=no-member
    finally:
        loop.close()
