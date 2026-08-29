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

"""Phase 0 smoke test: WS + SSE deps importable, asyncio API present."""

from __future__ import annotations


def test_websockets_asyncio_paths_importable() -> None:
    """Per A1: ``websockets >= 14.0`` exposes ``asyncio.{client,server}``."""
    import websockets.asyncio.client as ws_client
    import websockets.asyncio.server as ws_server

    assert hasattr(ws_client, "connect"), "websockets.asyncio.client.connect missing"
    assert hasattr(ws_server, "serve"), "websockets.asyncio.server.serve missing"


def test_websockets_top_level_alias_points_at_asyncio_api() -> None:
    """Top-level ``websockets.connect`` aliases the asyncio API on >=14.0."""
    import websockets

    assert hasattr(websockets, "connect"), "websockets.connect missing"
    assert hasattr(websockets, "serve"), "websockets.serve missing"


def test_httpx_sse_importable() -> None:
    import httpx_sse

    assert hasattr(httpx_sse, "aconnect_sse"), "httpx_sse.aconnect_sse missing"


def test_httpx_transitively_available() -> None:
    """httpx is pulled in by the anthropic SDK; we don't declare it."""
    import httpx

    assert hasattr(httpx, "AsyncClient")
