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

"""Lazy Feishu SDK surface used by the app adapter.

The runtime core is ``lark_oapi.channel.FeishuChannel`` (SDK 1.7.0), which
owns the WebSocket lifecycle, dedup, text batching, bot identity, outbound
sending and card-action dispatch. This module is now just the lazy import
boundary: a ``FeishuChannel`` factory wired from :class:`FeishuAppSettings`,
plus the dependency probe and the error types the adapter classifies on.

Importing ``lark_oapi.channel`` pulls in the large generated dispatcher, so
construction is expected to run off the event loop (the adapter does this via
``asyncio.to_thread``). The symbols below are lazily resolved to keep the
channels package importable without ``lark_oapi`` installed.
"""

from __future__ import annotations

import asyncio
import inspect
import importlib.util
import os
from typing import Any

from .feishu_settings import FeishuAppSettings

_WS_PROXY_ENV_KEYS = (
    "WSS_PROXY",
    "wss_proxy",
    "WS_PROXY",
    "ws_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)


class FeishuDependencyMissingError(RuntimeError):
    pass


def feishu_dependencies_available() -> bool:
    return importlib.util.find_spec("lark_oapi") is not None


def build_feishu_channel(settings: FeishuAppSettings) -> Any:
    """Construct a ``FeishuChannel`` from adapter settings.

    Raises :class:`FeishuDependencyMissingError` if ``lark_oapi`` is absent.
    The caller should run this off the event loop (heavy generated import).
    """
    if not feishu_dependencies_available():
        raise FeishuDependencyMissingError("install with `uv sync --locked --extra feishu`")
    _ensure_private_ws_loop()
    from lark_oapi.channel import (  # noqa: PLC0415 — lazy import boundary
        ChannelConfig,
        DedupConfig,
        FeishuChannel,
        SafetyConfig,
        TextBatchConfig,
        TransportConfig,
    )
    from lark_oapi.core.const import FEISHU_DOMAIN, LARK_DOMAIN  # noqa: PLC0415

    domain = LARK_DOMAIN if settings.domain == "lark" else FEISHU_DOMAIN
    config = ChannelConfig(
        app_id=settings.app_id,
        app_secret=settings.app_secret,
        encrypt_key=settings.encrypt_key or None,
        verification_token=settings.verification_token or None,
        domain=domain,
        transport=TransportConfig(kind="ws"),
        safety=SafetyConfig(
            dedup=DedupConfig(
                max_entries=settings.dedup_cache_size,
                ttl_seconds=settings.dedup_ttl_seconds,
            ),
            text_batch=TextBatchConfig(
                delay_ms=int(settings.text_batch_delay_seconds * 1000),
                long_delay_ms=int(settings.text_batch_split_delay_seconds * 1000),
                max_messages=settings.text_batch_max_messages,
                max_chars=settings.text_batch_max_chars,
            ),
        ),
    )
    return FeishuChannel(config=config)


def _ensure_private_ws_loop() -> None:
    """Keep the SDK WS runtime isolated while preserving configured proxies."""
    from lark_oapi.ws import client as ws_client  # noqa: PLC0415

    loop = getattr(ws_client, "loop", None)
    if loop is None or loop.is_closed() or loop.is_running():
        ws_client.loop = asyncio.new_event_loop()
    _ensure_ws_env_proxy(ws_client)


def _ensure_ws_env_proxy(ws_client: Any, environ: dict[str, str] | None = None) -> None:
    environ = environ or os.environ
    if not any(environ.get(key) for key in _WS_PROXY_ENV_KEYS):
        return
    helper = getattr(ws_client, "_ws_connect_kwargs", None)
    if helper is None:
        return
    connect = getattr(getattr(ws_client, "websockets", None), "connect", None)
    try:
        supports_proxy = connect is not None and "proxy" in inspect.signature(connect).parameters
    except (TypeError, ValueError):
        supports_proxy = False
    if not supports_proxy:
        return

    def _ws_connect_kwargs() -> dict[str, bool]:
        return {"proxy": True}

    ws_client._ws_connect_kwargs = _ws_connect_kwargs


def load_error_helpers() -> Any:
    """Lazily resolve the SDK error-classification helpers + error type.

    Returns a namespace with ``classify_error``, ``is_retryable``,
    ``is_format_error``, ``FeishuChannelErrorCode`` and ``FeishuChannelError``.
    """
    from lark_oapi.channel import errors as _errors  # noqa: PLC0415

    return _errors


__all__ = [
    "FeishuDependencyMissingError",
    "build_feishu_channel",
    "feishu_dependencies_available",
    "load_error_helpers",
]
