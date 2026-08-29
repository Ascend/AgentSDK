#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Null/dry-run channel used in tests and as a safe default."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from .base import BaseChannel
from .models import ChannelConfig, ChannelMessage
from .transport import (
    ChannelTransport,
    DEFAULT_TIMEOUT_SECONDS,
    TransportResponse,
    encode_json_body,
    default_headers,
)


@dataclass
class RecordedSend:
    message: ChannelMessage
    body: bytes
    headers: dict[str, str]


class NullChannel(BaseChannel):
    """A channel that never calls the network.

    Every :meth:`send` records the serialized body and headers in an
    in-memory log. Useful for tests and for safe-by-default deployments
    where no real channel is configured.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        config: ChannelConfig,
        *,
        transport: ChannelTransport | None = None,
    ) -> None:
        # The null channel does not touch the network, so it skips the
        # webhook URL safety check that ``BaseChannel.__init__`` runs.
        self._config = config
        self._transport = transport or _NullTransport()
        self._owns_transport = transport is None
        self._lock = threading.RLock()
        self._log: list[RecordedSend] = []

    @property
    def log(self) -> list[RecordedSend]:
        with self._lock:
            return list(self._log)

    def clear(self) -> None:
        with self._lock:
            self._log.clear()

    def format_message(self, message: ChannelMessage) -> tuple[bytes, dict[str, str]]:
        body = encode_json_body({"text": message.text, "level": message.level.value})
        return body, default_headers()

    async def send(self, message: ChannelMessage) -> bool:
        body, headers = self.format_message(message)
        with self._lock:
            self._log.append(RecordedSend(message=message, body=body, headers=headers))
        return True


class _NullTransport(ChannelTransport):
    """Transport that always returns success without touching the network."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.calls: list[dict[str, Any]] = []

    async def post(
        self,
        url: str,
        body: bytes,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> TransportResponse:
        self.calls.append({"url": url, "body": body, "headers": dict(headers or {}), "timeout": timeout})
        return TransportResponse(status=self.status, body=b"", headers={})


__all__ = ["NullChannel", "RecordedSend", "_NullTransport"]
