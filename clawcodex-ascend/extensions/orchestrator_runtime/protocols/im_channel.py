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

"""IM channel protocol (Feishu / Slack / Telegram, etc.).

The orchestrator binds one channel per origin. The channel owns
poll/websocket transport and platform-specific card rendering.
"""

from __future__ import annotations
# pylint: disable=W2301

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable


@dataclass(slots=True)
class ImInbound:
    """Mirrors ``clawcodex_ext.services.im_gateway.models.InboundMessage``
    shape (structurally).
    """

    origin: str
    text: str
    issue_id: str | None = None
    thread_id: str | None = None
    sender_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ImOutbound:
    """Mirrors ``clawcodex_ext.services.im_gateway.models.OutboundMessage``."""

    origin: str
    text: str
    issue_id: str | None = None
    card: dict[str, Any] | None = None  # platform-specific card payload


@runtime_checkable
class ImChannel(Protocol):
    """One integration with an IM platform (Feishu, Slack, Telegram…).

    The orchestrator wires one channel per origin; each channel handles
    its own poll/websocket transport.
    """

    channel_id: str

    async def deliver(self, message: ImOutbound) -> None:
        """Send ``message`` to the platform; raise on transport failure."""
        ...

    async def listen(self) -> AsyncIterator[ImInbound]:
        """Async generator of inbound messages from the platform."""
        ...

    async def close(self) -> None:
        """Release transport resources (websockets, etc.)."""
        ...


@runtime_checkable
class ImCommandRouter(Protocol):
    """Dispatch semantic commands (RETRY / FOLLOWUP / PAUSE / RESUME …)
    into orchestrator operations.

    Implementations may differ per channel; the orchestrator only depends
    on the contract. Mirrors ``clawcodex_ext.messaging.semantics.CommandRouter``.
    """

    async def dispatch(self, inbound: ImInbound) -> ImOutbound | None:
        """If ``inbound`` carries a recognised command, return response;
        otherwise return ``None`` (pass-through to orchestrator normal flow).
        """
        ...


__all__ = [
    "ImChannel",
    "ImCommandRouter",
    "ImInbound",
    "ImOutbound",
]
