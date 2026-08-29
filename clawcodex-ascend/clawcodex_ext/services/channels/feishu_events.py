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

"""Translate SDK inbound events into gateway ``InboundMessage``.

``lark_oapi.channel.FeishuChannel`` already deserializes the WebSocket event,
deduplicates, parses text/post content into ``content_text`` and drops
self-echo (``InboundConfig.drop_self_sent``). This module is the thin
bridge from the SDK's :class:`~lark_oapi.channel.types.InboundMessage` to the
gateway's :class:`InboundMessage`, applying the V1 p2p-only / optional
allowlist admission on the way.
"""

from __future__ import annotations

# Sibling clawcodex_ext packages (im_gateway) are migrated in separate branches;
# suppress E0611 until the full series lands.
# pylint: disable=no-name-in-module

import logging
from typing import Any

from .feishu_settings import FeishuAppSettings

logger = logging.getLogger(__name__)


def translate_inbound(inbound: Any, settings: FeishuAppSettings) -> Any | None:
    """Map an SDK ``InboundMessage`` to a gateway ``InboundMessage``.

    Returns ``None`` when the event is dropped (non-p2p, disallowed sender,
    empty text, missing ids). ``inbound`` is duck-typed so tests can pass a
    lightweight stand-in.
    """
    chat_type = str(_get(_get(inbound, "conversation"), "chat_type") or "").lower()
    if chat_type not in {"p2p", "private"}:
        logger.debug("feishu event dropped: not p2p (chat_type=%s)", chat_type)
        return None
    open_id = str(_get(_get(inbound, "sender"), "open_id") or "")
    if not open_id:
        logger.debug("feishu event dropped: no sender open_id")
        return None
    # SDK ``drop_self_sent`` already filters bot echo, but keep a defensive
    # check in case bot_open_id was resolved late / overridden in config.
    if settings.bot_open_id and open_id == settings.bot_open_id:
        return None
    if settings.allowed_user_open_id and open_id != settings.allowed_user_open_id:
        logger.debug("feishu event dropped: sender not in allowlist: %s", open_id[:16])
        return None
    text = str(_get(inbound, "content_text") or "").strip()
    if not text:
        logger.debug(
            "feishu event dropped: empty text (message_id=%s)",
            str(_get(inbound, "id") or "")[:16],
        )
        return None
    message_id = str(_get(inbound, "id") or "")
    chat_id = str(_get(_get(inbound, "conversation"), "chat_id") or "")
    if not message_id or not chat_id:
        logger.debug("feishu event dropped: missing message_id/chat_id")
        return None
    from clawcodex_ext.services.im_gateway.models import InboundMessage  # noqa: PLC0415

    logger.info(
        "feishu inbound normalized: from=%s msg_id=%s chat=%s len=%d",
        open_id[:16],
        message_id[:16],
        chat_id[:16],
        len(text),
    )
    return InboundMessage(
        origin=f"feishu:dm:{settings.app_id}:{open_id}",
        text=text,
        message_id=message_id,
        channel=settings.channel_id,
        context_token=chat_id,
        from_user_id=open_id,
        raw={
            "message_id": message_id,
            "chat_id": chat_id,
            "open_id": open_id,
            "create_time": _get(inbound, "create_time"),
            "raw_content_type": _get(inbound, "raw_content_type"),
        },
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute-or-key accessor bridging SDK model objects and raw dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


__all__ = ["translate_inbound"]
