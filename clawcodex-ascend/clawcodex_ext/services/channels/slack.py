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

"""Slack channel implementation (incoming webhooks)."""

from __future__ import annotations

import json

from .base import BaseChannel
from .models import ChannelMessage
from .transport import (
    DEFAULT_TIMEOUT_SECONDS,
    TransportResponse,
    default_headers,
    encode_json_body,
)


class SlackChannel(BaseChannel):
    def format_message(self, message: ChannelMessage) -> tuple[bytes, dict[str, str]]:
        if message.markdown and message.title:
            payload: dict[str, object] = {
                "text": message.title,
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{message.title}*\n{message.text}",
                        },
                    }
                ],
            }
        else:
            payload = {"text": message.text}
        return encode_json_body(payload), default_headers()

    async def send(self, message: ChannelMessage) -> bool:
        body, headers = self.format_message(message)
        response: TransportResponse = await self._transport.post(
            self._config.webhook_url,
            body,
            headers=headers,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        if response.body:
            try:
                decoded = json.loads(response.body.decode("utf-8"))
                if isinstance(decoded, dict):
                    # An ``ok`` field is authoritative when present.
                    if "ok" in decoded:
                        return bool(decoded.get("ok"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        if 200 <= response.status < 300:
            return True
        return False


__all__ = ["SlackChannel"]
