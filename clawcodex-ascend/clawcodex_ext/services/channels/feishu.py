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

"""Feishu (飞书) channel implementation.

Feishu bots accept a JSON payload at a webhook URL. When a ``secret`` is
configured in ``ChannelConfig.extra``, the request must include an HMAC
``timestamp`` / ``sign`` pair; this module signs the request when needed
and skips signing otherwise.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from .base import BaseChannel
from .exceptions import TransportError, WebhookSecretMissingError
from .models import ChannelConfig, ChannelMessage
from .transport import (
    DEFAULT_TIMEOUT_SECONDS,
    TransportResponse,
    default_headers,
    encode_json_body,
)


FEISHU_SUCCESS_CODE = 0


def sign_feishu(secret: str, timestamp: str) -> str:
    """Return the Feishu ``sign`` for a given secret and timestamp.

    The signing algorithm is documented at
    https://open.feishu.cn/document/ukTMukTMukTM/ucTM5YjL3ETO24yNxkjN
    and must remain byte-exact.
    """
    if not isinstance(secret, str) or not secret:
        raise WebhookSecretMissingError("feishu secret must be a non-empty string")
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(
        secret.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


class FeishuChannel(BaseChannel):
    def __init__(
        self,
        config: ChannelConfig,
        *,
        transport: "object | None" = None,
        clock: "callable | None" = None,
    ) -> None:
        super().__init__(config, transport=transport)  # type: ignore[arg-type]
        self._clock = clock or time.time

    def format_message(self, message: ChannelMessage) -> tuple[bytes, dict[str, str]]:
        if message.markdown and message.title:
            payload: dict[str, object] = {
                "msg_type": "interactive",
                "card": {
                    "header": {"title": {"tag": "plain_text", "content": message.title}},
                    "elements": [{"tag": "markdown", "content": message.text}],
                },
            }
        else:
            payload = {
                "msg_type": "text",
                "content": json.dumps({"text": message.text}, ensure_ascii=False),
            }

        secret = self._config.extra.get("secret") if self._config.extra else None
        if secret:
            timestamp = str(int(self._clock()))
            payload["timestamp"] = timestamp
            payload["sign"] = sign_feishu(secret, timestamp)
        return encode_json_body(payload), default_headers()

    async def send(self, message: ChannelMessage) -> bool:
        body, headers = self.format_message(message)
        response: TransportResponse = await self._transport.post(
            self._config.webhook_url,
            body,
            headers=headers,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        if response.status != 200:
            return False
        try:
            data = json.loads(response.body.decode("utf-8")) if response.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TransportError(f"feishu returned non-JSON response: {exc}") from exc
        code = data.get("code") if isinstance(data, dict) else None
        if code is None:
            # Some Feishu endpoints return 200 with an empty body; treat as
            # ambiguous but successful since the HTTP call itself worked.
            return True
        return code == FEISHU_SUCCESS_CODE


__all__ = ["FEISHU_SUCCESS_CODE", "FeishuChannel", "sign_feishu"]
