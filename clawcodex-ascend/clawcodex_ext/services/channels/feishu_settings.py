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

"""Settings parser for the Feishu App channel.

The WebSocket runtime, dedup, text batching, send retry backoff and bot
identity fetch are owned by ``lark_oapi.channel.FeishuChannel`` (SDK 1.7.0).
Only the knobs that still mean something at the adapter boundary are kept
here; the rest were folded into the SDK and removed (no released versions
carry them, so there is no backward-compat read path).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .models import ChannelConfig


@dataclass(frozen=True)
class FeishuAppSettings:
    channel_id: str
    connection_mode: str
    app_id: str = ""
    app_secret: str = ""
    encrypt_key: str = ""
    verification_token: str = ""
    domain: str = "feishu"
    allowed_user_open_id: str = ""
    bot_open_id: str | None = None
    bot_name: str | None = None
    # Adapter-level first-connect budget for ``connect_until_ready``. WS
    # ping/reconnect tuning is server-authoritative in the SDK and not exposed.
    startup_connect_timeout_seconds: float = 120.0
    # Fed into the SDK ``DedupConfig`` / ``TextBatchConfig``.
    dedup_cache_size: int = 2048
    dedup_ttl_seconds: int = 86400
    text_batch_delay_seconds: float = 0.6
    text_batch_split_delay_seconds: float = 1.2
    text_batch_max_messages: int = 8
    text_batch_max_chars: int = 4000
    # Adapter-level short retry around ``channel.send``.
    sdk_send_attempts: int = 3
    sdk_send_backoff_base_seconds: float = 1.0
    sdk_send_timeout_seconds: float = 30.0
    # Approval card business logic (fork-owned, not in the SDK).
    approval_cards_enabled: bool = True
    action_token_ttl_seconds: int = 900
    decision_ttl_seconds: int = 600
    reactions_enabled: bool = True

    @classmethod
    def from_config(
        cls,
        config: ChannelConfig,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> FeishuAppSettings:
        env = environ if environ is not None else os.environ
        extra = dict(config.extra or {})
        websocket = _mapping(extra.get("websocket"))
        batching = _mapping(extra.get("batching"))
        send = _mapping(extra.get("send"))
        approval_cards = _mapping(extra.get("approval_cards"))
        mode = str(extra.get("connection_mode") or "").strip().lower()
        if not mode and config.webhook_url:
            mode = "webhook"
        if not mode:
            mode = "websocket"
        return cls(
            channel_id=config.name,
            connection_mode=mode,
            app_id=_resolve(extra.get("app_id"), env, "FEISHU_APP_ID"),
            app_secret=_resolve(extra.get("app_secret"), env, "FEISHU_APP_SECRET"),
            encrypt_key=_resolve(extra.get("encrypt_key"), env, "FEISHU_ENCRYPT_KEY"),
            verification_token=_resolve(extra.get("verification_token"), env, "FEISHU_VERIFICATION_TOKEN"),
            domain=str(_resolve(extra.get("domain"), env, "FEISHU_DOMAIN") or "feishu").lower(),
            allowed_user_open_id=_resolve(extra.get("allowed_user_open_id"), env, "FEISHU_ALLOWED_USER_OPEN_ID"),
            bot_open_id=_none_if_empty(_resolve(extra.get("bot_open_id"), env, "FEISHU_BOT_OPEN_ID")),
            bot_name=_none_if_empty(str(extra.get("bot_name") or "")),
            startup_connect_timeout_seconds=_as_float(websocket.get("startup_connect_timeout_seconds"), 120.0),
            dedup_cache_size=_as_int(batching.get("dedup_cache_size"), 2048),
            dedup_ttl_seconds=_as_int(batching.get("dedup_ttl_seconds"), 86400),
            text_batch_delay_seconds=_as_float(batching.get("text_batch_delay_seconds"), 0.6),
            text_batch_split_delay_seconds=_as_float(batching.get("text_batch_split_delay_seconds"), 1.2),
            text_batch_max_messages=_as_int(batching.get("text_batch_max_messages"), 8),
            text_batch_max_chars=_as_int(batching.get("text_batch_max_chars"), 4000),
            sdk_send_attempts=_as_int(send.get("sdk_send_attempts"), 3),
            sdk_send_backoff_base_seconds=_as_float(send.get("sdk_send_backoff_base_seconds"), 1.0),
            sdk_send_timeout_seconds=_as_float(send.get("sdk_send_timeout_seconds"), 30.0),
            approval_cards_enabled=_as_bool(approval_cards.get("enabled"), True),
            action_token_ttl_seconds=_as_int(approval_cards.get("action_token_ttl_seconds"), 900),
            decision_ttl_seconds=_as_int(approval_cards.get("decision_ttl_seconds"), 600),
            reactions_enabled=_reaction_enabled(extra.get("reactions"), env),
        )

    def validation_errors(self) -> list[str]:
        if self.connection_mode == "webhook":
            return []
        errors: list[str] = []
        if not self.app_id:
            errors.append("app_id is required for feishu websocket mode")
        if not self.app_secret:
            errors.append("app_secret is required for feishu websocket mode")
        if self.domain not in {"feishu", "lark"}:
            errors.append("domain must be feishu or lark")
        return errors


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _resolve(value: Any, env: Mapping[str, str], env_name: str) -> str:
    text = "" if value is None else str(value).strip()
    if text.startswith("${") and text.endswith("}"):
        return str(env.get(text[2:-1], "")).strip()
    if text:
        return text
    return str(env.get(env_name, "")).strip()


def _none_if_empty(value: str) -> str | None:
    value = value.strip()
    return value or None


def _as_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _reaction_enabled(config_value: Any, env: Mapping[str, str]) -> bool:
    """Environment explicitly overrides channel config; default is enabled."""
    if "FEISHU_REACTIONS" in env:
        return _as_bool(env.get("FEISHU_REACTIONS"), True)
    if isinstance(config_value, dict):
        config_value = config_value.get("enabled")
    return _as_bool(config_value, True)


__all__ = ["FeishuAppSettings"]
