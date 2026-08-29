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

"""Feishu App channel settings tests."""
# pylint: disable=no-name-in-module

from __future__ import annotations

from clawcodex_ext.services.channels.feishu_settings import FeishuAppSettings
from clawcodex_ext.services.channels.models import ChannelConfig, ChannelType


def _websocket_config(extra: dict | None = None) -> ChannelConfig:
    payload = {
        "connection_mode": "websocket",
        "app_id": "cli_app",
        "app_secret": "cli_secret",
        "domain": "feishu",
        "allowed_user_open_id": "ou_allowed",
        "bot_open_id": "ou_bot",
    }
    if extra:
        payload.update(extra)
    return ChannelConfig(
        type=ChannelType.FEISHU,
        webhook_url="",
        name="feishu",
        extra=payload,
    )


def test_feishu_settings_reads_extra_and_env() -> None:
    cfg = _websocket_config(
        {
            "app_id": "${FEISHU_APP_ID}",
            "app_secret": "${FEISHU_APP_SECRET}",
            "encrypt_key": "${FEISHU_ENCRYPT_KEY}",
            "verification_token": "${FEISHU_VERIFICATION_TOKEN}",
            "allowed_user_open_id": "${FEISHU_ALLOWED_USER_OPEN_ID}",
        }
    )

    settings = FeishuAppSettings.from_config(
        cfg,
        environ={
            "FEISHU_APP_ID": "cli_env_app",
            "FEISHU_APP_SECRET": "env_secret",
            "FEISHU_ENCRYPT_KEY": "env_encrypt_key",
            "FEISHU_VERIFICATION_TOKEN": "env_verification_token",
            "FEISHU_ALLOWED_USER_OPEN_ID": "ou_env_user",
        },
    )

    assert settings.connection_mode == "websocket"
    assert settings.app_id == "cli_env_app"
    assert settings.app_secret == "env_secret"
    assert settings.encrypt_key == "env_encrypt_key"
    assert settings.verification_token == "env_verification_token"
    assert settings.allowed_user_open_id == "ou_env_user"
    assert settings.domain == "feishu"


def test_feishu_settings_ignores_sdk_owned_ws_tuning_fields() -> None:
    cfg = _websocket_config(
        {
            "websocket": {
                "ws_reconnect_interval": "180",
                "ws_ping_interval": "",
                "ws_ping_timeout": "12.5",
            },
            "batching": {
                "text_batch_delay_seconds": "0.2",
                "text_batch_max_messages": "3",
                "text_batch_max_chars": "1200",
            },
            "send": {
                "sdk_send_attempts": "2",
                "sdk_send_backoff_base_seconds": "0.1",
                "per_origin_serial": False,
            },
        }
    )

    settings = FeishuAppSettings.from_config(cfg)

    # WS tuning is server-authoritative in the SDK; none of it is surfaced.
    assert not hasattr(settings, "connect_attempts")
    assert not hasattr(settings, "ws_reconnect_interval")
    assert not hasattr(settings, "ws_ping_interval")
    assert not hasattr(settings, "ws_ping_timeout")
    assert not hasattr(settings, "per_origin_serial")
    assert settings.text_batch_delay_seconds == 0.2
    assert settings.text_batch_max_messages == 3
    assert settings.text_batch_max_chars == 1200
    assert settings.sdk_send_attempts == 2
    assert settings.sdk_send_backoff_base_seconds == 0.1


def test_feishu_settings_reads_startup_connect_timeout() -> None:
    cfg = _websocket_config(
        {
            "websocket": {
                "startup_connect_timeout_seconds": "5.5",
            },
        }
    )

    settings = FeishuAppSettings.from_config(cfg)

    assert settings.startup_connect_timeout_seconds == 5.5


def test_feishu_reactions_env_overrides_channel_config() -> None:
    cfg = _websocket_config({"reactions": False})

    assert FeishuAppSettings.from_config(cfg, environ={}).reactions_enabled is False
    assert (
        FeishuAppSettings.from_config(
            cfg,
            environ={"FEISHU_REACTIONS": "true"},
        ).reactions_enabled
        is True
    )


def test_feishu_settings_rejects_missing_credentials_in_websocket_mode() -> None:
    cfg = _websocket_config({"app_id": "", "app_secret": ""})

    settings = FeishuAppSettings.from_config(cfg)
    errors = settings.validation_errors()

    assert "app_id is required for feishu websocket mode" in errors
    assert "app_secret is required for feishu websocket mode" in errors
