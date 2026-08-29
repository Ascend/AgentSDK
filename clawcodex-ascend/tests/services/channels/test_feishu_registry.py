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

"""Feishu registry mode dispatch tests."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import pytest

from clawcodex_ext.services.channels.feishu_app import FeishuAppChannelAdapter
from clawcodex_ext.services.channels.models import ChannelConfig, ChannelType
from clawcodex_ext.services.channels.registry import WebhookChannelAdapter, build_default_registry


def test_feishu_registry_uses_webhook_when_legacy_webhook_url_present() -> None:
    cfg = ChannelConfig(
        type=ChannelType.FEISHU,
        webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/abcdef",
        name="feishu",
    )

    adapter = build_default_registry().create(cfg)

    assert isinstance(adapter, WebhookChannelAdapter)


def test_feishu_registry_uses_app_adapter_for_websocket_mode() -> None:
    cfg = ChannelConfig(
        type=ChannelType.FEISHU,
        webhook_url="",
        name="feishu",
        extra={
            "connection_mode": "websocket",
            "app_id": "cli_app",
            "app_secret": "secret",
            "allowed_user_open_id": "ou_allowed",
        },
    )

    adapter = build_default_registry().create(cfg)

    assert isinstance(adapter, FeishuAppChannelAdapter)


def test_channel_config_rejects_unknown_feishu_mode() -> None:
    with pytest.raises(ValueError, match="connection_mode"):
        ChannelConfig(
            type=ChannelType.FEISHU,
            webhook_url="",
            name="feishu",
            extra={"connection_mode": "sideways"},
        )
