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

"""Channels domain exceptions."""


class ChannelError(RuntimeError):
    """Base error for channel operations."""


class InvalidWebhookURLError(ChannelError):
    """Raised when a webhook URL is malformed, has the wrong scheme, or
    points at a private network address when public-only is enforced.
    """


class WebhookSecretMissingError(ChannelError):
    """Raised when a channel requires a signing secret (Feishu, WeChat) but
    none was configured.
    """


class TransportError(ChannelError):
    """Raised when the underlying HTTP transport fails."""


class ChannelNotFoundError(ChannelError):
    """Raised when sending to a channel name that is not registered."""


class ChannelDisabledError(ChannelError):
    """Raised when sending to a channel that is registered but disabled."""
