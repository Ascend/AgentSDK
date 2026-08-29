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

"""Capability gate — fail-closed check before a gateway calls a channel.

Every gateway→channel call goes through :meth:`CapabilityGate.require`
so an undeclared capability (e.g. sending media over a text-only
channel) is rejected before any platform API is touched.
"""

from __future__ import annotations

# Sibling clawcodex_ext packages (channels) are migrated in separate branches;
# suppress E0611 until the full series lands.
# pylint: disable=no-name-in-module

from clawcodex_ext.services.channels.capabilities import (
    ChannelAdapter,
    ChannelCapability,
)


class CapabilityGate:
    def __init__(self, registry) -> None:  # registry: ChannelAdapterRegistry
        self._registry = registry

    def require(
        self,
        channel: str | ChannelAdapter,
        capability: ChannelCapability,
    ) -> ChannelAdapter:
        """Resolve ``channel`` and fail closed if ``capability`` is undeclared."""
        return self._registry.require_capability(channel, capability)

    def require_outbound(self, channel: str | ChannelAdapter) -> ChannelAdapter:
        return self.require(channel, ChannelCapability.OUTBOUND_TEXT)

    def require_context_reply(self, channel: str | ChannelAdapter) -> ChannelAdapter:
        adapter = self.require(channel, ChannelCapability.CONTEXT_REPLY)
        return adapter

    def require_media(self, channel: str | ChannelAdapter, capability: ChannelCapability) -> ChannelAdapter:
        if capability not in (
            ChannelCapability.MEDIA_IMAGE,
            ChannelCapability.MEDIA_FILE,
            ChannelCapability.MEDIA_VIDEO,
        ):
            raise ValueError(f"{capability!r} is not a media capability")
        return self.require(channel, capability)


__all__ = ["CapabilityGate"]
