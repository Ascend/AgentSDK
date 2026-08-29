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

"""Bounded parallel provider invocation."""

from __future__ import annotations

import asyncio
from typing import Any

from clawcodex_ext.capabilities.multimodel_protocol import MultiModelResult  # pylint: disable=no-name-in-module
from clawcodex_ext.providers.base import MessageInput

from .base import MultiModelStrategyBase


class ParallelStrategy(MultiModelStrategyBase):
    """Send the same request to every enabled slot concurrently."""

    name = "parallel"

    async def execute(self, router: Any, messages: list[MessageInput], **kwargs: Any) -> list[MultiModelResult]:
        slots = [slot for slot in router.slots if slot.enabled]
        if not slots:
            return []
        semaphore = asyncio.Semaphore(router.config.max_concurrent)

        async def call(slot: Any) -> MultiModelResult:
            async with semaphore:
                return await router._call_slot(slot, messages, **kwargs)

        return list(await asyncio.gather(*(call(slot) for slot in slots)))
