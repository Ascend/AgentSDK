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

"""Sequential failover scheduling."""

from __future__ import annotations

from typing import Any

from clawcodex_ext.capabilities.multimodel_protocol import MultiModelResult  # pylint: disable=no-name-in-module
from clawcodex_ext.providers.base import MessageInput

from .base import MultiModelStrategyBase


class FallbackStrategy(MultiModelStrategyBase):
    """Try enabled slots in order and stop after the first successful call."""

    name = "fallback"

    async def execute(self, router: Any, messages: list[MessageInput], **kwargs: Any) -> list[MultiModelResult]:
        results: list[MultiModelResult] = []
        for slot in router.slots:
            if not slot.enabled:
                continue
            result = await router._call_slot(slot, messages, **kwargs)
            results.append(result)
            if result.error is None and not result.cancelled:
                break
        return results
