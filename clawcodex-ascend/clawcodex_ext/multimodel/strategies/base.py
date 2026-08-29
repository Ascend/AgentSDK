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

"""Common strategy base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from clawcodex_ext.capabilities.multimodel_protocol import MultiModelResult  # pylint: disable=no-name-in-module
from clawcodex_ext.providers.base import MessageInput


class MultiModelStrategyBase(ABC):
    """Small concrete-friendly base for the scheduling protocol."""

    name: str

    @abstractmethod
    async def execute(self, router: Any, messages: list[MessageInput], **kwargs: Any) -> list[MultiModelResult]:
        """Execute one router turn and retain every attempted result."""
