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

"""Parallel scheduling for aggregator-backed voting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clawcodex_ext.capabilities.multimodel_protocol import AggregatorProtocol, MultiModelResult  # pylint: disable=no-name-in-module
from clawcodex_ext.providers.base import MessageInput

from .parallel import ParallelStrategy


@dataclass
class VotingStrategy(ParallelStrategy):
    """Run candidates in parallel; the router's aggregator picks a response.

    ``aggregator`` is optional to support a self-contained strategy while
    retaining the router-level aggregator injection documented.
    """

    aggregator: AggregatorProtocol | None = None
    min_votes: int = 2
    name = "voting"

    async def execute(self, router: Any, messages: list[MessageInput], **kwargs: Any) -> list[MultiModelResult]:
        return await super().execute(router, messages, **kwargs)
