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

"""The no-selection multi-model aggregator."""

from __future__ import annotations

from typing import Any

from clawcodex_ext.capabilities.multimodel_protocol import AggregatedOutput, MultiModelResult  # pylint: disable=no-name-in-module

from .base import fallback_output


class PassThroughAggregator:
    """Keep every result and select the first successful one for compatibility."""

    async def aggregate(self, results: list[MultiModelResult], context: dict[str, Any]) -> AggregatedOutput:
        del context
        return fallback_output(results)
