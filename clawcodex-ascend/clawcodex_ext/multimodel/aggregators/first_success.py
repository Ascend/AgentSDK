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

"""Completion-order aggregation for parallel model calls."""

from __future__ import annotations

from typing import Any

from clawcodex_ext.capabilities.multimodel_protocol import AggregatedOutput, MultiModelResult  # pylint: disable=no-name-in-module

from .base import fallback_output, require_results, valid_results


class FirstSuccessAggregator:
    """Choose the successful slot that actually completed first.

    Unlike ``passthrough``, this ignores configured slot order.  The router
    records a monotonic completion timestamp for every slot, including errors,
    so queued and concurrently-completed requests are ordered correctly.
    """

    async def aggregate(self, results: list[MultiModelResult], context: dict[str, Any]) -> AggregatedOutput:
        del context
        require_results(results)
        valid = valid_results(results)
        if not valid:
            return fallback_output(results)
        chosen = min(
            enumerate(valid),
            key=lambda indexed: (indexed[1].completed_at or float("inf"), indexed[0]),
        )[1]
        return AggregatedOutput(
            chosen=chosen.response,
            runners_up=[result for result in results if result is not chosen],
            provenance=list(results),
            vote_summary={"selection": "first_success", "winning_slot": chosen.slot_name},
        )
