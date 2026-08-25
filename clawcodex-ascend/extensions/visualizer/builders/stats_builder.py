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

"""Aggregate operation statistics from timeline bars."""

from __future__ import annotations

from extensions.visualizer.models.viz_models import OperationStats, TimelineBar


class StatsBuilder:
    """Build OperationStats from timeline bars."""

    def build(
        self,
        bars: list[TimelineBar],
        base: OperationStats | None = None,
    ) -> OperationStats:
        """Build stats, preserving base cost and context fields that bars cannot derive."""
        if not bars:
            if base is not None:
                return OperationStats(
                    total_ops=0,
                    by_type={},
                    avg_duration_ms=0.0,
                    max_concurrent=0,
                    total_duration_ms=0,
                    wall_clock_duration_ms=0,
                    context_tokens=base.context_tokens,
                    cost_usd=base.cost_usd,
                )
            return OperationStats()

        total_ops = len(bars)
        by_type: dict[str, int] = {}
        durations: list[int] = []
        max_concurrent = 0

        events: list[tuple[float, int]] = []  # (time, delta)
        wall_start: float | None = None
        wall_end: float | None = None
        for timeline_bar in bars:
            bar_type = timeline_bar.type.value
            by_type[bar_type] = by_type.get(bar_type, 0) + 1
            durations.append(timeline_bar.duration_ms)
            events.append((timeline_bar.start_time, +1))
            events.append((timeline_bar.end_time, -1))
            if wall_start is None or timeline_bar.start_time < wall_start:
                wall_start = timeline_bar.start_time
            if wall_end is None or timeline_bar.end_time > wall_end:
                wall_end = timeline_bar.end_time

        events.sort(key=lambda x: (x[0], -x[1]))
        current = 0
        for _, delta in events:
            current += delta
            max_concurrent = max(max_concurrent, current)

        total_duration = sum(durations)
        avg_duration = total_duration / len(durations) if durations else 0.0
        wall_clock_duration = (
            int((wall_end - wall_start) * 1000)
            if wall_start is not None and wall_end is not None and wall_end >= wall_start
            else 0
        )

        context_tokens = base.context_tokens if base is not None else 0
        cost_usd = base.cost_usd if base is not None else 0.0

        return OperationStats(
            total_ops=total_ops,
            by_type=by_type,
            avg_duration_ms=avg_duration,
            max_concurrent=max_concurrent,
            total_duration_ms=total_duration,
            wall_clock_duration_ms=wall_clock_duration,
            context_tokens=context_tokens,
            cost_usd=cost_usd,
        )
