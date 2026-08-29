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

"""Dashboard and recorder structural protocol contracts."""

from __future__ import annotations

from extensions.visualizer.protocols.dashboard import (
    DashboardEntry,
    filter_entries,
    normalize_source_name,
)
from extensions.visualizer.protocols.recorder import AsciicastHeader


def test_dashboard_protocol_normalizes_filters_and_progress() -> None:
    entries = [
        DashboardEntry(id="goal:1", source="goal", title="one", progress_pct=2.0),
        DashboardEntry(id="task:1", source="task", title="two", status="completed"),
    ]

    assert entries[0].progress_pct == 1.0
    assert normalize_source_name(" Goal--Queue ") == "goal_queue"
    assert filter_entries(entries, source="GOAL") == [entries[0]]
    assert filter_entries(entries, status="completed") == [entries[1]]


def test_dashboard_invalid_progress_remains_unknown() -> None:
    invalid = DashboardEntry(
        id="goal:invalid",
        source="goal",
        title="invalid",
        progress_pct="not-a-number",  # type: ignore[arg-type]
    )
    non_finite = DashboardEntry(
        id="goal:nan",
        source="goal",
        title="nan",
        progress_pct=float("nan"),
    )
    zero = DashboardEntry(
        id="goal:zero",
        source="goal",
        title="zero",
        progress_pct=0,
    )

    assert invalid.progress_pct is None
    assert non_finite.progress_pct is None
    assert zero.progress_pct == 0.0


def test_recorder_header_omits_absent_optional_values() -> None:
    header = AsciicastHeader(width=120, height=40, title="Visualizer")

    assert header.to_dict() == {
        "version": 2,
        "width": 120,
        "height": 40,
        "title": "Visualizer",
    }
