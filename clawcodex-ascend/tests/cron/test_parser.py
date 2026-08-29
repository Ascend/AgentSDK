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

from __future__ import annotations

from datetime import datetime, timezone

from clawcodex_ext.cron_system.parser import (
    compute_next_cron_run,
    cron_to_human,
    parse_cron_expression,
)


def _local_offset() -> int:
    """Return the local UTC offset in hours."""
    now = datetime.now(timezone.utc)
    off = now.astimezone().utcoffset()
    assert off is not None
    return int(off.total_seconds() // 3600)


def test_parse_cron_expression_supports_common_forms() -> None:
    assert parse_cron_expression("* * * * *") is not None
    assert parse_cron_expression("*/15 * * * *") is not None
    assert parse_cron_expression("0-30/10 1,2,3 * jan mon-fri") is not None


def test_parse_cron_expression_rejects_invalid_forms() -> None:
    assert parse_cron_expression("* * *") is None
    assert parse_cron_expression("*/0 * * * *") is None
    assert parse_cron_expression("61 * * * *") is None
    assert parse_cron_expression("20-10 * * * *") is None


def test_compute_next_cron_run_is_strictly_future() -> None:
    fields = parse_cron_expression("*/15 * * * *")
    assert fields is not None
    result = compute_next_cron_run(fields, datetime(2026, 1, 1, 12, 0, 0))
    assert result == datetime(2026, 1, 1, 12, 15, 0)


def test_day_of_month_and_day_of_week_use_or_semantics() -> None:
    fields = parse_cron_expression("0 9 15 * 1")
    assert fields is not None
    result = compute_next_cron_run(fields, datetime(2026, 6, 14, 9, 0, 0))
    assert result == datetime(2026, 6, 15, 9, 0, 0)


def test_cron_to_human_common_strings() -> None:
    assert cron_to_human("* * * * *") == "Every minute"
    assert cron_to_human("*/10 * * * *") == "Every 10 minutes"
    assert cron_to_human("0 9 * * *") == "Daily at 09:00"


def test_cron_to_human_utc_offset() -> None:
    """With utc=True, hours are offset from UTC to local timezone."""
    offset = _local_offset()
    expected_hour = (9 + offset) % 24
    assert cron_to_human("0 9 * * *", utc=True) == f"Daily at {expected_hour:02d}:00 UTC"

    expected_hour_15 = (15 + offset) % 24
    assert cron_to_human("0 15 * * 1") == "Weekly on day 1 at 15:00"
    assert cron_to_human("0 15 * * 1", utc=True) == f"Weekly on day 1 at {expected_hour_15:02d}:00 UTC"

    expected_hour_22 = (22 + offset) % 24
    assert cron_to_human("0 22 15 * *") == "Monthly on day 15 at 22:00"
    assert cron_to_human("0 22 15 * *", utc=True) == f"Monthly on day 15 at {expected_hour_22:02d}:00 UTC"


def test_cron_to_human_utc_interval_patterns() -> None:
    """Interval-based patterns get the suffix but no hour offset (no specific hour)."""
    assert cron_to_human("* * * * *", utc=True) == "Every minute UTC"
    assert cron_to_human("*/5 * * * *", utc=True) == "Every 5 minutes UTC"
    assert cron_to_human("0 * * * *", utc=True) == "Hourly at minute 0 UTC"
    assert cron_to_human("0 */2 * * *", utc=True) == "Every 2 hours at minute 0 UTC"


def test_cron_to_human_utc_wraps_midnight() -> None:
    """Hour 0 UTC wraps to local (e.g. UTC+8 → hour 8)."""
    offset = _local_offset()
    expected_hour = (0 + offset) % 24
    assert cron_to_human("0 0 * * *", utc=True) == f"Daily at {expected_hour:02d}:00 UTC"
