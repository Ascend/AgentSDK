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

"""Cron expression parsing and next-run calculation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .models import CronFields

_FIELD_RANGES = (
    (0, 59),
    (0, 23),
    (1, 31),
    (1, 12),
    (0, 6),
)

# DOW field accepts 0-7 (both 0 and 7 mean Sunday). The raw range is 0..7;
# after expansion, 7 is folded into 0 and the set is deduplicated.
_DOW_RAW_RANGE = (0, 7)


_NAMES = (
    {},
    {},
    {},
    {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    },
    {
        "sun": 0,
        "mon": 1,
        "tue": 2,
        "wed": 3,
        "thu": 4,
        "fri": 5,
        "sat": 6,
    },
)


def parse_cron_expression(expr: str) -> CronFields | None:
    parts = expr.split()
    if len(parts) != 5:
        return None

    parsed: list[frozenset[int]] = []
    for idx, part in enumerate(parts):
        if idx == 4:
            # DOW field: expand in the raw 0..7 domain, then fold 7→0.
            values = _parse_field(part, *_DOW_RAW_RANGE, names=_NAMES[idx], normalize_sunday=False)
            if values is None:
                return None
            values = {(v if v != 7 else 0) for v in values}  # pylint: disable=not-an-iterable
        else:
            values = _parse_field(part, *_FIELD_RANGES[idx], names=_NAMES[idx], normalize_sunday=False)
            if values is None:
                return None
        parsed.append(frozenset(values))

    return CronFields(
        minutes=parsed[0],
        hours=parsed[1],
        days_of_month=parsed[2],
        months=parsed[3],
        days_of_week=parsed[4],
    )


def compute_next_cron_run(fields: CronFields, from_time: datetime) -> datetime | None:
    """Find the next time matching the cron fields after ``from_time``.

    Uses a field-skipping algorithm (month → day → hour → minute) instead
    of brute-force per-minute enumeration, so even sparse expressions like
    ``0 0 29 2 *`` resolve in at most a few hundred iterations.
    """
    candidate = (from_time + timedelta(minutes=1)).replace(second=0, microsecond=0)
    limit = candidate + timedelta(days=366 * 5)
    # Cap iterations as a safety budget (normal expressions need < 1000).
    max_iterations = 100_000
    for _ in range(max_iterations):
        if candidate > limit:
            return None
        if candidate.month not in fields.months:
            candidate = _skip_to_next_month(candidate)
            continue
        cron_weekday = (candidate.weekday() + 1) % 7
        if not _day_matches(candidate.day, cron_weekday, fields):
            candidate = (candidate + timedelta(days=1)).replace(hour=0, minute=0)
            continue
        if candidate.hour not in fields.hours:
            candidate = (candidate + timedelta(hours=1)).replace(minute=0)
            continue
        if candidate.minute not in fields.minutes:
            candidate += timedelta(minutes=1)
            continue
        return candidate
    return None


def _skip_to_next_month(candidate: datetime) -> datetime:
    """Advance to the first day of the next month at 00:00."""
    if candidate.month == 12:
        return candidate.replace(year=candidate.year + 1, month=1, day=1, hour=0, minute=0)
    return candidate.replace(month=candidate.month + 1, day=1, hour=0, minute=0)


def _local_utc_offset_hours() -> int:
    """Return the local timezone offset from UTC in hours (e.g. 8 for UTC+8, -5 for UTC-5).

    Uses DST-aware local timezone from the system clock.

    Note: only whole-hour offsets are supported. Non-integer offsets such as
    UTC+5:30 (India) or UTC+3:30 (Iran) are truncated to the nearest lower
    whole hour (e.g. +5:30 → +5). This affects ``cron_to_human(utc=True)``
    display only, not the actual cron scheduling which uses local time directly.
    """
    now = datetime.now(timezone.utc)
    local_offset = now.astimezone().utcoffset()
    assert local_offset is not None
    return int(local_offset.total_seconds() // 3600)


def cron_to_human(cron: str, utc: bool = False) -> str:
    """Convert a cron expression to a human-readable schedule string.

    When ``utc=True``, hours are offset from UTC to the local timezone. The
    displayed time is the clock-hour only — if the offset crosses a day
    boundary (e.g. UTC 23:00 + UTC+8 = 07:00 next day), no "next day" suffix
    is appended. This matches the original CCB behaviour and keeps the output
    concise; callers needing day-rollover semantics should compute separately.
    """
    fields = parse_cron_expression(cron)
    if fields is None:
        return cron

    suffix = " UTC" if utc else ""
    minutes, hours, dom, months, dow = cron.split()

    # When utc=True, offset the cron hour from UTC to local timezone.
    local_offset = _local_utc_offset_hours() if utc else 0

    def _offset_hour(h: int) -> int:
        return (h + local_offset) % 24

    if cron == "* * * * *":
        return f"Every minute{suffix}"
    if hours == "*" and dom == "*" and months == "*" and dow == "*":
        if minutes.startswith("*/"):
            return f"Every {minutes[2:]} minutes{suffix}"
        if minutes.isdigit():
            return f"Hourly at minute {minutes}{suffix}"
    if dom == "*" and months == "*" and dow == "*" and minutes.isdigit():
        if hours.startswith("*/"):
            return f"Every {hours[2:]} hours at minute {minutes}{suffix}"
        if hours.isdigit():
            h = _offset_hour(int(hours))
            return f"Daily at {h:02d}:{int(minutes):02d}{suffix}"
    if months == "*" and dow == "*" and minutes.isdigit() and hours.isdigit() and dom.isdigit():
        h = _offset_hour(int(hours))
        return f"Monthly on day {int(dom)} at {h:02d}:{int(minutes):02d}{suffix}"
    if months == "*" and dom == "*" and minutes.isdigit() and hours.isdigit() and dow.isdigit():
        h = _offset_hour(int(hours))
        return f"Weekly on day {int(dow)} at {h:02d}:{int(minutes):02d}{suffix}"
    return f"Cron schedule {cron}{suffix}"


def _day_matches(day_of_month: int, day_of_week: int, fields: CronFields) -> bool:
    dom_restricted = len(fields.days_of_month) != 31
    dow_restricted = len(fields.days_of_week) != 7
    dom_matches = day_of_month in fields.days_of_month
    dow_matches = day_of_week in fields.days_of_week
    if dom_restricted and dow_restricted:
        return dom_matches or dow_matches
    return dom_matches and dow_matches


def datetime_to_ms(value: datetime) -> int:
    if value.tzinfo is None:
        return int(value.timestamp() * 1000)
    return int(value.astimezone(timezone.utc).timestamp() * 1000)


def ms_to_datetime(value: int, tzinfo=timezone.utc) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=tzinfo)


def _parse_field(
    field: str,
    minimum: int,
    maximum: int,
    *,
    names: dict[str, int],
    normalize_sunday: bool = False,
) -> set[int] | None:
    if not field:
        return None
    values: set[int] = set()
    for segment in field.split(","):
        segment_values = _parse_segment(
            segment.strip().lower(),
            minimum,
            maximum,
            names=names,
            normalize_sunday=normalize_sunday,
        )
        if segment_values is None:
            return None
        values.update(segment_values)
    return values or None


def _parse_segment(
    segment: str,
    minimum: int,
    maximum: int,
    *,
    names: dict[str, int],
    normalize_sunday: bool,
) -> set[int] | None:
    if not segment:
        return None

    base, step = segment, 1
    if "/" in segment:
        base, step_text = segment.split("/", 1)
        if not step_text.isdigit():
            return None
        step = int(step_text)
        if step <= 0:
            return None

    if base == "*":
        start, end = minimum, maximum
    elif "-" in base:
        start_text, end_text = base.split("-", 1)
        start = _parse_value(start_text, names=names)
        end = _parse_value(end_text, names=names)
        if start is None or end is None or start > end:
            return None
    else:
        value = _parse_value(base, names=names)
        if value is None:
            return None
        start = end = value

    if start < minimum or end > maximum:
        return None
    return set(range(start, end + 1, step))


def _parse_value(value: str, *, names: dict[str, int]) -> int | None:
    if value in names:
        return names[value]
    if not value.isdigit():
        return None
    return int(value)
