"""Regression tests for cron_system parser fixes (issues #1, #2).

Moved from cron-b1 (test_parser.py) to keep PR #537 under 1k lines.
"""

# pylint: disable=no-name-in-module

from __future__ import annotations

import time
from datetime import datetime

from clawcodex_ext.cron_system.parser import compute_next_cron_run, parse_cron_expression


# ---- Regression: issue #2 — Sunday=7 normalization ----


def test_dow_sunday_as_7_single_value() -> None:
    """``7`` alone means Sunday (same as ``0``)."""
    fields = parse_cron_expression("0 0 * * 7")
    assert fields is not None
    assert 0 in fields.days_of_week
    assert 7 not in fields.days_of_week


def test_dow_range_0_to_7_covers_full_week() -> None:
    """``0-7`` should expand to all 7 days (Mon-Sun), not just Sunday."""
    fields = parse_cron_expression("0 0 * * 0-7")
    assert fields is not None
    assert len(fields.days_of_week) == 7


def test_dow_range_5_to_7_covers_fri_sat_sun() -> None:
    """``5-7`` should expand to {5, 6, 0} (Fri, Sat, Sun)."""
    fields = parse_cron_expression("0 0 * * 5-7")
    assert fields is not None
    assert fields.days_of_week == frozenset({0, 5, 6})


def test_dow_step_1_to_7_step_2() -> None:
    """``1-7/2`` should expand to {1, 3, 5, 0} (Mon, Wed, Fri, Sun)."""
    fields = parse_cron_expression("0 0 * * 1-7/2")
    assert fields is not None
    assert fields.days_of_week == frozenset({0, 1, 3, 5})


# ---- Regression: issue #1 — field-skipping performance ----


def test_compute_next_cron_run_impossible_date_returns_none() -> None:
    """``0 0 31 2 *`` (Feb 31) is impossible; should return None quickly."""
    fields = parse_cron_expression("0 0 31 2 *")
    assert fields is not None
    result = compute_next_cron_run(fields, datetime(2026, 1, 1, 0, 0, 0))
    assert result is None


def test_compute_next_cron_run_leap_day() -> None:
    """``0 0 29 2 *`` (Feb 29) should find the next leap year quickly."""
    fields = parse_cron_expression("0 0 29 2 *")
    assert fields is not None
    result = compute_next_cron_run(fields, datetime(2026, 1, 1, 0, 0, 0))
    assert result == datetime(2028, 2, 29, 0, 0, 0)


def test_compute_next_cron_run_sparse_month_day() -> None:
    """Sparse expression resolves in reasonable iterations."""
    fields = parse_cron_expression("30 9 1 1 *")
    assert fields is not None
    start = time.monotonic()
    result = compute_next_cron_run(fields, datetime(2026, 7, 15, 10, 0, 0))
    elapsed = time.monotonic() - start
    assert result == datetime(2027, 1, 1, 9, 30, 0)
    assert elapsed < 1.0
