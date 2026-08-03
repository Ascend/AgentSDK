"""Regression tests for cron_system runs fixes (issues #27, #31).

Moved from cron-b3 (test_runs.py) to keep PR #539 under 1k lines.
"""

# pylint: disable=no-name-in-module

from __future__ import annotations

import json

from clawcodex_ext.cron_system.runs import (
    MAX_CRON_RUNS,
    CreateCronRunParams,
    CronRun,
    create_queued_run,
    read_cron_runs,
    runs_file_path,
    write_cron_runs,
)


# ---- Regression: issue #27 — active runs must survive truncation ----


def test_write_cron_runs_preserves_active_runs_on_truncation(tmp_path) -> None:
    """Active runs must not be evicted when the total exceeds MAX_CRON_RUNS."""

    # 1 active run (oldest by queued_at) + MAX_CRON_RUNS terminal runs (newer)
    active = CronRun(
        id="active1",
        task_id="task1",
        prompt="active",
        status="running",
        queued_at=1000,
    )
    terminal_runs = [
        CronRun(
            id=f"term{i}",
            task_id=f"task-{i}",
            prompt=f"term-{i}",
            status="completed",
            queued_at=2000 + i,
        )
        for i in range(MAX_CRON_RUNS)
    ]
    all_runs = [active] + terminal_runs
    write_cron_runs(tmp_path, all_runs)

    persisted = read_cron_runs(tmp_path)
    persisted_ids = {r.id for r in persisted}
    assert "active1" in persisted_ids, "active run must survive truncation"


def test_truncation_does_not_break_dedup(tmp_path) -> None:
    """After truncation, create_queued_run must still deduplicate active runs."""

    # Fill with MAX_CRON_RUNS terminal runs + 1 active run for source "dedup-task"
    active = CronRun(
        id="activerun",
        task_id="dedup-task",
        prompt="dedup",
        status="running",
        queued_at=1000,
        trigger="scheduled-task",
        source_id="dedup-task",
    )
    terminal_runs = [
        CronRun(
            id=f"term{i}",
            task_id=f"other-{i}",
            prompt=f"other-{i}",
            status="completed",
            queued_at=2000 + i,
        )
        for i in range(MAX_CRON_RUNS)
    ]
    write_cron_runs(tmp_path, [active] + terminal_runs)

    # Attempting to queue another run for the same source must return None
    # (dedup guard finds the still-persisted active run).
    result = create_queued_run(
        tmp_path,
        CreateCronRunParams(
            task_id="dedup-task",
            prompt="dedup",
            source_id="dedup-task",
            trigger="scheduled-task",
            queued_at=9999,
        ),
    )
    assert result is None, "dedup must still find the active run after truncation"


# ---- Regression: issue #31 — queued_at=0 must not be skipped by or-chain ----


def test_cron_run_from_dict_preserves_queued_at_zero(tmp_path) -> None:
    """queued_at=0 (epoch) must be preserved, not skipped to a fallback key."""
    runs_file_path(tmp_path).parent.mkdir(parents=True)
    runs_file_path(tmp_path).write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "id": "r0",
                        "task_id": "t0",
                        "prompt": "zero",
                        "status": "completed",
                        "queued_at": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runs = read_cron_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].queued_at == 0
