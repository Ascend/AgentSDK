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

from clawcodex_ext.cron_system.runs import create_queued_run_for_task, finalize_cron_run
from clawcodex_ext.cron_system.status import build_autonomy_runs, build_autonomy_status
from clawcodex_ext.cron_system.tasks import add_cron_task


def test_autonomy_status_shows_jobs_and_runs(tmp_path) -> None:
    task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="ping", durable=True, created_at=1_000)
    run = create_queued_run_for_task(tmp_path, task, queued_at=2_000)

    output = build_autonomy_status(tmp_path)

    assert "Autonomy status" in output
    assert task.id in output
    assert run is not None
    assert run.id in output
    assert "queued" in output


def test_autonomy_runs_empty_message(tmp_path) -> None:
    assert build_autonomy_runs(tmp_path) == "No scheduled-task runs."


def test_autonomy_runs_deep_shows_source_path_and_error(tmp_path) -> None:
    task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="ping", durable=True, created_at=1_000)
    run = create_queued_run_for_task(tmp_path, task, queued_at=2_000, current_dir=tmp_path / "work")
    assert run is not None
    finalize_cron_run(tmp_path, run.id, "failed", timestamp=3_000, error="boom")

    output = build_autonomy_runs(tmp_path, deep=True)

    assert f"Run {run.id}" in output
    assert "source:" in output
    assert task.id in output
    assert "root_dir:" in output
    assert "current_dir:" in output
    assert "boom" in output
