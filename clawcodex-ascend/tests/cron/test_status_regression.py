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

"""Regression tests for cron_system status fixes (issue #15).

Moved from cron-b5 (test_status.py) to keep PR #541 under 1k lines.
"""

# pylint: disable=no-name-in-module

from __future__ import annotations

from clawcodex_ext.cron_system.status import build_autonomy_status, build_schedule_list
from clawcodex_ext.cron_system.tasks import add_cron_task


# ---- Regression: issue #15 — Orphaned column semantics ----


def test_orphaned_column_marks_agentless_tasks(tmp_path) -> None:
    """Tasks with agent_id=None (global) should show ✓ in Orphaned column."""
    task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="global", durable=True, created_at=1_000)
    assert task.agent_id is None
    output = build_autonomy_status(tmp_path)
    assert "✓" in output


def test_orphaned_column_marks_owned_tasks_as_not_orphaned(tmp_path) -> None:
    """Tasks with an agent_id should show — (not orphaned)."""
    task = add_cron_task(
        tmp_path,
        cron="*/5 * * * *",
        prompt="owned",
        durable=True,
        created_at=1_000,
        agent_id="agent-1",
    )
    assert task.agent_id is not None
    output = build_schedule_list(tmp_path)
    assert "—" in output
