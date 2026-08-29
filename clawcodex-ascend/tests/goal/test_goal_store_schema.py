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

"""Regression tests for GoalStore schema upgrades."""

from __future__ import annotations

import sqlite3

from clawcodex_ext.goal.store import GoalStore


def test_goal_store_adds_all_columns_to_legacy_schema(tmp_path) -> None:
    db_path = tmp_path / "goals_1.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE thread_goals (thread_id TEXT PRIMARY KEY NOT NULL)")

    with GoalStore(db_path):
        pass

    with sqlite3.connect(db_path) as connection:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(thread_goals)")}

    assert {"completion_mode", "evaluation_count", "last_evaluation_reason"} <= columns
