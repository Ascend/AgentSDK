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

# pylint: disable=no-name-in-module
"""Cron CLI subcommands — ``autonomy`` and ``schedule``."""

from __future__ import annotations

import sys
from pathlib import Path

from clawcodex_ext.cli.subcommand_registry import register


@register("autonomy")
def run_autonomy_command(args: list[str]) -> int:
    """Handle ``clawcodex autonomy [status|runs] [--deep]``."""
    from clawcodex_ext.cron_system.status import (
        build_autonomy_runs,
        build_autonomy_status,
    )

    deep = "--deep" in args
    filtered_args = [arg for arg in args if arg != "--deep"]
    command = filtered_args[0] if filtered_args else "status"

    if command == "status":
        print(build_autonomy_status(Path.cwd(), deep=deep))
        return 0
    if command == "runs":
        print(build_autonomy_runs(Path.cwd(), deep=deep))
        return 0

    print("usage: clawcodex autonomy [status|runs] [--deep]", file=sys.stderr)
    return 2


@register("schedule")
def run_schedule_command(args: list[str]) -> int:
    """Handle ``clawcodex schedule [list|get ID|run ID]``."""
    from clawcodex_ext.cron_system.schedule import (
        format_cron_task_detail,
        format_manual_fire_result,
        get_cron_task_detail,
        manual_fire_cron_task,
    )
    from clawcodex_ext.cron_system.status import build_schedule_list

    command = args[0] if args else "list"

    if command == "list":
        print(build_schedule_list(Path.cwd()))
        return 0

    if command == "get" and len(args) >= 2:
        detail = get_cron_task_detail(Path.cwd(), args[1])
        if detail is None:
            print(f"No scheduled job with id '{args[1]}'", file=sys.stderr)
            return 1
        print(format_cron_task_detail(detail))
        return 0

    if command == "run" and len(args) >= 2:
        cwd = Path.cwd()
        run = manual_fire_cron_task(cwd, args[1], current_dir=cwd)
        if run is None and get_cron_task_detail(cwd, args[1]) is None:
            print(f"No scheduled job with id '{args[1]}'", file=sys.stderr)
            return 1
        print(format_manual_fire_result(args[1], run))
        return 0

    print("usage: clawcodex schedule [list|get ID|run ID]", file=sys.stderr)
    return 2
