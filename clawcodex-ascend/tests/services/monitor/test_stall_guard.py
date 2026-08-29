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

"""Tests for stall-watchdog exemption predicate."""
# pylint: disable=no-name-in-module

from __future__ import annotations

from clawcodex_ext.services.monitor.stall_guard import StallWatchdogExemptor
from src.tasks.local_shell import LocalShellTaskState


class TestStallWatchdogExemptor:
    def test_monitor_kind_skipped(self):
        state = LocalShellTaskState(
            id="b12345678",
            type="local_bash",
            status="running",
            description="x",
            start_time=0.0,
            output_file="/tmp/x",
            kind="monitor",
        )
        assert StallWatchdogExemptor.should_skip_stall_check(state) is True

    def test_shell_kind_not_skipped(self):
        state = LocalShellTaskState(
            id="b12345678",
            type="local_bash",
            status="running",
            description="x",
            start_time=0.0,
            output_file="/tmp/x",
            kind="shell",
        )
        assert StallWatchdogExemptor.should_skip_stall_check(state) is False

    def test_missing_kind_defaults_to_shell(self):
        # Simulate a legacy object without a ``kind`` field.
        class Legacy:
            pass

        assert StallWatchdogExemptor.should_skip_stall_check(Legacy()) is False

    def test_arbitrary_object_no_kind(self):
        assert StallWatchdogExemptor.should_skip_stall_check(None) is False
        assert StallWatchdogExemptor.should_skip_stall_check({"kind": "monitor"}) is True
