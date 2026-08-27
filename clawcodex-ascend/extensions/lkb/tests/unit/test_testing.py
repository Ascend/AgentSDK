#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# Pytest loads this namespace package in the complete ordered migration state.
# pylint: disable=relative-beyond-top-level
"""Tests for deterministic helpers shared by the LKB test suite."""

from __future__ import annotations

from .._support import DeterministicClock


def test_deterministic_clock_rolls_over_the_minute_boundary() -> None:
    clock = DeterministicClock()
    clock.advance(59)

    assert clock.now() == "2026-01-01T00:00:59.000Z"
    assert clock.now() == "2026-01-01T00:01:00.000Z"


def test_deterministic_clock_advance_uses_real_timedelta_arithmetic() -> None:
    clock = DeterministicClock()
    clock.advance(3_661)

    assert clock.peek() == "2026-01-01T01:01:01.000Z"
