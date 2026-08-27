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

# Pytest injects the imported ``drill`` fixture through the same-named test parameter.
# pylint: disable=redefined-outer-name
"""End-to-end entry point for the Agent Loop release drill."""

from tests.smoke._agent_loop_drill_phases_a import (
    _assert_phase1,
    _assert_phase2,
    _assert_phase3,
    _assert_phase4,
    _phase1,
    _phase2,
    _phase3,
    _phase4,
)
from tests.smoke._agent_loop_drill_phases_b import (
    _assert_phase5,
    _assert_phase6,
    _phase5,
    _phase6,
)
from tests.smoke._agent_loop_drill_support import drill  # noqa: F401


def test_agent_loop_drill(drill, capsys) -> None:  # noqa: F811
    """Run the complete six-phase drill once, asserting every checkpoint."""
    d = drill
    phases = (
        (_phase1, _assert_phase1),
        (_phase2, _assert_phase2),
        (_phase3, _assert_phase3),
        (_phase4, _assert_phase4),
        (_phase5, _assert_phase5),
        (_phase6, _assert_phase6),
    )
    for execute, verify in phases:
        execute(d)
        verify(d, capsys)
