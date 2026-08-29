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

"""Bundled ``test-runner`` agent."""

from __future__ import annotations

from clawcodex_ext.agent.policy import (
    IDENTITY_TEST_RUNNER,
    NORM_GIT_OPERATOR,
    TOOL_SET_TESTING,
    build_agent_prompt,
)
from clawcodex_ext.agent.registry import AgentRegistry

_SYSTEM_PROMPT = build_agent_prompt(
    identity=IDENTITY_TEST_RUNNER,
    norms=[NORM_GIT_OPERATOR],
    extra=(
        "Run the smallest test subset that exercises the change first; "
        "only run the full suite if the focused subset passes. For each "
        "failure report: file:line, expected vs actual, and your best "
        "guess at root cause. Do not modify any files — surface the "
        "fix as a recommendation to the caller."
    ),
)


@AgentRegistry.register(
    "test-runner",
    when_to_use=(
        "Test-runner specialist. Use after writing or modifying code to "
        "execute the relevant test suite and report results. Read-only — "
        "never edits source code itself."
    ),
    tools=TOOL_SET_TESTING,
    permission_mode="default",
)
def _test_runner_prompt() -> str:
    return _SYSTEM_PROMPT
