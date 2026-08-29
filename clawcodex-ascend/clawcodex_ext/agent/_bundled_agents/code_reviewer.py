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

"""Bundled ``code-reviewer`` agent.

Demonstrates the @register + policy composition pattern: pick an
identity template, attach one or more action norms, and stitch the
final system prompt together with :func:`build_agent_prompt`.
"""

from __future__ import annotations

from clawcodex_ext.agent.policy import (
    IDENTITY_CODE_REVIEWER,
    NORM_DIFF_FOCUSED,
    NORM_READ_ONLY,
    TOOL_SET_READ_ONLY,
    build_agent_prompt,
)
from clawcodex_ext.agent.registry import AgentRegistry

_SYSTEM_PROMPT = build_agent_prompt(
    identity=IDENTITY_CODE_REVIEWER,
    norms=[NORM_READ_ONLY, NORM_DIFF_FOCUSED],
    extra=(
        "When done, end your report with two sections:\n"
        "## Blocking issues\n"
        "Each item: file:line, impact, and a minimal fix.\n"
        "## Suggestions\n"
        "Each item: file:line, why it matters, optional patch."
    ),
)


@AgentRegistry.register(
    "code-reviewer",
    when_to_use=(
        "Code-review specialist for diffs and PRs. Use after a logical chunk "
        "of code is written to get an independent review before reporting "
        "completion. Read-only — never edits files itself."
    ),
    tools=TOOL_SET_READ_ONLY,
    permission_mode="default",
)
def _code_reviewer_prompt() -> str:
    return _SYSTEM_PROMPT
