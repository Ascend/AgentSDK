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

"""Bundled ``docs-writer`` agent."""

from __future__ import annotations

from clawcodex_ext.agent.policy import (
    IDENTITY_DOCS_WRITER,
    NORM_CODE_AUTHOR,
    TOOL_SET_AUTHOR,
    build_agent_prompt,
)
from clawcodex_ext.agent.registry import AgentRegistry

_SYSTEM_PROMPT = build_agent_prompt(
    identity=IDENTITY_DOCS_WRITER,
    norms=[NORM_CODE_AUTHOR],
    extra=(
        "Only create or edit documentation files when the user has "
        "explicitly asked for documentation. Default to editing existing "
        "files. Do not add sections that the user did not ask for."
    ),
)


@AgentRegistry.register(
    "docs-writer",
    when_to_use=(
        "Documentation specialist. Use when the user explicitly asks for documentation to be written or updated."
    ),
    tools=TOOL_SET_AUTHOR,
    permission_mode="acceptEdits",
)
def _docs_writer_prompt() -> str:
    return _SYSTEM_PROMPT
