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

from __future__ import annotations

import re

CLAUDEAI_SERVER_PREFIX = "claude.ai "

# Per chapter §"Apply This" + TS regex, MCP tool names must match
# ``^[a-zA-Z0-9_-]{1,64}$``. Enforce the 64-char cap (Phase 10 WI-10.2).
MAX_MCP_NAME_LENGTH = 64


def normalize_name_for_mcp(name: str) -> str:
    """Lower-case-preserving normalization to the MCP name grammar.

    Steps:
      1. Replace any non-``[a-zA-Z0-9_-]`` character with ``_``.
      2. For Claude.ai-sourced server names (``"claude.ai "`` prefix),
         collapse runs of underscores and strip leading/trailing
         underscores so they don't pollute the ``mcp__server__tool``
         delimiter.
      3. Truncate to 64 chars to satisfy the API regex's length bound.
    """
    normalized = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    if name.startswith(CLAUDEAI_SERVER_PREFIX):
        normalized = re.sub(r"_+", "_", normalized)
        normalized = normalized.strip("_")
    if len(normalized) > MAX_MCP_NAME_LENGTH:
        normalized = normalized[:MAX_MCP_NAME_LENGTH]
    return normalized
