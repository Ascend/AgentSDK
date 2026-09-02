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

"""XML tag constants for chapter-10 task notifications.

Mirrors ``typescript/src/constants/xml.ts``. Single source of truth for
the tag names that appear in ``<task-notification>`` envelopes the model
sees in its conversation flow. Keep names byte-identical to the TS
source so prompt-cache stability holds across TS↔Python interop and
snapshot tests pin the surface.
"""

from __future__ import annotations

from typing import Final

COMMON_HELP_ARGS: tuple[str, ...] = ("help", "-h", "--help")
COMMON_INFO_ARGS: tuple[str, ...] = (
    "list",
    "show",
    "display",
    "current",
    "view",
    "get",
    "check",
    "describe",
    "print",
    "version",
    "about",
    "status",
    "?",
)

TASK_NOTIFICATION_TAG: Final[str] = "task-notification"
TASK_ID_TAG: Final[str] = "task-id"
TOOL_USE_ID_TAG: Final[str] = "tool-use-id"
OUTPUT_FILE_TAG: Final[str] = "output-file"
STATUS_TAG: Final[str] = "status"
SUMMARY_TAG: Final[str] = "summary"
RESULT_TAG: Final[str] = "result"
USAGE_TAG: Final[str] = "usage"
TOTAL_TOKENS_TAG: Final[str] = "total_tokens"
TOOL_USES_TAG: Final[str] = "tool_uses"
DURATION_MS_TAG: Final[str] = "duration_ms"
WORKTREE_TAG: Final[str] = "worktree"
WORKTREE_PATH_TAG: Final[str] = "worktree-path"
WORKTREE_BRANCH_TAG: Final[str] = "worktree-branch"


__all__ = [
    "COMMON_HELP_ARGS",
    "COMMON_INFO_ARGS",
    "TASK_NOTIFICATION_TAG",
    "TASK_ID_TAG",
    "TOOL_USE_ID_TAG",
    "OUTPUT_FILE_TAG",
    "STATUS_TAG",
    "SUMMARY_TAG",
    "RESULT_TAG",
    "USAGE_TAG",
    "TOTAL_TOKENS_TAG",
    "TOOL_USES_TAG",
    "DURATION_MS_TAG",
    "WORKTREE_TAG",
    "WORKTREE_PATH_TAG",
    "WORKTREE_BRANCH_TAG",
]
