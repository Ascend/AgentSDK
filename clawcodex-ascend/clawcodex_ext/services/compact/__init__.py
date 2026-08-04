#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""
Compression pipeline for context management.

Implements the 5-layer compression pipeline matching TypeScript
``typescript/src/services/compact/``:

1. tool_result_budget — Persist large tool results to disk
2. snip_compact       — Trim old tool results (preserve structure)
3. microcompact       — Compress intermediate tool calls
4. context_collapse   — Read-time projection via collapse store
5. autocompact        — Full LLM summarization (last resort)

The pipeline runs cheap → expensive; if earlier layers free enough tokens,
later layers are no-ops.
"""

from __future__ import annotations

from . import reactive_compact
from .autocompact import (
    AutoCompactTracking,
    auto_compact_if_needed,
    calculate_token_warning_state,
    get_auto_compact_threshold,
    get_effective_context_window_size,
    is_auto_compact_enabled,
    should_auto_compact,
)
from .compact import CompactionResult, compact_conversation, truncate_head_for_ptl_retry
from .compact_warning import (
    clear_compact_warning_suppression,
    is_compact_warning_suppressed,
    suppress_compact_warning,
)
from .context_collapse import CollapseCommit, ContextCollapseStore
from .gating import (
    DEFAULT_COMPRESSION_GATE_SKIP_RATIO,
    ENV_COMPRESSION_GATE_SKIP_RATIO,
    resolve_skip_ratio_from_env,
    should_run_compression_pipeline,
)
from .grouping import ApiRound, group_messages_by_api_round
from .pipeline import CompressionPipeline, CompressionResult, run_compression_pipeline
from .post_compact_attachments import (
    POST_COMPACT_MAX_FILES_TO_RESTORE,
    POST_COMPACT_MAX_TOKENS_PER_FILE,
    POST_COMPACT_TOKEN_BUDGET,
    create_plan_attachment_if_needed,
    create_post_compact_file_attachments,
    create_skill_attachment_if_needed,
)
from .post_compact_cleanup import run_post_compact_cleanup
from .prompt import format_compact_summary, get_compact_prompt
from .snip_compact import snip_compact
from .tool_result_budget import apply_tool_result_budget

__all__ = [
    "DEFAULT_COMPRESSION_GATE_SKIP_RATIO",
    "ENV_COMPRESSION_GATE_SKIP_RATIO",
    "POST_COMPACT_MAX_FILES_TO_RESTORE",
    "POST_COMPACT_MAX_TOKENS_PER_FILE",
    "POST_COMPACT_TOKEN_BUDGET",
    "ApiRound",
    "AutoCompactTracking",
    "CollapseCommit",
    "CompactionResult",
    "CompressionPipeline",
    "CompressionResult",
    "ContextCollapseStore",
    "apply_tool_result_budget",
    "auto_compact_if_needed",
    "calculate_token_warning_state",
    "clear_compact_warning_suppression",
    "compact_conversation",
    "create_plan_attachment_if_needed",
    "create_post_compact_file_attachments",
    "create_skill_attachment_if_needed",
    "format_compact_summary",
    "get_auto_compact_threshold",
    "get_compact_prompt",
    "get_effective_context_window_size",
    "group_messages_by_api_round",
    "is_auto_compact_enabled",
    "is_compact_warning_suppressed",
    "reactive_compact",
    "resolve_skip_ratio_from_env",
    "run_compression_pipeline",
    "run_post_compact_cleanup",
    "should_auto_compact",
    "should_run_compression_pipeline",
    "snip_compact",
    "suppress_compact_warning",
    "truncate_head_for_ptl_retry",
]
