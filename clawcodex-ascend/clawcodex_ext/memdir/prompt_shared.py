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

"""Shared constants and helpers for private and team memory prompts."""

from __future__ import annotations

from pathlib import Path

from clawcodex_ext.memdir.paths import get_claude_config_home_dir

ENTRYPOINT_NAME = "MEMORY.md"
MAX_ENTRYPOINT_LINES = 200
# ~125 chars/line at 200 lines. p100 observed: 197KB under 200 lines.
MAX_ENTRYPOINT_BYTES = 25_000


def get_sessions_dir() -> Path:
    """Return the session store without an unmigrated core dependency."""
    return Path(get_claude_config_home_dir()) / "sessions"


def format_file_size(byte_count: int) -> str:
    """Return a compact human-readable file size."""
    if byte_count < 1024:
        return f"{byte_count}B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f}KB"
    return f"{byte_count / (1024 * 1024):.1f}MB"


def build_searching_past_context_section(auto_mem_dir: str) -> list[str]:
    """Build the memory and transcript search guidance.

    MEMDIR-1: upstream gates this on ``tengu_coral_fern``, which the vendored
    GrowthBook stub's ``_openBuildDefaults`` sets to TRUE — so the reference
    build emits it for every user, and this port emits it unconditionally
    (no flag system here). TS picks shell-grep forms when the dedicated Grep
    tool is hidden (ant-native embedded search / REPL script mode); this
    port always ships the Grep tool, so the tool-invocation forms are used
    unconditionally. The transcript target is this port's saved-session
    store (the ``sessions/`` dir under the clawcodex config root —
    ``$CLAWCODEX_CONFIG_DIR`` or ``~/.clawcodex`` — ``*.json``) rather than
    the reference project-transcript dir.
    """
    sessions_dir = str(get_sessions_dir())
    mem_search = f'Grep with pattern="<search term>" path="{auto_mem_dir}" glob="*.md"'
    transcript_search = f'Grep with pattern="<search term>" path="{sessions_dir}/" glob="*.json"'
    return [
        "## Searching past context",
        "",
        "When looking for past context:",
        "1. Search topic files in your memory directory:",
        "```",
        mem_search,
        "```",
        "2. Session transcript logs (last resort — large files, slow):",
        "```",
        transcript_search,
        "```",
        "Use narrow search terms (error messages, file paths, function names) rather than broad keywords.",
        "",
    ]
