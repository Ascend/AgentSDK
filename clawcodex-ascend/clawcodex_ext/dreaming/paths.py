#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Auto-memory path re-exports for the dreaming subsystem.

The upstream ``claude-code-best/src/memdir/paths.ts`` logic is fully
ported in ``src/memdir/paths.py``; this module is a thin
forwarder so :mod:`clawcodex_ext.dreaming` does not need to reach
across the package boundary at every call site.

Also exposes :func:`project_transcript_dir` and :func:`is_kairos_active`
helpers that upstream packs into the bootstrap state but clawcodex
does not currently model the same way. Both are best-effort
approximations — KAIROS is upstream-only for now and the transcript
dir falls back to ``SessionStorage.SESSIONS_DIR`` when the per-cwd
project dir is not yet registered.
"""

from __future__ import annotations

import os
from pathlib import Path

# Re-export the upstream path helpers unchanged.
from clawcodex_ext.memdir.paths import (
    get_auto_mem_entrypoint,
    get_auto_mem_path,
    is_auto_memory_enabled,
)

__all__ = [
    "get_auto_mem_entrypoint",
    "get_auto_mem_path",
    "is_auto_memory_enabled",
    "is_kairos_active",
    "project_transcript_dir",
]


def is_kairos_active() -> bool:
    """Whether the KAIROS feature gate is active for this session.

    clawcodex does not yet implement KAIROS (deferred — F-100 design
    decision #1 in PROGRESS.md §十三). Returns ``False`` unless
    explicitly opted in via the ``CLAWCODEX_KAIROS`` env var so
    dream can still be tested in isolation.

    Returned by the service main loop so future KAIROS-mode dream
    (a disk-skill variant) can branch off the autoDream path.
    """
    return os.environ.get("CLAWCODEX_KAIROS", "").lower() in ("1", "true", "yes")


def project_transcript_dir(cwd: str | os.PathLike[str] | None = None) -> str:
    """Resolve the transcript directory for *cwd*.

    Upstream uses ``getProjectDir(getOriginalCwd())`` which lives in
    a typed session storage module. clawcodex's
    ``src.services.session_storage.SessionStorage.SESSIONS_DIR`` is
    global; we approximate the per-cwd project dir by joining it
    with the sanitized path tail. When the per-cwd subdir does not
    exist, we still return it (not the global SESSIONS_DIR) so
    callers that scan for sessions see a per-project empty view
    rather than inflating the count with unrelated sessions.

    Returns an absolute path string. Never raises.
    """
    try:
        from clawcodex_ext.services.session_storage import SESSIONS_DIR
    except Exception:
        return os.getcwd()

    base = Path(SESSIONS_DIR)
    target = Path(cwd) if cwd is not None else Path.cwd()
    try:
        from clawcodex_ext.memdir.paths import sanitize_path

        slug = sanitize_path(str(target))
    except Exception:
        slug = target.name or "default"
    candidate = base / slug
    return str(candidate)
