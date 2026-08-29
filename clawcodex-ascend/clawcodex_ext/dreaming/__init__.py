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

# pylint: disable=cyclic-import

"""Dreaming — background memory consolidation subsystem.

Mirrors the upstream ``claude-code-best/src/services/autoDream/`` stack:

* :mod:`clawcodex_ext.dreaming.config` — enable + thresholds.
* :mod:`clawcodex_ext.dreaming.paths` — auto-memory path re-exports.
* :mod:`clawcodex_ext.dreaming.lock` — mtime-as-lastConsolidatedAt
  file lock inside the auto-memory dir.
* :mod:`clawcodex_ext.dreaming.prompt` — 4-phase consolidation prompt
  builder.
* :mod:`clawcodex_ext.dreaming.runner` — placeholder for the actual
  LLM-backed "forked agent" (Phase A stub; real wiring in a follow-up).
* :mod:`clawcodex_ext.dreaming.service` — main loop / gates / dispatch.

Public surface intentionally re-exports the consolidated API so callers
can ``from clawcodex_ext.dreaming import …`` without picking modules.
"""

from __future__ import annotations

from clawcodex_ext.dreaming.config import (
    DEFAULT_DREAM_CONFIG,
    DreamConfig,
    get_dream_config,
    is_auto_dream_enabled,
    set_dream_config,
)
from clawcodex_ext.dreaming.cron_integration import (
    DREAM_DEFAULT_CRON,
    DREAM_PERMANENT_PROMPT,
    DREAM_PERMANENT_TASK_ID,
    install_and_wire_dream,
    install_dream_permanent_cron_task,
    wire_dream_fire_handler,
)
from clawcodex_ext.dreaming.lock import (
    HOLDER_STALE_MS,
    LOCK_FILE_NAME,
    force_release_if_stale,
    get_holder_pid,
    get_lock_age_seconds,
    is_lock_stale,
    list_sessions_touched_since,
    read_last_consolidated_at,
    record_consolidation,
    rollback_consolidation_lock,
    try_acquire_consolidation_lock,
)
from clawcodex_ext.dreaming.paths import (
    get_auto_mem_entrypoint,
    get_auto_mem_path,
    is_auto_memory_enabled,
    is_kairos_active,
    project_transcript_dir,
)
from clawcodex_ext.dreaming.prompt import (
    DREAM_PROMPT_PREFIX,
    build_consolidation_prompt,
)
from clawcodex_ext.dreaming.runner import (
    DreamRunResult,
    run_dream_consolidation,
)
from clawcodex_ext.dreaming.service import (
    execute_auto_dream,
    get_active_registry,
    init_auto_dream,
    kill_dream_task,
    manual_dream,
)

__all__ = [
    # config
    "DEFAULT_DREAM_CONFIG",
    "DREAM_DEFAULT_CRON",
    "DREAM_PERMANENT_PROMPT",
    "DREAM_PERMANENT_TASK_ID",
    "DreamConfig",
    "get_dream_config",
    "install_and_wire_dream",
    "install_dream_permanent_cron_task",
    "is_auto_dream_enabled",
    "set_dream_config",
    "wire_dream_fire_handler",
    # paths
    "get_auto_mem_entrypoint",
    "get_auto_mem_path",
    "is_auto_memory_enabled",
    "is_kairos_active",
    "project_transcript_dir",
    # lock
    "HOLDER_STALE_MS",
    "LOCK_FILE_NAME",
    "force_release_if_stale",
    "get_holder_pid",
    "get_lock_age_seconds",
    "is_lock_stale",
    "list_sessions_touched_since",
    "read_last_consolidated_at",
    "record_consolidation",
    "rollback_consolidation_lock",
    "try_acquire_consolidation_lock",
    # prompt
    "DREAM_PROMPT_PREFIX",
    "build_consolidation_prompt",
    # runner
    "DreamRunResult",
    "run_dream_consolidation",
    # service
    "execute_auto_dream",
    "get_active_registry",
    "init_auto_dream",
    "kill_dream_task",
    "manual_dream",
]
