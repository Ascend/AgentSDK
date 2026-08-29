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

"""
Post-compact cleanup — clear caches and tracking state after compaction.

Port of ``typescript/src/services/compact/postCompactCleanup.ts``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PostCompactContext:
    """Minimal context for post-compact cleanup."""

    # Caches to clear (name → clear callable)
    caches: dict[str, Callable[[], None]] = field(default_factory=dict)
    # Read-file state tracking
    read_file_state: dict[str, Any] | None = None
    # Loaded nested memory paths
    loaded_nested_memory_paths: set[str] | None = None


def run_post_compact_cleanup(
    context: PostCompactContext | None = None,
) -> list[str]:
    """
    Clear caches and tracking state after a successful compaction.

    Returns a list of cache names that were cleared.
    """
    cleared: list[str] = []

    if context is None:
        return cleared

    # Clear registered caches
    for name, clear_fn in context.caches.items():
        try:
            clear_fn()
            cleared.append(name)
        except Exception:
            logger.warning("Failed to clear cache %r during post-compact cleanup", name)

    # Clear read-file state
    if context.read_file_state is not None:
        context.read_file_state.clear()
        cleared.append("read_file_state")

    # Clear loaded nested memory paths
    if context.loaded_nested_memory_paths is not None:
        context.loaded_nested_memory_paths.clear()
        cleared.append("loaded_nested_memory_paths")

    return cleared
