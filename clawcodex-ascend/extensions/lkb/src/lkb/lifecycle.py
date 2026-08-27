#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# AgentSDK validates these split-package and target-lint diagnostics in the complete tested source.
# pylint: disable=E0402
"""Public facade for LKB board lifecycle operations."""

from .lifecycle_core import (
    GC_QUARANTINE_AGE_SECONDS,
    GC_SESSION_ORPHAN_AGE_SECONDS,
    GC_TEMP_AGE_SECONDS,
    GC_TOMBSTONE_AGE_SECONDS,
    LifecycleData,
    LifecycleError,
    LifecycleTransitionDenied,
    VALID_STATES,
    archive_board,
    board_lifecycle_state,
    close_board,
    genesis_lifecycle,
    ordinary_write_allowed,
    ordinary_write_denial_reason,
    read_archive,
    reopen_board,
    restore_board,
    transition,
    trash_board,
)
from .lifecycle_gc import GcCandidate, gc_apply, gc_scan
from .lifecycle_purge import purge_board, read_tombstone, tombstone_path

__all__ = [
    "GC_QUARANTINE_AGE_SECONDS",
    "GC_SESSION_ORPHAN_AGE_SECONDS",
    "GC_TEMP_AGE_SECONDS",
    "GC_TOMBSTONE_AGE_SECONDS",
    "GcCandidate",
    "LifecycleData",
    "LifecycleError",
    "LifecycleTransitionDenied",
    "VALID_STATES",
    "archive_board",
    "board_lifecycle_state",
    "close_board",
    "gc_apply",
    "gc_scan",
    "genesis_lifecycle",
    "ordinary_write_allowed",
    "ordinary_write_denial_reason",
    "purge_board",
    "read_archive",
    "read_tombstone",
    "reopen_board",
    "restore_board",
    "tombstone_path",
    "transition",
    "trash_board",
]
