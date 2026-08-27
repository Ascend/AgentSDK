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

"""Stable machine-readable error codes shared by every LKB layer."""

from __future__ import annotations

from enum import Enum


class LkbErrorCode(str, Enum):
    REQUEST_HASH_MISMATCH = "request_hash_mismatch"
    REVISION_CONFLICT = "revision_conflict"
    STALE_REVISION = "stale_revision"
    VALIDATION_DENIED = "validation_denied"
    PLAN_NOT_ACTIVE = "plan_not_active"
    TASK_NOT_FOUND = "task_not_found"
    NEEDS_RECHECK = "needs_recheck"
    NEEDS_REVIEW = "needs_review"
    INVALID_TASK = "invalid_task"
    DUPLICATE_TASK = "duplicate_task"
    UNKNOWN_TASK_REFERENCE = "unknown_task_reference"
    SELF_DEPENDENCY = "self_dependency"
    DEPENDENCY_CYCLE = "dependency_cycle"
    ALREADY_CLAIMED = "already_claimed"
    ALREADY_RESOLVED = "already_resolved"
    AGENT_BUSY = "agent_busy"
    INVALID_TRANSFER = "invalid_transfer"
    OWNER_REQUIRED = "owner_required"
    NOT_OWNER = "not_owner"
    BLOCKED = "blocked"
    INVALID_TRANSITION = "invalid_transition"
    DANGLING_DEPENDENCY = "dangling_dependency"
    INVALID_STATUS = "invalid_status"
    EMPTY_PATCH = "empty_patch"
    UNKNOWN_COMMAND = "unknown_command"
    OVERRIDE_REASON_REQUIRED = "override_reason_required"
    OVERRIDE_NOT_AUTHORIZED = "override_not_authorized"
    ADAPTER_ERROR = "adapter_error"
    INVALID_METADATA = "invalid_metadata"
    CREATE_DENIED = "create_denied"
    CROSS_AUTHORITY_DEPENDENCY = "cross_authority_dependency"
    TASK_ID_COLLISION = "task_id_collision"
    BOARD_STORE_CORRUPT = "board_store_corrupt"
    BOARD_STORE_BUSY = "board_store_busy"
    BOARD_STORE_IO_ERROR = "board_store_io_error"
    BOARD_STORE_DISK_FULL = "board_store_disk_full"
    BOARD_STORE_HASH_MISMATCH = "board_store_hash_mismatch"
    BOARD_STORE_UNSUPPORTED_FILESYSTEM = "board_store_unsupported_filesystem"
    INVALID_BOARD_ID = "invalid_board_id"
    BOARD_NOT_FOUND = "board_not_found"
    BOARD_TOMBSTONED = "board_tombstoned"
    BOARD_SCHEMA_TOO_NEW = "board_schema_too_new"
    IDEMPOTENCY_KEY_REUSED = "idempotency_key_reused"
    AUDIT_SIZE_LIMIT = "audit_size_limit"
    MIGRATION_FAILED = "migration_failed"
    LIFECYCLE_TRANSITION_DENIED = "lifecycle_transition_denied"

    def __str__(self) -> str:
        return self.value


ERROR_CODES = frozenset(code.value for code in LkbErrorCode)


def coerce_error_code(value: object) -> LkbErrorCode | None:
    if isinstance(value, LkbErrorCode):
        return value
    if isinstance(value, str) and value:
        try:
            return LkbErrorCode(value)
        except ValueError:
            return None
    return None


__all__ = ["ERROR_CODES", "LkbErrorCode", "coerce_error_code"]
