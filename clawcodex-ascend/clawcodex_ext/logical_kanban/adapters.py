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

"""Compatibility shim — delegate to lkb.adapters with LkbToolResult → ToolResult conversion."""

from typing import Any

from lkb.adapters import (  # noqa: F401
    _accepted_lkb,
    _denied_result as _lkb_denied_result,
    maybe_commit_task_update,
    maybe_commit_todo_write,
    prepare_task_change,
    prepare_todo_write,
)
from clawcodex_ext.tool_system.protocol import ToolResult


def _denied_result(
    tool_name: str,
    proposal: Any,
    validation: Any,
    commit: Any,
) -> ToolResult:
    """Convert LkbToolResult → clawcodex ToolResult at boundary."""
    lkb_result = _lkb_denied_result(tool_name, proposal, validation, commit)
    return ToolResult(
        name=lkb_result.name,
        output=lkb_result.output,
        is_error=lkb_result.is_error,
    )
