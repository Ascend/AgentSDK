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

"""MCP description-truncation helper.

Phase 10 WI-10.3 (gap #27). Mirrors TS' 2048-char truncation logic
that previously lived inline in both ``client.py:141-147`` (server
``instructions``) and ``tool_wrapper.py`` (tool ``description``). One
helper, two callers.
"""

from __future__ import annotations

MAX_MCP_DESCRIPTION_LENGTH = 2048


def truncate_description(text: str | None) -> str | None:
    """Truncate ``text`` to ``MAX_MCP_DESCRIPTION_LENGTH`` chars + suffix.

    Returns ``None`` if input is None or empty. Suffix
    ``"... [truncated]"`` is appended when truncation actually fires,
    so the model can see why the field is cut off.
    """
    if not text:
        return text
    if len(text) <= MAX_MCP_DESCRIPTION_LENGTH:
        return text
    return text[:MAX_MCP_DESCRIPTION_LENGTH] + "... [truncated]"
