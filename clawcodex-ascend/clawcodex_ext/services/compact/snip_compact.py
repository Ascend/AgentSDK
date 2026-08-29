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
Layer 2: Snip compact — stub matching typescript/src/services/compact/snipCompact.ts.

The TS implementation is a stub that returns null (not implemented).
We match that behavior here to avoid aggressively trimming tool results
that the model may need to reference later in the conversation.
"""

from __future__ import annotations

from clawcodex_ext.types.messages import Message  # pylint: disable=no-name-in-module

SNIPPED_MARKER = "[Snipped: tool result too old]"
DEFAULT_KEEP_RECENT = 10


def snip_compact(
    messages: list[Message],
    keep_recent: int = DEFAULT_KEEP_RECENT,
) -> tuple[list[Message], int]:
    return list(messages), 0
