# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Transcript row widgets.

Each widget is a self-contained Textual ``Widget`` representing one row
in the scrollable transcript. Rows correspond roughly 1-to-1 with the
components under ``typescript/src/components/messages/`` in the ink
reference implementation:

* :class:`BaseRow`                  — shared padding / header plumbing.
* :class:`UserTextMessage`          — the ``❯`` user-turn row.
* :class:`AssistantTextMessage`     — live-streaming assistant text that
  finalises to Markdown at end-of-turn.
* :class:`AssistantToolUseMessage`  — the pre-run announcement for a
  tool call; its body is a
  :class:`src.tui.widgets.tool_activity.ToolActivity` subclass that
  transitions through ``requested → running → done / error``.
* :class:`ToolResultRow`            — terminal summary row shown when a
  tool result comes back (used for non-grouped paths).
* :class:`SystemMessage`            — muted system/error notifications.
"""

from .base import BaseRow, SystemMessage
from .user_text import UserTextMessage
from .assistant_text import AssistantTextMessage
from .assistant_tool_use import AssistantToolUseMessage
from .assistant_advisor import AssistantAdvisorMessage
from .tool_result import ToolResultRow

__all__ = [
    "BaseRow",
    "SystemMessage",
    "UserTextMessage",
    "AssistantTextMessage",
    "AssistantToolUseMessage",
    "AssistantAdvisorMessage",
    "ToolResultRow",
]
