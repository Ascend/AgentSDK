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

"""Widgets used by the Claw Codex Textual TUI.

Public API (Phase 1):

* :class:`StartupHeader`   — one-shot banner at the top of the scroll region.
* :class:`TranscriptView`  — scrollable message list (replaces
  ``RichLog``-based Phase 0 ``Transcript``).
* :class:`StatusLine`      — spinner + verb + metrics bar.
* :class:`PromptInput`     — multi-line input + slash palette.
* :class:`FullscreenLayout` — four-region parity shell.

Backward-compat aliases kept so Phase 0 callers (tests, handoff) work
unchanged: :class:`Transcript` (now :class:`TranscriptView`) and
:class:`StatusBar` (now :class:`StatusLine`).
"""

from .fullscreen_layout import FullscreenLayout
from .header import StartupHeader
from .prompt_input import PromptInput, PromptSubmitted
from .status_line import StatusLine
from .transcript_view import Transcript, TranscriptView


# Phase 0 backward-compat alias.
StatusBar = StatusLine


__all__ = [
    "FullscreenLayout",
    "StartupHeader",
    "PromptInput",
    "PromptSubmitted",
    "StatusLine",
    "StatusBar",
    "Transcript",
    "TranscriptView",
]
