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

"""User turn row.

Port of ``typescript/src/components/messages/UserTextMessage.tsx``.
Renders a single row with a ``❯`` prefix in the primary color followed
by the user's prompt in bold text. Multi-line prompts are preserved.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import Static

from .base import BaseRow


class UserTextMessage(BaseRow):
    """A user prompt shown in the transcript."""

    DEFAULT_CSS = """
    UserTextMessage {
        height: auto;
    }
    UserTextMessage > Static {
        padding: 0 1;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static(self._build_text(), markup=False)

    def _build_text(self) -> Text:
        try:
            color = self.app.palette.user
        except Exception:
            color = "#8ab4f8"
        prefix = Text("❯ ", style=f"bold {color}")
        body = Text(self._text, style="bold")
        return prefix + body

    def snapshot(self) -> Text:
        """Return a Rich :class:`Text` for post-exit scrollback dump."""

        return self._build_text()
