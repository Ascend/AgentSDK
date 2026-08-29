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

"""Dismissible status view for the Claude-style ``/goal`` command."""

from __future__ import annotations

from collections.abc import Iterator

from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import Static

from .dialog_base import DialogScreen


class GoalStatusScreen(DialogScreen[None]):
    """Show local goal state without adding it to transcript scrollback."""

    title_text = "Goal"
    footer_hint = "Esc to dismiss"
    BINDINGS = [
        Binding("escape", "cancel", "Close", show=False),
        Binding("q", "cancel", "Close", show=False),
    ]

    def __init__(self, status_text: str) -> None:
        self._status_text = status_text.strip()
        super().__init__()

    def build_body(self) -> Iterator[Widget]:
        body = self._status_text
        if body.startswith("Goal\n\n"):
            body = body[len("Goal\n\n") :]
        yield Static(body, markup=False)


__all__ = ["GoalStatusScreen"]
