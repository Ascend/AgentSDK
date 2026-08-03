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

"""Shared primitives for transcript row widgets.

Every row is a plain Textual ``Widget`` so the :class:`TranscriptView`
container can scroll a homogeneous list. We deliberately avoid subclassing
``Static`` here because several rows (streaming text, tool activity) need
dynamic children, not a single rendered block.
"""

from __future__ import annotations

from rich.markdown import Markdown
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Static


class BaseRow(Widget):
    """Common base for transcript rows.

    Subclasses typically override :meth:`compose` and add their own
    reactive attributes. ``BaseRow`` only provides the vertical padding
    / margin that keeps consecutive rows visually separated.
    """

    DEFAULT_CSS = """
    BaseRow {
        layout: vertical;
        height: auto;
        margin: 0 0 1 0;
    }
    """


class SystemMessage(BaseRow):
    """A muted, one-off informational / error row.

    Analogue of ``<Text dimColor>…</Text>`` usages scattered through
    ``typescript/src/screens/REPL.tsx`` (onboarding callouts, exit
    reminders, and error rows emitted by :meth:`AgentBridge`).
    """

    DEFAULT_CSS = """
    SystemMessage {
        height: auto;
        margin: 0 0 1 0;
    }
    SystemMessage.-error > Static {
        color: $error;
    }
    SystemMessage.-warning > Static {
        color: $warning;
    }
    SystemMessage.-muted > Static {
        color: $text-muted;
    }
    SystemMessage.-light > Static {
        color: $text-disabled;
    }
    """

    def __init__(
        self,
        text: str,
        *,
        style: str = "muted",
        render: str = "plain",
    ) -> None:
        super().__init__()
        self._text = text
        self._style = style
        self._render_mode = render
        if style == "error":
            self.add_class("-error")
        elif style == "light":
            self.add_class("-light")
        elif style == "warning":
            self.add_class("-warning")
        else:
            self.add_class("-muted")

    def compose(self) -> ComposeResult:
        if self._render_mode == "markdown":
            yield Static(Markdown(self._text))
        else:
            yield Static(Text(self._text), markup=False)

    def update_text(self, text: str) -> None:
        """Swap the row contents in place without re-mounting children."""
        self._text = text
        try:
            static = self.query_one(Static)
            if self._render_mode == "markdown":
                static.update(Markdown(text))
            else:
                static.update(Text(text))
        except Exception:  # nosec B110
            # Before mount — next compose() will pick up the new text.
            pass

    def snapshot(self):
        """Return a Rich renderable for post-exit scrollback dump."""

        style = "dim" if self._style != "error" else "red"
        if self._render_mode == "markdown":
            return Markdown(self._text)
        return Text(self._text, style=style)


class RowHeader(Static):
    """One-line row header shared by user / assistant / tool rows.

    Kept as a standalone ``Static`` so row subclasses can reuse the same
    styling across the transcript without duplicating CSS.
    """

    DEFAULT_CSS = """
    RowHeader {
        height: 1;
        width: auto;
        padding: 0 1;
    }
    RowHeader.-user { color: $primary; text-style: bold; }
    RowHeader.-assistant { color: $secondary; text-style: bold; }
    RowHeader.-tool { color: $warning; text-style: bold; }
    RowHeader.-tool-success { color: $success; text-style: bold; }
    RowHeader.-tool-error { color: $error; text-style: bold; }
    """


def row_container() -> Vertical:
    """Shortcut for the standard ``Vertical`` that wraps a row body.

    Extracted so future rows stay consistent without copy-pasting the
    DEFAULT_CSS sizing rules.
    """

    return Vertical()
