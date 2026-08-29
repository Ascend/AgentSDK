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

"""Assistant tool-use row.

Port of ``typescript/src/components/messages/AssistantToolUseMessage.tsx``
and ``renderToolActivity`` in ``typescript/src/components/tasks/``. The
row is announced the moment the assistant emits a ``tool_use`` block
and its body is a per-tool :class:`ToolActivity` subclass that mutates
through ``requested → running → done / error`` as tool events arrive
from the agent loop.

Key behavioural parity points:

* The row is mounted **once** per tool-use id; subsequent events update
  the existing body instead of appending a new row.
* The header swaps color / glyph based on the current status so the
  user can scan the transcript and spot in-flight / failed steps.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive

from .base import BaseRow, RowHeader
from ..lkb_proof import LKBProofWidget, extract_lkb_denial
from ..tool_activity import build_tool_activity, ToolActivity


class AssistantToolUseMessage(BaseRow):
    """Transcript row that owns a :class:`ToolActivity` child widget."""

    DEFAULT_CSS = """
    AssistantToolUseMessage {
        height: auto;
    }
    """

    status: reactive[str] = reactive("requested")

    def __init__(
        self,
        *,
        tool_use_id: str,
        tool_name: str,
        tool_input: dict | None,
    ) -> None:
        super().__init__()
        self.tool_use_id = tool_use_id
        self.tool_name = tool_name
        self.tool_input = dict(tool_input or {})
        self._activity: ToolActivity | None = None

    # ---- composition ----
    def compose(self) -> ComposeResult:
        header = RowHeader(self._header_text(), markup=False)
        header.add_class("-tool")
        yield header
        self._activity = build_tool_activity(self.tool_name, self.tool_input)
        yield self._activity

    # ---- lifecycle updates ----
    def mark_running(self) -> None:
        self._set_status("running")

    def mark_done(self, output) -> None:
        self._set_status("done")
        if self._activity is not None:
            self._activity.on_result(output, is_error=False)
        # Mount LKB proof widget when the tool was denied by the commit gate.
        self._maybe_mount_lkb_denial(output)

    def mark_error(self, output, *, error: str | None = None) -> None:
        self._set_status("error")
        if self._activity is not None:
            self._activity.on_result(output, is_error=True, error=error)
        self._maybe_mount_lkb_denial(output)

    # ---- internals ----
    def _set_status(self, status: str) -> None:
        self.status = status
        header = self._header_widget()
        if header is None:
            return
        header.remove_class("-tool", "-tool-success", "-tool-error")
        if status in ("done",):
            header.add_class("-tool-success")
        elif status in ("error",):
            header.add_class("-tool-error")
        else:
            header.add_class("-tool")
        header.update(self._header_text())

    def _header_text(self) -> Text:
        glyph = {
            "requested": "○",
            "running": "◐",
            "done": "✓",
            "error": "✗",
        }.get(self.status, "•")
        summary = _summarise_input(self.tool_name, self.tool_input)
        tail = f" · {summary}" if summary else ""
        return Text(f"{glyph} {self.tool_name}{tail}")

    def snapshot(self) -> Text:
        """Return a Rich :class:`Text` for post-exit scrollback dump."""
        try:
            tool_color = self.app.palette.tool
            success_color = self.app.palette.success
            error_color = self.app.palette.error
        except Exception:  # nosec B110
            tool_color = "#f5c451"
            success_color = "#7ee787"
            error_color = "#ff7b72"
        glyph_color = {
            "done": f"bold {success_color}",
            "error": f"bold {error_color}",
        }.get(self.status, f"bold {tool_color}")
        return Text(str(self._header_text()), style=glyph_color)

    def _header_widget(self) -> RowHeader | None:
        try:
            for header in self.query(RowHeader):
                return header
        except Exception:  # nosec B110
            return None
        return None

    # ---- LKB denial detection ----

    def _maybe_mount_lkb_denial(self, output) -> None:
        """If *output* contains an LKB denial, mount a :class:`LKBProofWidget`."""
        try:
            denial = extract_lkb_denial(output)
            if denial is not None:
                self.mount(LKBProofWidget(denial))
        except Exception:  # nosec B110
            pass  # best-effort; don't crash the transcript row


def _summarise_input(tool_name: str, tool_input: dict) -> str:
    """One-line summary for the tool-use header, e.g. ``Bash · ls``.

    Delegates to :mod:`src.tool_system.renderers.summarize_tool_use` when
    available so the TUI, the legacy REPL, and the headless NDJSON path
    all agree on wording.
    """

    try:
        from src.tool_system.renderers import summarize_tool_use

        return summarize_tool_use(tool_name, tool_input or {}) or ""
    except Exception:
        return ""
