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

"""F-88 TUI monitor panel.

Opened with Shift+Down.  Shows the list of active monitor tasks on the left
and a live tail of the selected task's log on the right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import RichLog, Static

from clawcodex_ext.services.monitor.controller import MonitorController
from clawcodex_ext.services.monitor.text_tail import TextTailFollower


class MonitorPanel(ModalScreen[None]):
    """Shift+Down monitor task panel."""

    BINDINGS = [
        Binding("up", "prev_task", "Previous", show=False),
        Binding("down", "next_task", "Next", show=False),
        Binding("d", "delete_task", "Delete", show=False),
        Binding("r", "refresh", "Refresh", show=False),
        Binding("q,escape", "cancel", "Close", show=False),
    ]

    DEFAULT_CSS = """
    MonitorPanel {
        align: center middle;
    }
    MonitorPanel > #monitor-panel-container {
        width: 96;
        max-width: 95%;
        height: 90%;
        border: round $primary;
        background: $surface;
    }
    MonitorPanel #monitor-task-list {
        width: 40%;
        height: 100%;
        border: solid $primary-darken-2;
    }
    MonitorPanel #monitor-task-list .monitor-task {
        padding: 0 1;
    }
    MonitorPanel #monitor-task-list .monitor-task.selected {
        background: $primary-darken-2;
        text-style: bold;
    }
    MonitorPanel #monitor-output {
        width: 60%;
        height: 100%;
        border: solid $primary-darken-2;
    }
    MonitorPanel #monitor-footer {
        dock: bottom;
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(self, tool_context: Any) -> None:
        super().__init__()
        self._ctrl = MonitorController(tool_context)
        self._tasks: list[Any] = []
        self._selected_idx: int = 0
        self._follower: TextTailFollower | None = None
        self._output_widget: RichLog | None = None
        self._list_widget: Static | None = None
        self._refresh_interval: Any | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="monitor-panel-container"):
            with Vertical(id="monitor-task-list"):
                yield Static("Loading tasks…", id="monitor-list-content")
            with Vertical(id="monitor-output-pane"):
                log = RichLog(id="monitor-output", highlight=False, wrap=True)
                yield log
        yield Static(
            "↑/↓ select · d stop · r refresh · q/Esc close",
            id="monitor-footer",
        )

    def on_mount(self) -> None:
        self._output_widget = self.query_one("#monitor-output", RichLog)
        self._list_widget = self.query_one("#monitor-list-content", Static)
        self._refresh_tasks()
        self._refresh_interval = self.set_interval(0.5, self._tick)

    def on_unmount(self) -> None:
        if self._refresh_interval is not None:
            self._refresh_interval.stop()
        if self._follower is not None:
            # TailFollower.stop is async; fire-and-forget is fine for teardown.
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._follower.stop())
            except RuntimeError:
                pass

    def _refresh_tasks(self) -> None:
        self._tasks = self._ctrl.list_active()
        if self._selected_idx >= len(self._tasks):
            self._selected_idx = max(0, len(self._tasks) - 1)
        self._render_list()
        self._select_task(self._selected_idx)

    def _render_list(self) -> None:
        if self._list_widget is None:
            return
        if not self._tasks:
            self._list_widget.update("No active monitor tasks.")
            return
        lines: list[str] = []
        for idx, task in enumerate(self._tasks):
            desc = task.description or task.command
            interval = getattr(task, "interval_sec", None)
            interval_str = f" [{interval}s]" if interval else ""
            marker = "▸ " if idx == self._selected_idx else "  "
            lines.append(f"{marker}{task.id}{interval_str}: {desc}")
        self._list_widget.update("\n".join(lines))

    def _select_task(self, idx: int) -> None:
        if not self._tasks or idx < 0 or idx >= len(self._tasks):
            return
        self._selected_idx = idx
        self._render_list()
        task = self._tasks[idx]
        output_path = Path(task.output_path or task.output_file)
        self._follower = TextTailFollower(output_path, ring_size=task.tail_buffer_size)
        if self._output_widget is not None:
            self._output_widget.clear()
            try:
                text = output_path.read_text(encoding="utf-8", errors="replace")
                self._output_widget.write(text[-task.tail_buffer_size :])
            except OSError:
                pass

    def _tick(self) -> None:
        if self._follower is None or self._output_widget is None:
            return
        chunk = self._follower.read_available_now()
        if chunk:
            self._output_widget.write(chunk)

    def action_prev_task(self) -> None:
        if self._tasks:
            self._select_task((self._selected_idx - 1) % len(self._tasks))

    def action_next_task(self) -> None:
        if self._tasks:
            self._select_task((self._selected_idx + 1) % len(self._tasks))

    def action_delete_task(self) -> None:
        if not self._tasks:
            return
        task = self._tasks[self._selected_idx]
        self._ctrl.stop(task.id)
        self._refresh_tasks()

    def action_refresh(self) -> None:
        self._refresh_tasks()

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = ["MonitorPanel"]
