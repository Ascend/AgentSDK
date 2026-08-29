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


"""TUI render helpers and optional screen for ultraplan status."""

from __future__ import annotations
# pylint: disable=C0104

from typing import Iterator

from rich.console import Group
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from clawcodex_ext.services.ultraplan.executor import PlanExecutor
from clawcodex_ext.services.ultraplan.models import Plan, StepStatus


_STATUS_STYLE = {
    StepStatus.PENDING: "dim",
    StepStatus.IN_PROGRESS: "cyan",
    StepStatus.COMPLETED: "green",
    StepStatus.FAILED: "red",
    StepStatus.SKIPPED: "yellow",
    StepStatus.BLOCKED: "magenta",
}


def render_ultraplan_panel(plan: Plan) -> Group:
    executor = PlanExecutor(plan)
    progress = executor.progress()
    title = Text(f"{plan.id} - {plan.title}", style="bold")
    status = Text(
        f"status: {plan.status.value}  progress: {progress.done}/{progress.total} ({int(progress.ratio * 100)}%)",
        style="dim",
    )
    bar = ProgressBar(total=max(progress.total, 1), completed=progress.done, width=40)
    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("Sub-plan", ratio=2)
    table.add_column("Step", ratio=4)
    table.add_column("Status", ratio=1)
    table.add_column("Notes", ratio=3)
    for sub_plan in plan.sub_plans:
        for step in sub_plan.steps:
            style = _STATUS_STYLE.get(step.status, "")
            table.add_row(
                sub_plan.title,
                f"{step.id}: {step.title}",
                Text(step.status.value, style=style),
                step.error or step.notes or "",
            )
    return Group(title, status, bar, table)


try:
    from textual.app import ComposeResult
    from textual.widget import Widget
    from textual.widgets import Static

    from .dialog_base import DialogScreen

    class UltraplanPanelScreen(DialogScreen[None]):
        title_text = "Ultraplan"
        footer_hint = "n next step · p pause · e edit · q close"

        def __init__(self, plan: Plan) -> None:
            self.plan = plan
            super().__init__()

        def build_body(self) -> Iterator[Widget]:
            yield Static(render_ultraplan_panel(self.plan), markup=False)

        def compose(self) -> ComposeResult:
            yield from super().compose()

        def on_key(self, event) -> None:  # type: ignore[override]
            if event.key in {"q", "escape"}:
                self.dismiss(None)
                event.stop()

except ImportError:  # pragma: no cover - exercised in minimal test envs
    UltraplanPanelScreen = None  # type: ignore[assignment]


__all__ = ["UltraplanPanelScreen", "render_ultraplan_panel"]
