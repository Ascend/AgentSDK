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

"""Startup header widget — the Claw Codex banner.

Port of :meth:`src.repl.core.ClawcodexREPL._print_startup_header` rendered
inside a Textual ``Static`` so the banner becomes a first-class, persistent
component of the TUI (matching the Ink ``<Header>`` in
``typescript/src/screens/REPL.tsx``) instead of a one-shot ``print`` at
startup.
"""

from __future__ import annotations

from pathlib import Path

from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.widgets import Static


def _display_cwd(workspace_root: Path) -> str:
    try:
        home = Path.home()
        rel = workspace_root.relative_to(home)
        return f"~/{rel}" if str(rel) != "." else "~"
    except Exception:
        return str(workspace_root)


def _truncate_middle(text: str, max_width: int) -> str:
    if max_width <= 0 or len(text) <= max_width:
        return text
    if max_width <= 3:
        return text[:max_width]
    keep = max_width - 1
    head = keep // 2
    tail = keep - head
    return text[:head] + "…" + text[-tail:]


class StartupHeader(Static):
    """Fixed banner at the top of the TUI.

    Re-uses the Rich layout from the legacy REPL so the visual identity
    stays consistent between the two interactive stacks.
    """

    DEFAULT_CSS = """
    StartupHeader {
        height: auto;
        padding: 0 0;
        background: $background;
    }
    """

    def __init__(
        self,
        *,
        version: str,
        model: str,
        provider: str,
        workspace_root: Path,
        slash_hints: str = "/help  •  /tools  •  /stream  •  /render-last  •  /exit",
        width_hint: int | None = None,
    ) -> None:
        self._version = version
        self._model = model
        self._provider = provider
        self._workspace_root = Path(workspace_root)
        self._slash_hints = slash_hints
        self._width_hint = width_hint
        super().__init__(self._render_banner(), markup=False)

    def refresh_banner(self, *, model: str | None = None, provider: str | None = None) -> None:
        """Update model/provider labels live (e.g. after `/model ...`)."""
        if model is not None:
            self._model = model
        if provider is not None:
            self._provider = provider
        self.update(self._render_banner())

    def _render_banner(self) -> Panel:
        display_path = _display_cwd(self._workspace_root)
        width = self._width_hint or 80
        content_width = max(28, min(width - 12, 72))
        # Use palette colors when available, fall back to hardcoded defaults
        try:
            palette = self.app.palette
            muted = palette.text_muted
            text = palette.text
            primary = palette.primary
            secondary = palette.secondary
            success = palette.success
            info = palette.info
            border = palette.border
        except Exception:
            muted = "#9a9a9a"
            text = "#e6e6e6"
            primary = "#8ab4f8"
            secondary = "#c58af9"
            success = "#7ee787"
            info = "#79c0ff"
            border = "#2a2a33"
        table = Table.grid(padding=(0, 1))
        table.add_column(style=muted, justify="right", no_wrap=True)
        table.add_column(style=text, ratio=1)
        table.add_row(
            "Version",
            Text.assemble(
                ("ClawCodex", f"bold {text}"),
                ("  ", ""),
                (f"v{self._version}", f"bold {info}"),
            ),
        )
        table.add_row("Model", Text(self._model or "unknown", style=f"bold {secondary}"))
        table.add_row(
            "Provider",
            Text(f"{self._provider.upper()} Provider", style=f"bold {success}"),
        )
        table.add_row(
            "Workspace",
            Text(_truncate_middle(display_path, content_width - 12), style=f"bold {primary}"),
        )

        footer = Text(self._slash_hints, style="dim")

        # telemetry notice — show when both stats collection and error
        # reporting are enabled.  Best-effort & swallowed on failure so a
        # misconfigured telemetry package never blocks TUI startup.
        try:
            from telemetry.config import load_config as _load_telemetry_cfg

            _tc = _load_telemetry_cfg()
            if _tc.enabled and _tc.reporting.reporting_enabled:
                telemetry_notice = Group(
                    Align.center(
                        Text(
                            "Telemetry: stats ✓ · error reporting ✓  — /telemetry to configure",
                            style="dim italic",
                        )
                    ),
                    Align.center(
                        Text(
                            "Collects usage data & error reports; may be uploaded periodically.",
                            style="dim italic",
                        )
                    ),
                )
                body = Group(table, telemetry_notice, Text(""), Align.center(footer))
            else:
                body = Group(table, Text(""), Align.center(footer))
        except Exception:
            body = Group(table, Text(""), Align.center(footer))
        return Panel(
            body,
            border_style=border,
            title=f"[bold {primary}] CLAWCODEX [/bold {primary}]",
            subtitle="[dim]interactive terminal[/dim]",
            padding=(0, 2),
        )
