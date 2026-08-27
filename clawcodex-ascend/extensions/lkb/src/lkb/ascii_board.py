#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# AgentSDK validates these split-package and target-lint diagnostics in the complete tested source.
# pylint: disable=C0104,E0402
"""Pure-ASCII LKB board renderer (spec §8.4, Phase 7).

Renders an :class:`lkb.read_model.LkbBoardView` as plain ASCII.  No ANSI
colours, no external Rich dependency - the text content is complete on
its own and stable for golden-snapshot tests.

* Display width follows East-Asian width (Chinese/full-width chars
  count as 2 columns, not ``len()``) - spec §8.4 / LKB-VIEW-011.
* Long titles are truncated with an ellipsis.
* Narrow terminals degrade to multi-line cards (LKB-VIEW-005).
* Sorting is stable by task ID.
"""

from __future__ import annotations

import unicodedata

from .read_model import LkbBoardView

__all__ = ["display_width", "truncate", "render_board"]


def display_width(text: str) -> int:
    """Return the display column width of *text*.

    East-Asian wide/full-width characters count as 2; zero-width
    combining marks as 0; everything else as 1.
    """
    width = 0
    for ch in text:
        eaw = unicodedata.east_asian_width(ch)
        if eaw in ("W", "F"):
            width += 2
        elif unicodedata.category(ch).startswith("M"):
            pass  # combining mark - no advance
        else:
            width += 1
    return width


def truncate(text: str, max_width: int) -> str:
    """Truncate *text* to *max_width* display columns with an ellipsis."""
    if display_width(text) <= max_width:
        return text
    out: list[str] = []
    width = 0
    ellipsis = "..."
    ellipsis_w = display_width(ellipsis)
    for ch in text:
        eaw = unicodedata.east_asian_width(ch)
        cw = 2 if eaw in ("W", "F") else (0 if unicodedata.category(ch).startswith("M") else 1)
        if width + cw + ellipsis_w > max_width:
            break
        out.append(ch)
        width += cw
    return "".join(out) + ellipsis


def _pad(text: str, width: int) -> str:
    """Left-align *text* in a field of *width* display columns."""
    dw = display_width(text)
    if dw >= width:
        return text
    return text + " " * (width - dw)


def render_board(view: LkbBoardView, *, width: int = 100, compact: bool = False) -> str:
    """Render *view* as a pure-ASCII board string."""
    lines: list[str] = []
    header = (
        f"LKB BOARD: {view.display_name} / {view.plan_title} "
        f"[{view.plan_state}] Rev {view.store_revision}.{view.plan_revision}"
    )
    if compact:
        lines.append(f"# {header}")
    else:
        bar = _rule(header, width)
        lines.append(bar)
    lines.append(
        f"| Ready {view.summary.ready} | Running {view.summary.running} | "
        f"Blocked {view.summary.blocked} | Recheck {view.summary.needs_recheck} | "
        f"Issues {view.summary.issues} |"
    )

    if width < 60:
        lines.extend(_render_cards(view))
    else:
        lines.extend(_render_table(view, width))

    if view.issues and not compact:
        lines.append(_rule("Active issues", width))
        for issue in view.issues:
            lines.append(f"| ! {issue.message}")
    if view.suggested_actions:
        lines.append(_rule("Next", width))
        lines.append("| " + " | ".join(view.suggested_actions))
    lines.append(_rule("", width))
    return "\n".join(lines) + "\n"


def _rule(title: str, width: int, *, fill: str = "-") -> str:
    inner = width - 2
    if not title:
        return "+" + (fill * inner)[:inner] + "+"
    tw = display_width(title)
    dashes = max(2, inner - tw - 2)
    left = dashes // 2
    right = dashes - left
    return "+" + (fill * left) + " " + title + " " + (fill * right) + "+"


def _render_table(view: LkbBoardView, width: int) -> list[str]:
    headers = ("ID", "TASK", "OWNER", "BASE", "LKB")
    rows = [
        (r.task_id, r.title, r.owner, r.base_status, r.badge.upper())
        for r in sorted(view.rows, key=lambda r: r.task_id)
    ]
    if not rows:
        return ["(no tasks)"]

    # Column widths: start from headers, grow to content, cap so the
    # total fits within *width*.
    col_widths = [display_width(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], display_width(cell))
    # TASK and LKB columns are the truncation candidates.
    overhead = 2 + 5 * 3  # borders + separators
    while sum(col_widths) + overhead > width and col_widths[1] > 8:
        col_widths[1] -= 1
    while sum(col_widths) + overhead > width and col_widths[4] > 6:
        col_widths[4] -= 1

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    lines = [sep]
    lines.append("| " + " | ".join(_pad(h, w) for h, w in zip(headers, col_widths)) + " |")
    lines.append(sep)
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            text = truncate(str(cell), col_widths[i])
            cells.append(_pad(text, col_widths[i]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append(sep)
    return lines


def _render_cards(view: LkbBoardView) -> list[str]:
    lines: list[str] = []
    for r in sorted(view.rows, key=lambda r: r.task_id):
        lines.append(f"[{r.task_id}] {r.title}")
        lines.append(f"  owner: {r.owner} | base: {r.base_status} | lkb: {r.badge}")
        if r.active_blockers:
            lines.append(f"  blocked by: {', '.join(r.active_blockers)}")
    return lines
