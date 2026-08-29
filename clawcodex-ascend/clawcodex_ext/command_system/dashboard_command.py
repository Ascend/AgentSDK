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

"""``/dashboard`` slash command.

Provides a Rich-markup snapshot of every ``DashboardEntry`` currently
visible in the cross-system aggregator. Pure read-only — never
mutates the underlying ``DashboardStore`` or any source.

Command forms:
    /dashboard              # all sources, default ordering
    /dashboard goal         # only the "goal" partition
    /dashboard task         # only the "task" partition
    /dashboard orchestrator # only the "orchestrator" partition
    /dashboard sop          # only the "sop" partition
    /dashboard --status failed   # filter by status (any source)
    /dashboard --id goal:thread-1  # show a single entry detail

Output is a multi-line Rich markup block (icons + per-source
sections) — see ``_format_snapshot`` for the rendering. The
command engine forwards the text via :class:`InteractiveOutcome`;
downstream surfaces (REPL/TUI) interpret the markup. In the REPL,
long snapshots are rendered in a keyboard-scrolled viewer.

Design notes:

  * The store is looked up via :func:`get_default_store`; if no
    store exists (e.g. in a unit test that never called
    :func:`get_default_store`), we instantiate a fresh
    :class:`DashboardStore` so the command still works in
    isolation.
  * If the registry has zero sources we emit a friendly hint
    instead of a blank screen, since the most common
    "no sources" case is "you ran this in a test or CI run with
    the dashboard not wired up".
  * The Rich markup is intentionally conservative — bold + dim +
    colour — so it degrades gracefully when the consumer is a
    plain-text terminal.
  * The command is registered as an :class:`InteractiveCommand`
    so the REPL can route long snapshots through its
    keyboard-scrolled viewer. Headless / non-interactive
    surfaces degrade to a flat text print.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

from extensions.agent_dashboard import (
    DashboardEntry,
    DashboardStore,
    get_default_store,
)
from extensions.capabilities.dashboard_entry import (
    DASHBOARD_STATUS_BLOCKED,
    DASHBOARD_STATUS_COMPLETED,
    DASHBOARD_STATUS_FAILED,
    DASHBOARD_STATUS_IN_PROGRESS,
    DASHBOARD_STATUS_PENDING,
    normalize_source_name,
)

from .types import (
    CommandAvailability,
    CommandContext,
    InteractiveCommand,
    InteractiveOutcome,
    LocalCommandResult,
)

logger = logging.getLogger(__name__)

__all__ = ["DASHBOARD_COMMAND", "DashboardCommand", "dashboard_command_call"]

# when the formatted snapshot exceeds this many lines we mark the
# InteractiveOutcome as scrollable so the REPL enters its keyboard-scrolled
# view. Below the threshold a flat print is friendlier.
_SCROLLABLE_LINE_THRESHOLD = 8


_STATUS_ICON: dict[str, str] = {
    DASHBOARD_STATUS_PENDING: ("◻", "dim"),
    DASHBOARD_STATUS_IN_PROGRESS: ("◼", "cyan"),
    DASHBOARD_STATUS_COMPLETED: ("✓", "green"),
    DASHBOARD_STATUS_FAILED: ("✕", "red"),
    DASHBOARD_STATUS_BLOCKED: ("◆", "yellow"),
}


def _format_entry_line(entry: DashboardEntry, indent: int = 2) -> str:
    """Render a single entry as a Rich-markup line."""
    icon, style = _STATUS_ICON.get(entry.status, ("?", "dim"))
    pad = " " * indent
    title = entry.title or "(untitled)"
    title_markup = f"[{style}]{title}[/{style}]" if style else title
    bits: list[str] = [f"{pad}[{style}]{icon}[/{style}] {title_markup}"]
    bits.append(f"[dim]\\[{entry.id}][/dim]")
    if entry.progress_pct is not None:
        try:
            pct = float(entry.progress_pct) * 100.0
            bits.append(f"[dim]({pct:.0f}%)[/dim]")
        except (TypeError, ValueError):
            pass  # Invalid candidate; continue with the surrounding fallback.
    if entry.owner:
        bits.append(f"[dim] @{entry.owner}[/dim]")
    line = " ".join(bits)
    if entry.detail:
        line += f"\n{pad}[dim]{entry.detail}[/dim]"
    return line


def _format_section(
    source: str,
    entries: list[DashboardEntry],
) -> str:
    """Render a single source's section.

    The section header shows source + counts by status. Entries
    are sorted by ``order`` first, then by ``id`` for
    determinism, so re-running the command with the same data
    produces a stable layout.
    """
    if not entries:
        return f"[bold][info]{source}[/info][/bold] [dim](no entries)[/dim]"
    by_status: dict[str, int] = {}
    for e in entries:
        by_status[e.status] = by_status.get(e.status, 0) + 1
    summary_bits: list[str] = []
    for status in (
        DASHBOARD_STATUS_IN_PROGRESS,
        DASHBOARD_STATUS_PENDING,
        DASHBOARD_STATUS_COMPLETED,
        DASHBOARD_STATUS_FAILED,
        DASHBOARD_STATUS_BLOCKED,
    ):
        n = by_status.get(status, 0)
        if n:
            icon, _ = _STATUS_ICON.get(status, ("?", "dim"))
            summary_bits.append(f"{icon} {n}")
    summary = "  ".join(summary_bits) if summary_bits else "0"
    header = f"[bold][info]■ {source}[/info][/bold] [dim]([bold]{len(entries)}[/bold] entries · {summary})[/dim]"
    sorted_entries = sorted(
        entries,
        key=lambda e: (e.order, e.id),
    )
    body = "\n".join(_format_entry_line(e) for e in sorted_entries)
    return f"{header}\n{body}"


def _format_summary_line(by_source: dict[str, list[DashboardEntry]]) -> str:
    """Top-of-output source-stats banner."""
    total = sum(len(v) for v in by_source.values())
    sources = sorted(by_source.keys())
    parts: list[str] = ["[bold]●[/bold] [bold][info]Dashboard[/info][/bold]"]
    parts.append(f"[dim]([bold]{total}[/bold] entries across [bold]{len(sources)}[/bold] sources)[/dim]")
    if sources:
        source_str = ", ".join(sources)
        parts.append(f"[dim]· sources: {source_str}[/dim]")
    return " ".join(parts)


def _filter_entries(
    entries: list[DashboardEntry],
    *,
    source: Optional[str],
    status: Optional[str],
    entry_id: Optional[str],
) -> list[DashboardEntry]:
    src_norm = normalize_source_name(source) if source else None
    out: list[DashboardEntry] = []
    for e in entries:
        if src_norm is not None and e.source != src_norm:
            continue
        if status is not None and e.status != status:
            continue
        if entry_id is not None and e.id != entry_id:
            continue
        out.append(e)
    return out


# --status / --id flag parsing. We keep this dead-simple (no
# argparse) so the command stays import-cheap and so the
# contract is obvious from a single read.
_FLAG_PATTERN = re.compile(r"--(?P<key>status|id|source)\s+(?P<value>\S+)")


def _parse_args(args: str) -> dict[str, str]:
    """Return a dict of ``{key: value}`` for any ``--flag value`` pairs.

    Unknown flags are ignored. The remaining positional text is
    treated as a source filter (e.g. ``/dashboard goal``) for
    ergonomics — without that shortcut users would have to type
    ``/dashboard --source goal`` every time.
    """
    flags: dict[str, str] = {}
    for m in _FLAG_PATTERN.finditer(args or ""):
        flags[m.group("key")] = m.group("value")
    return flags


def _resolve_store(context: CommandContext) -> DashboardStore:
    """Return the active :class:`DashboardStore`.

    Command callers can override the store by setting
    ``context.app_state_store.dashboard_store`` (or the
    ``dashboard_store`` attribute directly). This lets the REPL
    and TUI share a store with a custom registry pre-wired for
    the current session. Default behaviour is to use the
    process-wide store.
    """
    state = getattr(context, "app_state_store", None) or getattr(context, "app_state", None)
    if state is not None:
        store = getattr(state, "dashboard_store", None)
        if isinstance(store, DashboardStore):
            return store
    store = getattr(context, "dashboard_store", None)
    if isinstance(store, DashboardStore):
        return store
    return get_default_store()


def _entries_by_source(entries: Iterable[DashboardEntry]) -> dict[str, list[DashboardEntry]]:
    grouped: dict[str, list[DashboardEntry]] = {}
    for e in entries:
        grouped.setdefault(e.source, []).append(e)
    return grouped


def _format_snapshot(
    entries: list[DashboardEntry],
    *,
    source: Optional[str] = None,
) -> str:
    """Render a complete snapshot block.

    Used by the command and by tests (so the test can
    snapshot-test the output without going through the
    command engine).
    """
    grouped = _entries_by_source(entries)
    if source:
        grouped = {source: grouped.get(source, [])}
    if not any(grouped.values()):
        return "[bold]●[/bold] [bold][info]Dashboard[/info][/bold] [dim](no entries match)[/dim]"
    lines: list[str] = [_format_summary_line(grouped)]
    for src in sorted(grouped.keys()):
        lines.append(_format_section(src, grouped[src]))
    return "\n\n".join(lines)


def dashboard_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """LocalCommand handler. See module docstring for the command forms."""
    raw = (args or "").strip()
    flags = _parse_args(raw)
    # Strip the flag tokens from ``raw`` so the remaining positional
    # text can act as a bare source filter (``/dashboard goal``).
    residual = _FLAG_PATTERN.sub("", raw).strip()
    source: Optional[str] = flags.get("source")
    if source is None and residual:
        # Use the first positional token as the source shortcut.
        first = residual.split()[0]
        if first.lower() in {"all", "summary"}:
            first = None
        if first:
            source = normalize_source_name(first)
    status: Optional[str] = flags.get("status")
    entry_id: Optional[str] = flags.get("id")
    store = _resolve_store(context)
    try:
        merged = store.snapshot()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("/dashboard: store.snapshot() raised: %s", exc)
        merged = []
    entries = _filter_entries(
        merged,
        source=source,
        status=status,
        entry_id=entry_id,
    )
    if entry_id and not entries:
        return LocalCommandResult(
            type="text",
            value=(f"[yellow]No dashboard entry with id {entry_id!r}.[/yellow]"),
        )
    text = _format_snapshot(entries, source=source)
    if not store.registry.names():
        text = (
            f"{text}\n\n"
            "[dim]No data sources are registered. The dashboard will "
            "appear automatically once GoalService / ToolContext.tasks "
            "are wired into the active session.[/dim]"
        )
    return LocalCommandResult(type="text", value=text)


def _should_render_scrollable(message: str) -> bool:
    """Decide whether the dashboard snapshot should be rendered scrollable.

    Conservative heuristic: if the formatted body exceeds the threshold line
    count, suggest scrollable mode. The REPL re-checks against the live
    terminal height, so this is just the suggestion.
    """
    if not message:
        return False
    body = message.strip()
    line_count = body.count("\n") + 1 if body else 0
    return line_count > _SCROLLABLE_LINE_THRESHOLD


class DashboardCommand(InteractiveCommand):
    """``/dashboard`` — cross-system task progress dashboard (read-only).

    Implemented as an :class:`InteractiveCommand` so the REPL can route long
    snapshots through its keyboard-scrolled viewer. On headless or
    non-interactive surfaces the command degrades to a flat text print; it
    never calls ``ctx.ui``.
    """

    async def run(self, args: str, context: CommandContext) -> InteractiveOutcome:
        result = dashboard_command_call(args, context)
        if result.type == "skip":
            return InteractiveOutcome.skip()
        return InteractiveOutcome(
            message=result.value,
            display=result.display_text or "system",
            scrollable=_should_render_scrollable(result.value),
        )


DASHBOARD_COMMAND: DashboardCommand = DashboardCommand(
    name="dashboard",
    description="Show the cross-system task progress dashboard.",
    argument_hint="[goal|task|orchestrator|sop] [--status S] [--id ID]",
    aliases=["dash"],
    availability=CommandAvailability.CONSOLE,
)
