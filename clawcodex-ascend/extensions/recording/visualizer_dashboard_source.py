"""Visualizer dashboard adapter for asciicast recording.

Originally lived at ``extensions/visualizer/asciicast_dashboard_source.py``
as part of the visualizer package. It moved here because its
real consumer is :func:`extensions.recording._factories._visualizer_factory`,
not the visualizer Web UI — the recording CLI polls snapshots on a 1 Hz
tick and renders them as ASCII panels via :func:`extensions.visualizer._rendering.panel`.

This module now lives in the recording package and depends *only* on
the visualizer package's primitives (Protocol definitions in
:mod:`extensions.visualizer.protocols` and the ASCII panel helper in
:mod:`extensions.visualizer._rendering`). The dependency direction is
intentional: ``recording → visualizer`` for rendering primitives, never
the reverse.

The adapter is registered into the
:class:`extensions.agent_dashboard.source_registry.DashboardSourceRegistry`
under the name ``"visualizer_asciicast"`` and is independent from the
live HTML visualizer — it pulls from the same DashboardSource
implementations but produces text instead of HTML.
"""

from __future__ import annotations

import logging
from typing import Any

from extensions.visualizer._rendering import panel
from extensions.visualizer.protocols.dashboard import (
    DASHBOARD_STATUSES,
    DashboardEntry,
)
from extensions.visualizer.protocols.recorder import (
    AsciicastCapture,
    AsciicastEvent,
)

logger = logging.getLogger(__name__)


_STATUS_BADGE = {
    "pending": "⏳ pending",
    "in_progress": "🔵 running",
    "completed": "✅ done",
    "failed": "❌ failed",
    "blocked": "🚧 blocked",
}


class AsciicastDashboardSource:
    """A :class:`DashboardSource` that renders snapshots to ASCII panels.

    The adapter is registered into the
    :class:`extensions.agent_dashboard.source_registry.DashboardSourceRegistry`
    under the name ``"visualizer_asciicast"`` and is independent from the
    live HTML visualizer — it pulls from the same DashboardSource
    implementations but produces text instead of HTML.
    """

    source_name = "visualizer_asciicast"

    def __init__(self) -> None:
        # The adapter wraps the underlying DashboardSource list at
        # record time — there's no need to hold references at
        # construction time, which keeps this class trivial to test.
        pass

    def pull(self, **filters: Any) -> list[DashboardEntry]:
        """Return an empty snapshot.

        The asciicast adapter is a recording-only view of the
        dashboard; it doesn't participate in the normal pull path that
        powers the Web UI / TUI dashboard command. ``pull`` is still
        implemented (per the :class:`DashboardSource` Protocol) so the
        adapter registers cleanly into the source registry.
        """
        return []

    @property
    def cache_ttl_ms(self) -> int:
        # 1 second — same cadence the recording tick uses. The
        # recording CLI ignores this value (it calls record_snapshot
        # directly) but a downstream consumer reading via
        # DashboardStore would see a reasonable cache lifetime.
        return 1000

    # -- recording-specific surface ------------------------------------

    def record_snapshot(
        self,
        capture: AsciicastCapture,
        entries: list[DashboardEntry],
        *,
        title: str = "Dashboard",
    ) -> None:
        """Render ``entries`` into the capture as ASCII panel frames.

        One ``marker`` per status group plus one ``o`` frame per panel.
        Designed to be called from a 1 Hz tick inside the recording
        CLI.
        """
        try:
            capture.marker("dashboard:snapshot", text=f"{title} snapshot")
            for frame in _render_panels(entries, title=title):
                capture.emit(AsciicastEvent(t=0.0, kind="o", data=frame + "\n"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("record_snapshot failed: %s", exc)


def _render_panels(
    entries: list[DashboardEntry],
    *,
    title: str,
    width: int = 80,
) -> list[str]:
    """Group entries by status and render one panel per group.

    Mirrors the HTML panel layout in
    ``extensions/visualizer/templates/orchestrator_dashboard.html``
    (stat-card grid + issue table) using ``─`` rules and indented rows.
    """
    if not entries:
        return [panel(title, ["(no entries)"], width=width)]

    by_status: dict[str, list[DashboardEntry]] = {s: [] for s in _STATUS_BADGE}
    for entry in entries:
        status = entry.status if entry.status in DASHBOARD_STATUSES else "pending"
        by_status.setdefault(status, []).append(entry)

    panels: list[str] = []

    # Summary stat-card row
    stats_line = "  " + "  ".join(
        f"{label}: {len(by_status.get(status, []))}" for status, label in _STATUS_BADGE.items()
    )
    panels.append(panel(title, [stats_line], width=width))

    # Per-status detail rows (active statuses only)
    for status, label in _STATUS_BADGE.items():
        group = by_status.get(status, [])
        if not group:
            continue
        rows = [f"  {label}  ({len(group)})"]
        for entry in group[:8]:  # cap per-panel rows for legibility
            title_col = entry.title[:48] if entry.title else "(untitled)"
            detail = f"  • {entry.id:<20} {title_col:<48} {_pct(entry.progress_pct)}"
            rows.append(detail)
        if len(group) > 8:
            rows.append(f"  … and {len(group) - 8} more")
        panels.append(panel(f"{title} — {label}", rows, width=width))

    return panels


def _pct(p: float | None) -> str:
    if p is None:
        return ""
    return f"{int(round(p * 100))}%"


__all__ = ["AsciicastDashboardSource"]
