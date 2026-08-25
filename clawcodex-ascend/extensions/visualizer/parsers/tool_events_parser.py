#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""Parse tool-event audit NDJSON into timeline bars.

Parses ``events.ndjson`` files written by ``AgentRunner._append_tool_event_log``.
In the new format the canonical location is::

    ~/.clawcodex/tool-events/<run_id>/events.ndjson

This on-disk shape is preserved for compatibility. The parser is
path-agnostic — the caller hands it an explicit ``path`` — so the
default location does not constrain callers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from extensions.visualizer.models.viz_models import BarStatus, BarType, TimelineBar

logger = logging.getLogger(__name__)

# Default tool-event audit-log root.
DEFAULT_TOOL_EVENTS_DIR = Path.home() / ".clawcodex" / "tool-events"


class ToolEventsParser:
    """Parse tool-event audit logs into TimelineBars."""

    def __init__(self) -> None:
        self._bar_counter = 0
        # Lazy import: ``builders/__init__.py`` imports ``TimelineBuilder``,
        # which in turn imports this module. Importing OperationCategorizer
        # at module level would therefore re-enter this module mid-init
        # and crash with a circular-import error. Defer until first use.
        from extensions.visualizer.builders.operation_categorizer import OperationCategorizer

        self._categorizer = OperationCategorizer()

    def parse_file(self, path: Path | str) -> list[TimelineBar]:
        """Parse an entire events.ndjson file.

        Two-pass: ``agent_result`` rows (written when an Agent spawn
        RETURNS, carrying the child's ``agent_id``) are collected first
        and joined onto their call rows via ``tool_use_id`` — the spawn
        bars then carry an exact ``agent_id`` and the tree layout can
        attribute connectors precisely instead of guessing by time.
        An orphan ``agent_result`` whose call row was lost is
        synthesized into a spawn bar so a single dropped write no
        longer shifts every later pairing by one.
        """
        path = Path(path)
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Pass 1: index agent_result rows by tool_use_id.
        agent_id_by_use: dict[str, str] = {}
        for entry in entries:
            if entry.get("kind") == "agent_result" and entry.get("tool_use_id") and entry.get("agent_id"):
                agent_id_by_use[str(entry["tool_use_id"])] = str(entry["agent_id"])

        # Pass 2: build bars from call rows; join agent ids; track which
        # result rows found their call row.
        bars: list[TimelineBar] = []
        joined_use_ids: set[str] = set()
        for entry in entries:
            if entry.get("kind") == "agent_result":
                continue
            timeline_bar = self._entry_to_bar(entry)
            if timeline_bar is None:
                continue
            use_id = entry.get("tool_use_id")
            if entry.get("tool") == "Agent" and use_id and str(use_id) in agent_id_by_use:
                timeline_bar.detail["agent_id"] = agent_id_by_use[str(use_id)]
                joined_use_ids.add(str(use_id))
            bars.append(timeline_bar)

        # Pass 3: synthesize spawn bars for orphan agent_result rows.
        for entry in entries:
            if entry.get("kind") != "agent_result":
                continue
            use_id = str(entry.get("tool_use_id") or "")
            if use_id and use_id in joined_use_ids:
                continue
            timeline_bar = self._entry_to_bar(entry)
            if timeline_bar is None:
                continue
            if entry.get("agent_id"):
                timeline_bar.detail["agent_id"] = str(entry["agent_id"])
            timeline_bar.detail["is_agent_invocation"] = True
            bars.append(timeline_bar)
        return bars

    def _entry_to_bar(self, entry: dict[str, Any]) -> TimelineBar | None:
        ts = entry.get("ts", 0.0)
        tool = entry.get("tool", "unknown")
        approved = entry.get("approved")
        deny_reason = entry.get("deny_reason")
        turn = entry.get("turn", 0)

        self._bar_counter += 1
        status = BarStatus.SUCCESS if approved is True else BarStatus.ERROR if approved is False else BarStatus.WARNING

        timeline_bar = TimelineBar(
            id=f"tev-{self._bar_counter}",
            type=BarType.TOOL_CALL,
            label=tool,
            start_time=ts,
            end_time=ts + 0.01,
            duration_ms=10,
            status=status,
            detail={
                "tool": tool,
                "approved": approved,
                "deny_reason": deny_reason,
                "turn": turn,
                "params": entry.get("params", {}),
            },
        )
        # Agent spawns: mark the bar so the agent-tree layout and
        # ``_apply_spawn_metadata`` recognize it without relying on the
        # label/category fallback, and hoist the task description out of
        # ``params`` so sub-agent lanes get a human-readable name instead
        # of the opaque agent id.
        if tool == "Agent":
            timeline_bar.detail["is_agent_invocation"] = True
            params = entry.get("params") or {}
            if isinstance(params, dict):
                description = params.get("description")
                if isinstance(description, str) and description:
                    timeline_bar.detail["description"] = description
        timeline_bar.category = self._categorizer.categorize(timeline_bar)
        return timeline_bar
