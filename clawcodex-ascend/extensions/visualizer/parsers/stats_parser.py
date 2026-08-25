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

"""Parse Visualizer tool statistics from tool_stats.jsonl.

Reads ``~/.clawcodex/tool_stats.jsonl`` written by
``clawcodex_ext.tool_stats`` and returns aggregated summaries
suitable for display in the Visualizer dashboard.

Usage::

    parser = StatsFileParser()
    summary = parser.get_summary()      # global summary
    tools = parser.get_summary(kind="tool")
    top10 = parser.get_recent(limit=10)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_STATS_PATH = Path.home() / ".clawcodex" / "tool_stats.jsonl"


class StatsFileParser:
    """Read and aggregate tool_stats.jsonl records."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else DEFAULT_STATS_PATH

    def get_summary(
        self,
        kind: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate summary across all records.

        Returns::

            {
                "total_calls": int,
                "by_kind": {"tool": N, "skill": N},
                "by_name": {name: count, ...},        # sorted desc
                "by_name_ok": {name: ok_count, ...},  # sorted desc
                "avg_duration_ms": float,
                "error_rate": float,
            }
        """
        rows = self._load(kind=kind, agent_id=agent_id)
        if not rows:
            return {
                "total_calls": 0,
                "by_kind": {},
                "by_name": {},
                "by_name_ok": {},
                "avg_duration_ms": 0.0,
                "error_rate": 0.0,
            }

        total = len(rows)
        by_kind: dict[str, int] = {}
        by_name: dict[str, int] = {}
        by_name_ok: dict[str, int] = {}
        total_dur = 0.0
        errors = 0

        for r in rows:
            k = r.get("kind", "?")
            by_kind[k] = by_kind.get(k, 0) + 1
            name = r.get("tool") or r.get("skill") or "unknown"
            by_name[name] = by_name.get(name, 0) + 1
            if r.get("ok"):
                by_name_ok[name] = by_name_ok.get(name, 0) + 1
            else:
                errors += 1
            total_dur += r.get("dur_ms", 0.0)

        return {
            "total_calls": total,
            "by_kind": dict(sorted(by_kind.items(), key=lambda x: -x[1])),
            "by_name": dict(sorted(by_name.items(), key=lambda x: -x[1])),
            "by_name_ok": dict(sorted(by_name_ok.items(), key=lambda x: -x[1])),
            "avg_duration_ms": round(total_dur / total, 1) if total else 0.0,
            "error_rate": round(errors / total, 3) if total else 0.0,
        }

    def get_recent(
        self,
        limit: int = 20,
        kind: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most recent N records."""
        rows = self._load(kind=kind, agent_id=agent_id)
        rows.sort(key=lambda r: r.get("ts", 0.0), reverse=True)
        return rows[:limit]

    def _load(
        self,
        kind: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self._path.exists() or self._path.stat().st_size == 0:
            return []
        records: list[dict[str, Any]] = []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if kind is not None and row.get("kind") != kind:
                        continue
                    if agent_id is not None and row.get("agent_id") != agent_id:
                        continue
                    records.append(row)
        except OSError as e:
            logger.warning("cannot read %s: %s", self._path, e)
        return records
