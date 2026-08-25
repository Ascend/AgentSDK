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

"""Map timeline operations to the waterfall legend categories.

Maps a ``TimelineBar`` to one of eight ``OperationCategory`` buckets that
match the reference visualization's top legend:

  READ          🟢 Read / Glob / Grep / WebFetch / WebSearch
  EXECUTE       🔵 Bash / Execute / TaskKill / BashOutput / KillShell
  WRITE         🟡 Write / Edit / MultiEdit / NotebookEdit / TodoWrite
  ORCHESTRATE   🟣 Agent / Task (subagent invocation)
  LLM_TEXT      🔷 LLM_CALL bars (assistant text/thinking spans)
  TURN          🟪 TURN bars (turn boundaries)
  BACKGROUND    ⬜ polling / wait spans flagged isBackground
  OTHER         ⚪ anything else (residual catch-all)

Resolution order:
  1. Explicit ``isAgentInvocation`` / ``is_agent_invocation`` flag in detail  → ORCHESTRATE
  2. ``tool_name`` against per-category rule sets
  3. BarType-based fallback:
        LLM_CALL      → LLM_TEXT
        TURN          → TURN
        PHASE/SESSION → ORCHESTRATE
  4. ``isBackground`` detail flag → BACKGROUND
  5. anything else → OTHER

The categorizer is pure (no I/O), and is safe to invoke from any parser.
"""

from __future__ import annotations

from extensions.visualizer.models.viz_models import BarType, OperationCategory, TimelineBar


class OperationCategorizer:
    """Rule-based mapper from TimelineBar to OperationCategory."""

    _TOOL_RULES: dict[OperationCategory, frozenset[str]] = {
        OperationCategory.READ: frozenset({"Read", "Glob", "Grep", "WebFetch", "WebSearch", "LS"}),
        OperationCategory.EXECUTE: frozenset({"Bash", "Execute", "TaskKill", "BashOutput", "KillShell", "Shell"}),
        OperationCategory.WRITE: frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit", "TodoWrite", "Patch"}),
        OperationCategory.ORCHESTRATE: frozenset({"Agent", "Task", "SendMessage", "TeamCreate"}),
    }

    def categorize(self, timeline_bar: TimelineBar) -> OperationCategory:
        """Return the operation category for the given bar.

        Never returns ``None`` — falls back to ``OTHER`` for unclassifiable bars.
        """
        if timeline_bar.category is not None and timeline_bar.category != OperationCategory.OTHER:
            return timeline_bar.category

        detail = timeline_bar.detail or {}
        # 1. Explicit orchestration flag wins.
        if detail.get("isAgentInvocation") or detail.get("is_agent_invocation"):
            return OperationCategory.ORCHESTRATE

        # 2. Tool name lookup.
        tool_name = (detail.get("tool_name") or detail.get("tool") or timeline_bar.label or "").strip()
        for cat, names in self._TOOL_RULES.items():
            if tool_name.lower() in {name.lower() for name in names}:
                return cat
        lowered = tool_name.lower()
        if any(word in lowered for word in ("read", "search", "grep", "glob", "fetch", "list", "view", "find")):
            return OperationCategory.READ
        if any(word in lowered for word in ("write", "edit", "patch", "create", "update", "delete", "move")):
            return OperationCategory.WRITE
        if any(word in lowered for word in ("bash", "shell", "exec", "run", "command", "terminal")):
            return OperationCategory.EXECUTE
        if any(word in lowered for word in ("task", "agent", "workflow", "spawn", "dispatch", "plan")):
            return OperationCategory.ORCHESTRATE

        # 3. Background flag wins over bar-type fallback.
        if detail.get("isBackground") or detail.get("is_background"):
            return OperationCategory.BACKGROUND

        # 4. BarType-based fallback (split the previous catch-all OTHER).
        if timeline_bar.type == BarType.LLM_CALL:
            return OperationCategory.LLM_TEXT
        if timeline_bar.type == BarType.TURN:
            return OperationCategory.TURN
        if timeline_bar.type == BarType.PHASE:
            return OperationCategory.ORCHESTRATE
        if timeline_bar.type == BarType.SESSION:
            return OperationCategory.ORCHESTRATE

        # 5. Residual catch-all.
        return OperationCategory.OTHER


__all__ = ["OperationCategorizer"]
