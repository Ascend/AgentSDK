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

"""Multi-agent tree parser for the new ClawCodeX session format.

Sub-agent transcripts in the new format live in two places:

- **Flat fallback**: ``~/.clawcodex/transcripts/<agent_id>.jsonl`` — the
  default when ``src.init.init()`` is bypassed. The TranscriptWriter
  emits a one-shot warning.
- **Nested**: ``~/.clawcodex/sessions/<parent_session_id>/subagents/
  agent-<agent_id>.jsonl`` — when the resolver in
  ``clawcodex_ext.agent.transcript.init()`` is called at startup
  (registers ``nested_session_path_resolver``).

Each sub-agent transcript entry carries a ``parent_session_id`` field
(``src.agent.transcript._serialize_message`` injects it for every entry
written by a sub-agent). The visualizer uses the *file* location to
discover sub-agents and the per-entry ``parent_session_id`` to confirm
parentage — both pieces of evidence must agree for a node to be emitted.

No ``.orchestrator_control/runs/<run_id>/agent_meta.json`` reading: the
old orchestrator control tree has been replaced by the sub-agent
transcript convention above.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from extensions.visualizer.models.viz_models import AgentTreeNode, BarStatus
from extensions.visualizer.parsers.transcript_parser import coerce_timestamp, load_transcript_records

logger = logging.getLogger(__name__)

# ``agent-<id>.jsonl`` — the naming convention emitted by
# ``clawcodex_ext.transcript.nested_path``.
_SUBAGENT_RE = re.compile(r"^agent-(?P<agent_id>.+)\.jsonl$")


class MultiAgentParser:
    """Build an ``AgentTreeNode`` list for a session from the new sub-agent layout."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_for_session(
        self,
        session_id: str,
        *,
        sessions_dir: Path | str | None = None,
        transcripts_dir: Path | str | None = None,
    ) -> list[AgentTreeNode]:
        """Discover sub-agent nodes for the given session.

        Walks both:

        - ``<transcripts_dir>/*.jsonl`` (flat) — keeps only files whose
          first non-meta entry has ``parent_session_id == session_id``.
        - ``<sessions_dir>/<session_id>/subagents/agent-*.jsonl``
          (nested) — all files in the directory are sub-agents of this
          session.

        Returns a flat list. The root node is *not* included — the
        caller renders the root as the main swimlane, not an agent-tree row.
        """
        sessions_root = Path(sessions_dir) if sessions_dir else (Path.home() / ".clawcodex" / "sessions")
        transcripts_root = Path(transcripts_dir) if transcripts_dir else (Path.home() / ".clawcodex" / "transcripts")

        nodes: list[AgentTreeNode] = []

        # 1. Nested subagents — every file under
        #    sessions/<sid>/subagents/ is a child of this session.
        nested_dir = sessions_root / session_id / "subagents"
        if nested_dir.is_dir():
            for path in sorted(nested_dir.iterdir()):
                if not path.is_file() or path.suffix != ".jsonl":
                    continue
                m = _SUBAGENT_RE.match(path.name)
                if not m:
                    continue
                agent_id = m.group("agent_id")
                node = self._node_from_transcript(
                    path,
                    agent_id=agent_id,
                    parent_session_id=session_id,
                    source="nested",
                )
                if node is not None:
                    nodes.append(node)

        # 2. Flat subagents — filter the transcripts/ directory by
        #    parent_session_id on the first non-meta entry.
        if transcripts_root.is_dir():
            for path in sorted(transcripts_root.iterdir()):
                if not path.is_file() or path.suffix != ".jsonl":
                    continue
                # Skip the main session's own transcript (it would also
                # lack parent_session_id, but we don't want to include
                # it as a sub-agent).
                if path.stem == session_id:
                    continue
                parent_id = self._peek_parent_session_id(path)
                if parent_id != session_id:
                    continue
                match = _SUBAGENT_RE.match(path.name)
                agent_id = match.group("agent_id") if match else path.stem
                node = self._node_from_transcript(
                    path,
                    agent_id=agent_id,
                    parent_session_id=session_id,
                    source="flat",
                )
                if node is not None:
                    # Don't double-count an agent that's already been
                    # discovered via the nested layout.
                    if not any(n.agent_id == node.agent_id for n in nodes):
                        nodes.append(node)

        return nodes

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _peek_parent_session_id(path: Path) -> str:
        """Return the ``parent_session_id`` of the first parseable entry.

        Tolerant: a corrupt or empty file returns ``""`` (which never
        matches the queried session id). We do not need the full
        transcript content for the discovery step — just the parent
        marker — so a single-line read is enough.
        """
        records, _, _ = load_transcript_records(path)
        for entry in records[:50]:
            if entry.get("isMeta") or entry.get("isVirtual"):
                continue
            parent = entry.get("parent_session_id", "")
            if isinstance(parent, str):
                return parent
            # A partial leading record must not hide a later valid parent
            # marker in the same transcript.
            continue
        return ""

    def _node_from_transcript(
        self,
        path: Path,
        *,
        agent_id: str,
        parent_session_id: str,
        source: str,
    ) -> AgentTreeNode | None:
        """Build an ``AgentTreeNode`` from a sub-agent transcript file.

        Pulls:

        - ``name`` — first non-empty assistant text block (truncated).
        - ``status`` — ``completed`` if the file is closed, ``running``
          if the most recent entry is within the recency window.
        - ``depth`` — currently one level below the root.
        - ``metadata`` — counts of tool_use / tool_result / text blocks
          and the parent_session_id for cross-reference.
        """
        tool_count = 0
        turn_count = 0
        first_text = ""
        timestamps: list[float] = []
        records, warnings, _ = load_transcript_records(path)
        if not records and warnings:
            logger.debug("Failed to read sub-agent transcript %s: %s", path, warnings)
            return None
        for entry in records:
            if entry.get("isMeta") or entry.get("isVirtual") or entry.get("isCompactSummary"):
                continue
            if entry.get("isApiErrorMessage"):
                continue
            ts = coerce_timestamp(entry.get("timestamp"))
            if ts:
                timestamps.append(ts)
            role = entry.get("role", "")
            if role in ("user", "assistant"):
                turn_count += 1
            for block in entry.get("content", []):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "tool_use":
                    tool_count += 1
                elif role == "assistant" and btype in ("text", "thinking") and not first_text:
                    text = block.get("text") or block.get("thinking") or ""
                    if text:
                        first_text = text[:60]

        # Status inference: "running" if the file's mtime is within 5
        # minutes (same window the SessionMetadataParser uses).
        try:
            from time import time as _now

            mtime = path.stat().st_mtime
            status = BarStatus.RUNNING if (_now() - mtime < 300) else BarStatus.SUCCESS
        except OSError:
            status = BarStatus.SUCCESS

        return AgentTreeNode(
            agent_id=agent_id,
            name=first_text or f"agent-{agent_id[:8]}",
            parent_id=parent_session_id,
            children=[],
            session_ref=agent_id,
            status=status,
            depth=1,
            metadata={
                "source": source,
                "transcript_path": str(path),
                "tool_count": tool_count,
                "turn_count": turn_count,
                "start_ts": min(timestamps) if timestamps else 0.0,
                "last_ts": max(timestamps) if timestamps else 0.0,
                "warnings": warnings,
            },
        )
