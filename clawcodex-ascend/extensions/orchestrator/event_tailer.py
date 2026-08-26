#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""EventTailerManager - manages per-run_id file tailer threads.
Starts a background tailer thread for each active agent session (identified by
run_id in the IssueRegistry), tailing two files:
  1. events.ndjson - per tool-call events (including params)
  2. transcript.jsonl - tool_use/text blocks of assistant messages + tool_result blocks of user messages
Parsed events are pushed to a thread-safe queue consumed by
DashboardState.refresh_snapshot().
File tailing uses byte-offset polling (seek/readline/tell), matching the
Visualizer SessionLiveTail pattern. This design accepts duplication of that
pattern (4 independent implementations already exist) and does not extract a
shared class, keeping the Dashboard self-contained.
Design constraints:
- stdlib only (threading, queue, json, pathlib, time, logging)
- no imports of extensions.visualizer (zero coupling)
- compatible with the synchronous thread model of ThreadingHTTPServer
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_QUEUE_SIZE = 1000
_TAIL_POLL_INTERVAL_S = 0.5
_MAX_RESULT_CONTENT_CHARS = 500


def _truncate_content(raw: Any, max_chars: int = _MAX_RESULT_CONTENT_CHARS) -> str:
    """Flatten tool_result content to a truncated string.

    Content can be a string, a list of ``{type:"text", text:"..."}`` blocks,
    or other JSON.  Returns at most ``max_chars`` characters.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        text = "\n".join(parts)
    else:
        text = json.dumps(raw, ensure_ascii=False)
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


class EventTailerManager:
    """Manages per-run_id file tailer threads, emitting events to a thread-safe queue."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._tailers: dict[str, _SessionTailer] = {}
        self._event_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=_MAX_QUEUE_SIZE)
        self._lock = threading.Lock()

    def sync_active_run_ids(self, run_id_to_issue_info: dict[str, tuple[str, Path]]) -> None:
        """Start tailers for new run_ids, stop ones that have disappeared.

        Args:
            run_id_to_issue_info: mapping of active run_id -> (issue_id, workspace_path),
                sourced from IssueRegistry records with active status and a run_id.
        """
        with self._lock:
            # Stop tailers that are no longer active
            for run_id in list(self._tailers):
                if run_id not in run_id_to_issue_info:
                    self._tailers[run_id].stop()
                    del self._tailers[run_id]
            # Start new tailers
            for run_id, (issue_id, issue_workspace) in run_id_to_issue_info.items():
                if run_id and run_id not in self._tailers:
                    tailer = _SessionTailer(
                        run_id=run_id,
                        issue_id=issue_id,
                        workspace=issue_workspace,
                        event_queue=self._event_queue,
                    )
                    self._tailers[run_id] = tailer
                    tailer.start()

    def drain_events(self) -> list[dict[str, Any]]:
        """Consume all pending events from the queue without blocking."""
        events: list[dict[str, Any]] = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def load_historical(self, run_id: str, issue_id: str, workspace: Path) -> None:
        """One-shot read of all historical events for a completed session.

        Creates a temporary ``_SessionTailer`` (no thread), calls its read
        methods once to drain ``events.ndjson`` and ``transcript.jsonl`` from
        offset 0, pushing events into the shared queue.  Does NOT start a
        persistent tailer — completed sessions won't produce new events.
        """
        tailer = _SessionTailer(
            run_id=run_id,
            issue_id=issue_id,
            workspace=workspace,
            event_queue=self._event_queue,
        )
        tailer._tail_events_ndjson()
        tailer._tail_transcript()

    def stop_all(self) -> None:
        """Stop all tailers (called at atexit)."""
        with self._lock:
            for tailer in self._tailers.values():
                tailer.stop()
            self._tailers.clear()


class _SessionTailer:
    """File tailer for a single session, running on its own thread."""

    def __init__(
        self,
        run_id: str,
        issue_id: str,
        workspace: Path,
        event_queue: queue.Queue[dict[str, Any]],
    ) -> None:
        self._run_id = run_id
        self._issue_id = issue_id
        self._workspace = workspace
        self._event_queue = event_queue
        self._running = False
        self._thread: threading.Thread | None = None

        # File path (two-level fallback)
        self._events_ndjson_path = workspace / ".reports" / f"{run_id}.events.ndjson"
        self._events_ndjson_fallback = Path.home() / ".clawcodex" / "tool-events" / run_id / "events.ndjson"
        self._transcript_path = Path.home() / ".clawcodex" / "sessions" / run_id / "transcript.jsonl"

        # Byte offsets
        self._events_offset = 0
        self._transcript_offset = 0
        # True once events.ndjson yields any data — prevents double-counting
        # tool_call events from transcript.jsonl's tool_use blocks.
        self._events_ndjson_seen = False
        # tool_use_id → tool_name mapping, so tool_result events can show
        # the human-readable tool name instead of the opaque ID.
        self._tool_name_map: dict[str, str] = {}

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._tail_loop,
            name=f"tailer-{self._run_id[:12]}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _tail_loop(self) -> None:
        while self._running:
            try:
                self._tail_events_ndjson()
                self._tail_transcript()
            except Exception as exc:
                logger.debug("Tailer %s error: %s", self._run_id, exc)
            time.sleep(_TAIL_POLL_INTERVAL_S)

    def _tail_events_ndjson(self) -> None:
        """Tail events.ndjson - per tool-call events (including params, approved, etc.)."""
        path = self._resolve_events_path()
        if path is None:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.seek(self._events_offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._events_ndjson_seen = True
                    self._emit_event(
                        "tool_call",
                        {
                            "tool": entry.get("tool", "?"),
                            "approved": entry.get("approved"),
                            "turn": entry.get("turn", 0),
                            "deny_reason": entry.get("deny_reason"),
                            "params": entry.get("params"),
                            "ts": entry.get("ts", ""),
                        },
                    )
                self._events_offset = f.tell()
        except FileNotFoundError:
            pass

    def _tail_transcript(self) -> None:
        """Tail transcript.jsonl — extract tool events.

        Parses two kinds of message lines:
        1. Assistant messages (``role == "assistant"``) — ``tool_use`` content blocks → tool_call events
        2. User messages (``role == "user"``) — ``tool_result`` content blocks → tool_result events

        This makes the dashboard work even when ``audit_log=minimal`` (events.ndjson only
        records denied calls) because transcript.jsonl is always written per-turn.
        """
        if not self._transcript_path.exists():
            return
        try:
            with open(self._transcript_path, "r", encoding="utf-8") as f:
                f.seek(self._transcript_offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    role = entry.get("role")

                    if role == "assistant":
                        self._process_assistant_message(entry)
                    elif role == "user":
                        self._process_user_message(entry)

                self._transcript_offset = f.tell()
        except FileNotFoundError:
            pass

    def _process_assistant_message(self, entry: dict[str, Any]) -> None:
        """Extract tool_use + text events from an assistant message.

        If events.ndjson already provided tool_call events (richer data with
        approved/deny_reason), skip tool_use blocks here to avoid double-counting.
        Always emit text blocks (events.ndjson doesn't contain agent text).
        Always register tool_use_id → tool_name for tool_result lookups.
        """
        content = entry.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "tool_use":
                    tool_name = block.get("name", "?")
                    tool_use_id = block.get("id")
                    if tool_use_id:
                        self._tool_name_map[tool_use_id] = tool_name
                    # Only emit tool_call from transcript if events.ndjson
                    # hasn't already provided it (audit_log=minimal/none case)
                    if not self._events_ndjson_seen:
                        self._emit_event(
                            "tool_call",
                            {
                                "tool": tool_name,
                                "approved": True,
                                "turn": 0,
                                "deny_reason": None,
                                "tool_use_id": tool_use_id,
                                "params": block.get("input"),
                            },
                        )
                elif block_type == "text":
                    text = block.get("text", "")
                    if text and text.strip():
                        self._emit_event(
                            "agent_text",
                            {"content": _truncate_content(text, 500)},
                        )

    def _process_user_message(self, entry: dict[str, Any]) -> None:
        """Extract tool_result events from a user message's content blocks."""
        content = entry.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    is_error = bool(block.get("is_error"))
                    tool_use_id = block.get("tool_use_id")
                    # Look up the tool name from the corresponding tool_use block
                    tool_name = self._tool_name_map.get(tool_use_id or "", tool_use_id or "?")
                    raw_content = block.get("content")
                    # Truncate content to avoid queue bloat (keep first 500 chars)
                    result_content = _truncate_content(raw_content, 500)
                    self._emit_event(
                        "tool_result",
                        {
                            "tool": tool_name,
                            "is_error": is_error,
                            "tool_use_id": tool_use_id,
                            "result_content": result_content,
                        },
                    )

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Push a normalized event dict onto the queue."""
        try:
            self._event_queue.put_nowait(
                {
                    "type": "event",
                    "event_type": event_type,
                    "issue_id": self._issue_id,
                    "run_id": self._run_id,
                    "data": data,
                    "ts": time.time(),
                }
            )
        except queue.Full:
            pass  # Queue full — skip this event, offset still advances

    def _resolve_events_path(self) -> Path | None:
        """Return the existing events.ndjson path (two-level fallback)."""
        if self._events_ndjson_path.exists():
            return self._events_ndjson_path
        if self._events_ndjson_fallback.exists():
            return self._events_ndjson_fallback
        return None
