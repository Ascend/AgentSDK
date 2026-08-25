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

"""Example fixture data for the Visualizer.

The demo session is written in the new ClawCodeX envelope:

* ``metadata.json`` only carries the user-anchored fields the new format
  actually persists — ``session_id``, ``start_time``, ``model``,
  ``cwd``, ``title``, ``agent_name``, ``tags``. Operational counters
  (``turn_count`` / ``tool_count`` / ``duration_ms`` / ``status`` /
  ``provider``) are no longer cached on metadata; the parsers derive
  them by walking ``transcript.jsonl``.
* ``transcript.jsonl`` uses the full new envelope: each line carries
  ``role``, ``content`` (a list of typed content blocks), ``type``,
  ``uuid``, ``timestamp`` (ISO 8601), ``isMeta``, ``isVirtual``,
  ``isCompactSummary``, ``origin``, and (for assistant) ``model`` /
  ``usage`` / ``stop_reason``. No legacy ``tool_calls`` / ``tool_call_id``
  envelope remains.
* ``events.ndjson`` keeps the tool-event audit-log shape unchanged.
"""

import json
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent


def _iso(ts: float) -> str:
    """Format a Unix timestamp as ISO 8601 UTC (no microseconds)."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat()


def _new_uuid() -> str:
    return uuid.uuid4().hex


def create_demo_session(output_dir: Path | None = None) -> str:
    """Create a demo session with sample data for testing the visualizer.

    When no output directory is provided, use an isolated temporary root
    instead of writing into the user's real session directory.

    Returns the session_id of the created session.
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="clawcodex-visualizer-demo-"))

    session_id = f"demo-{int(time.time())}"
    session_dir = output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    now = time.time()

    # metadata.json — minimal user-anchored fields only. Operational
    # counters (turn_count / tool_count / status / duration_ms) are no
    # longer cached here; the parser derives them from transcript.jsonl.
    metadata = {
        "session_id": session_id,
        "title": "Demo Session — Full Pipeline",
        "cwd": str(output_dir),
        "model": "claude-opus-4-7",
        "agent_name": "clawcodex-overview",
        "start_time": now - 120,
        "tags": ["demo", "test"],
    }
    (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # transcript.jsonl — new ClawCodeX envelope. ``content`` is a list of
    # typed blocks, timestamps are ISO 8601, assistant entries carry
    # model / usage / stop_reason, and ``isMeta`` / ``isVirtual`` /
    # ``isCompactSummary`` / ``origin`` are filled in.
    user_uuid = _new_uuid()
    assistant_read_uuid = _new_uuid()
    tool_read_uuid = _new_uuid()
    assistant_bash_uuid = _new_uuid()
    tool_bash_uuid = _new_uuid()
    assistant_grep_uuid = _new_uuid()
    tool_grep_uuid = _new_uuid()
    assistant_final_uuid = _new_uuid()

    transcript_lines = [
        # ---- user prompt
        json.dumps(
            {
                "role": "user",
                "type": "message",
                "uuid": user_uuid,
                "timestamp": _iso(now - 120),
                "isMeta": False,
                "isVirtual": False,
                "isCompactSummary": False,
                "origin": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze the codebase and suggest improvements",
                    },
                ],
            }
        ),
        # ---- assistant: Read src/main.py
        json.dumps(
            {
                "role": "assistant",
                "type": "message",
                "uuid": assistant_read_uuid,
                "timestamp": _iso(now - 118),
                "isMeta": False,
                "isVirtual": False,
                "isCompactSummary": False,
                "origin": "agent",
                "model": "claude-opus-4-7",
                "stop_reason": "tool_use",
                "duration_ms": 1100,
                "usage": {
                    "input_tokens": 320,
                    "output_tokens": 80,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                "content": [
                    {"type": "text", "text": "I'll start by reading the main files..."},
                    {
                        "type": "tool_use",
                        "id": "tc-001",
                        "tool_use_id": "tc-001",
                        "name": "Read",
                        "input": {"file_path": "src/main.py"},
                    },
                ],
            }
        ),
        # ---- tool_result for tc-001 (carried as a user message with
        # ``tool_result`` block in the new envelope)
        json.dumps(
            {
                "role": "user",
                "type": "message",
                "uuid": tool_read_uuid,
                "timestamp": _iso(now - 117),
                "isMeta": False,
                "isVirtual": False,
                "isCompactSummary": False,
                "origin": "tool_result",
                "toolUseID": "tc-001",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tc-001",
                        "content": [{"type": "text", "text": "# main.py\nimport sys\n..."}],
                    },
                ],
            }
        ),
        # ---- assistant: Bash
        json.dumps(
            {
                "role": "assistant",
                "type": "message",
                "uuid": assistant_bash_uuid,
                "timestamp": _iso(now - 115),
                "isMeta": False,
                "isVirtual": False,
                "isCompactSummary": False,
                "origin": "agent",
                "model": "claude-opus-4-7",
                "stop_reason": "tool_use",
                "duration_ms": 950,
                "usage": {
                    "input_tokens": 410,
                    "output_tokens": 60,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                "content": [
                    {"type": "text", "text": "Now let me check the test coverage..."},
                    {
                        "type": "tool_use",
                        "id": "tc-002",
                        "tool_use_id": "tc-002",
                        "name": "Bash",
                        "input": {"command": "pytest --cov"},
                    },
                ],
            }
        ),
        # ---- tool_result for tc-002
        json.dumps(
            {
                "role": "user",
                "type": "message",
                "uuid": tool_bash_uuid,
                "timestamp": _iso(now - 110),
                "isMeta": False,
                "isVirtual": False,
                "isCompactSummary": False,
                "origin": "tool_result",
                "toolUseID": "tc-002",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tc-002",
                        "content": [{"type": "text", "text": "Coverage: 67% - needs improvement"}],
                    },
                ],
            }
        ),
        # ---- assistant: Grep
        json.dumps(
            {
                "role": "assistant",
                "type": "message",
                "uuid": assistant_grep_uuid,
                "timestamp": _iso(now - 108),
                "isMeta": False,
                "isVirtual": False,
                "isCompactSummary": False,
                "origin": "agent",
                "model": "claude-opus-4-7",
                "stop_reason": "tool_use",
                "duration_ms": 850,
                "usage": {
                    "input_tokens": 380,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                "content": [
                    {"type": "text", "text": "Let me also search for potential issues..."},
                    {
                        "type": "tool_use",
                        "id": "tc-003",
                        "tool_use_id": "tc-003",
                        "name": "Grep",
                        "input": {"pattern": "TODO|FIXME|HACK", "path": "src"},
                    },
                ],
            }
        ),
        # ---- tool_result for tc-003
        json.dumps(
            {
                "role": "user",
                "type": "message",
                "uuid": tool_grep_uuid,
                "timestamp": _iso(now - 105),
                "isMeta": False,
                "isVirtual": False,
                "isCompactSummary": False,
                "origin": "tool_result",
                "toolUseID": "tc-003",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tc-003",
                        "content": [{"type": "text", "text": "Found 5 TODOs and 2 FIXMEs"}],
                    },
                ],
            }
        ),
        # ---- assistant final summary
        json.dumps(
            {
                "role": "assistant",
                "type": "message",
                "uuid": assistant_final_uuid,
                "timestamp": _iso(now - 100),
                "isMeta": False,
                "isVirtual": False,
                "isCompactSummary": False,
                "origin": "agent",
                "model": "claude-opus-4-7",
                "stop_reason": "end_turn",
                "duration_ms": 1700,
                "usage": {
                    "input_tokens": 720,
                    "output_tokens": 240,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                "content": [
                    {
                        "type": "text",
                        "text": "Based on my analysis, here are the key improvements...",
                    },
                ],
            }
        ),
    ]
    (session_dir / "transcript.jsonl").write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")

    # events.ndjson — tool-event audit log (unchanged shape)
    events = [
        {
            "event": "tool_call",
            "tool_name": "Read",
            "tool_call_id": "tc-001",
            "timestamp": now - 118,
            "approved": True,
            "turn": 1,
            "params": {"file_path": "src/main.py"},
        },
        {
            "event": "tool_result",
            "tool_call_id": "tc-001",
            "timestamp": now - 117,
            "duration_ms": 800,
        },
        {
            "event": "tool_call",
            "tool_name": "Bash",
            "tool_call_id": "tc-002",
            "timestamp": now - 115,
            "approved": True,
            "turn": 2,
            "params": {"command": "pytest --cov"},
        },
        {
            "event": "tool_result",
            "tool_call_id": "tc-002",
            "timestamp": now - 110,
            "duration_ms": 5000,
        },
        {
            "event": "tool_call",
            "tool_name": "Grep",
            "tool_call_id": "tc-003",
            "timestamp": now - 108,
            "approved": True,
            "turn": 3,
            "params": {"pattern": "TODO|FIXME|HACK"},
        },
        {
            "event": "tool_result",
            "tool_call_id": "tc-003",
            "timestamp": now - 105,
            "duration_ms": 2500,
        },
    ]
    events_text = "\n".join(json.dumps(e) for e in events) + "\n"
    (session_dir / "events.ndjson").write_text(events_text, encoding="utf-8")

    return session_id
