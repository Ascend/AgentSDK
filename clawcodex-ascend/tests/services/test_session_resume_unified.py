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

"""F-49 Phase 0.4.6: Recursive resume consistency verification.

Verifies that ``Session.resume()`` (with the Phase 0.4.1 JSONL back-fill)
behaves correctly for orchestrator/cron sessions that only write JSONL
transcripts without a ``.json`` snapshot.

Scenarios covered:
  1. Orchestrator-style session (JSONL + metadata only) → resume → messages non-empty
  2. Resume → Save → Resume again (recursive round-trip) → messages identical
  3. Cron/background-runner session (same as orchestrator)
  4. Cross-scenario: orchestrator writes → CLI resume → REPL appends → resume again
  5. ``.json`` snapshot missing → fallback to JSONL works
  6. Recursive three-round consistency
"""
# pylint: disable=redefined-outer-name

from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

import pytest

from src.agent.session import Session
from src.types.messages import (
    AssistantMessage,
    UserMessage,
    message_to_dict,
)
from src.types.content_blocks import TextBlock, ToolUseBlock, ToolResultBlock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sessions_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate sessions under ``tmp_path`` so tests never touch real ``~/.clawcodex``.

    Patches both ``SESSIONS_DIR`` (used by ``SessionStorage``) and
    ``Path.home()`` (used by ``Session.save()`` to find the sessions dir)
    so all I/O goes to the isolated temp directory.
    """
    clawcodex_dir = tmp_path / ".clawcodex"
    sessions_dir = clawcodex_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "clawcodex_ext.services.session_storage.SESSIONS_DIR",
        sessions_dir,
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return sessions_dir


def _write_orchestrator_session(
    sessions_dir: Path,
    session_id: str,
    *,
    num_turns: int = 3,
) -> Path:
    """Write an orchestrator-style session (JSONL + metadata, no ``.json``).

    Returns the session directory path.
    """
    session_dir = sessions_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # --- metadata.json (minimal, like AgentRunner writes) ---
    metadata = {
        "session_id": session_id,
        "model": "claude-sonnet-4-20250514",
        "title": f"orchestrator-test-{session_id[:8]}",
        "start_time": time.time(),
        "last_updated": time.time(),
        "message_count": num_turns * 2,
        "tags": ["orchestrator"],
    }
    (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # --- transcript.jsonl (Message dicts, one per line) ---
    transcript_path = session_dir / "transcript.jsonl"
    with open(transcript_path, "w", encoding="utf-8") as f:
        for i in range(num_turns):
            # User turn
            user_msg = UserMessage(
                content=[TextBlock(text=f"Turn {i} prompt")],
            )
            f.write(json.dumps(message_to_dict(user_msg)) + "\n")

            # Assistant turn with a tool_use
            asst_msg = AssistantMessage(
                content=[
                    TextBlock(text=f"Turn {i} thinking..."),
                    ToolUseBlock(
                        id=f"tu_{i}",
                        name="Read",
                        input={"path": f"file_{i}.py"},
                    ),
                ],
            )
            f.write(json.dumps(message_to_dict(asst_msg)) + "\n")

            # Tool result (written as a UserMessage with tool_result block)
            result_msg = UserMessage(
                content=[
                    ToolResultBlock(
                        tool_use_id=f"tu_{i}",
                        content=f"Content of file_{i}.py",
                        is_error=False,
                    ),
                ],
            )
            f.write(json.dumps(message_to_dict(result_msg)) + "\n")

    return session_dir


def _messages_equal(
    msgs_a: list,
    msgs_b: list,
) -> bool:
    """Compare two message lists by role + content text (ignore UUIDs/timestamps)."""
    if len(msgs_a) != len(msgs_b):
        return False
    for a, b in zip(msgs_a, msgs_b):
        a_role = getattr(a, "role", "")
        b_role = getattr(b, "role", "")
        if a_role != b_role:
            return False
        a_text = str(getattr(a, "content", ""))
        b_text = str(getattr(b, "content", ""))
        # Compare serialised representation (ignores non-materialised fields)
        if a_text != b_text:
            return False
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResumeUnified:
    """F-49 Phase 0.4.6: Recursive resume consistency."""

    def test_orchestrator_resume_loads_messages(
        self,
        mock_sessions_dir: Path,
    ) -> None:
        """Scenario 1: Orchestrator session (JSONL only) → resume → messages non-empty."""
        session_id = f"orch-{uuid4().hex[:12]}"
        _write_orchestrator_session(mock_sessions_dir, session_id, num_turns=2)

        loaded = Session.resume(session_id)
        assert loaded is not None
        # Phase 0.4.1: should have loaded messages from JSONL
        assert len(loaded.conversation.messages) > 0
        # 2 turns × (user + assistant + tool_result) = 6 messages
        assert len(loaded.conversation.messages) == 6

    def test_orchestrator_resume_save_resume_roundtrip(
        self,
        mock_sessions_dir: Path,
    ) -> None:
        """Scenario 2: Resume → Save → Resume again → messages identical."""
        session_id = f"roundtrip-{uuid4().hex[:12]}"
        _write_orchestrator_session(mock_sessions_dir, session_id, num_turns=3)

        # First resume (from JSONL via Phase 0.4.1)
        first = Session.resume(session_id)
        assert first is not None
        first_msgs = list(first.conversation.messages)
        assert len(first_msgs) > 0

        # Save → creates .json snapshot
        first.save()

        # Second resume (should now fast-path via .json)
        second = Session.resume(session_id)
        assert second is not None
        second_msgs = list(second.conversation.messages)

        # Messages should be consistent
        assert _messages_equal(first_msgs, second_msgs), (
            f"Messages differ after save→resume round-trip: "
            f"first={len(first_msgs)} msgs, second={len(second_msgs)} msgs"
        )

    def test_recursive_three_round_consistency(
        self,
        mock_sessions_dir: Path,
    ) -> None:
        """Scenario 6: Three recursive resume calls all produce identical messages."""
        session_id = f"recursive-{uuid4().hex[:12]}"
        _write_orchestrator_session(mock_sessions_dir, session_id, num_turns=2)

        results: list[list] = []
        for _round in range(3):
            loaded = Session.resume(session_id)
            assert loaded is not None
            results.append(list(loaded.conversation.messages))
            loaded.save()  # Force .json write after each round

        # All three rounds must produce the same messages
        assert _messages_equal(results[0], results[1])
        assert _messages_equal(results[1], results[2])

    def test_cron_session_resume(
        self,
        mock_sessions_dir: Path,
    ) -> None:
        """Scenario 3: Cron/background-runner session resume."""
        session_id = f"cron-{uuid4().hex[:12]}"
        session_dir = mock_sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Cron-style: metadata with cron tags, no .json
        metadata = {
            "session_id": session_id,
            "model": "claude-sonnet-4-20250514",
            "title": "cron-nightly-build",
            "start_time": time.time(),
            "last_updated": time.time(),
            "message_count": 2,
            "tags": ["cron:task:nightly-build", "cron:run:abc123"],
        }
        (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # Write a minimal transcript
        transcript_path = session_dir / "transcript.jsonl"
        with open(transcript_path, "w", encoding="utf-8") as f:
            user_msg = UserMessage(content=[TextBlock(text="Run nightly build")])
            f.write(json.dumps(message_to_dict(user_msg)) + "\n")
            asst_msg = AssistantMessage(
                content=[TextBlock(text="Build completed successfully")],
            )
            f.write(json.dumps(message_to_dict(asst_msg)) + "\n")

        loaded = Session.resume(session_id)
        assert loaded is not None
        assert len(loaded.conversation.messages) == 2
        assert loaded.conversation.messages[0].role == "user"
        assert loaded.conversation.messages[1].role == "assistant"

    def test_json_snapshot_created_after_save(
        self,
        mock_sessions_dir: Path,
    ) -> None:
        """After resume→save, the transcript carries a trailing
        ``session_snapshot`` line (P5-A unified format).

        F-49 P5-A: ``session.json`` is no longer written. The cost
        snapshot lives in the **last line** of ``transcript.jsonl`` as
        a ``{"type": "session_snapshot", "cost": ...}`` entry, so
        ``cost_restore`` can pick it up via ``tail -1``.
        """
        session_id = f"snapshot-{uuid4().hex[:12]}"
        _write_orchestrator_session(mock_sessions_dir, session_id, num_turns=1)

        # Before save: no session.json
        json_path = mock_sessions_dir / session_id / "session.json"
        assert not json_path.exists()

        loaded = Session.resume(session_id)
        assert loaded is not None
        loaded.save()

        # After save: no session.json (P5-A — full snapshot dropped).
        assert not json_path.exists()
        # The transcript ends with a session_snapshot line.
        transcript_path = mock_sessions_dir / session_id / "transcript.jsonl"
        lines = [line for line in transcript_path.read_text().splitlines() if line.strip()]
        assert lines, "transcript.jsonl is empty after save"
        snapshot = json.loads(lines[-1])
        assert snapshot.get("type") == "session_snapshot"
        assert "cost" in snapshot
        # The first line is session_init (P5-E).
        first = json.loads(lines[0])
        assert first.get("type") == "session_init"
        assert first.get("session_id") == session_id

    def test_missing_json_fallback_to_jsonl(
        self,
        mock_sessions_dir: Path,
    ) -> None:
        """Scenario 5: ``.json`` missing → fallback to JSONL works."""
        session_id = f"fallback-{uuid4().hex[:12]}"
        _write_orchestrator_session(mock_sessions_dir, session_id, num_turns=2)

        # Ensure no .json exists
        json_path = mock_sessions_dir / session_id / "session.json"
        if json_path.exists():
            json_path.unlink()

        loaded = Session.resume(session_id)
        assert loaded is not None
        # Phase 0.4.1 fallback: messages loaded from JSONL
        assert len(loaded.conversation.messages) == 6

    def test_empty_transcript_does_not_crash(
        self,
        mock_sessions_dir: Path,
    ) -> None:
        """Empty transcript → resume succeeds with empty conversation."""
        session_id = f"empty-{uuid4().hex[:12]}"
        session_dir = mock_sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # metadata only, no transcript
        metadata = {
            "session_id": session_id,
            "model": "claude-sonnet-4-20250514",
            "title": "empty-test",
            "start_time": time.time(),
            "last_updated": time.time(),
            "message_count": 0,
            "tags": [],
        }
        (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (session_dir / "transcript.jsonl").write_text("")  # Empty transcript

        loaded = Session.resume(session_id)
        assert loaded is not None
        # Should not crash; messages should be empty
        assert len(loaded.conversation.messages) == 0

    def test_malformed_transcript_line_skipped(
        self,
        mock_sessions_dir: Path,
    ) -> None:
        """Malformed JSONL line → skipped without crashing."""
        session_id = f"malformed-{uuid4().hex[:12]}"
        session_dir = mock_sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "session_id": session_id,
            "model": "claude-sonnet-4-20250514",
            "title": "malformed-test",
            "start_time": time.time(),
            "last_updated": time.time(),
            "message_count": 0,
            "tags": [],
        }
        (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        transcript_path = session_dir / "transcript.jsonl"
        with open(transcript_path, "w", encoding="utf-8") as f:
            # Valid message
            user_msg = UserMessage(content=[TextBlock(text="Hello")])
            f.write(json.dumps(message_to_dict(user_msg)) + "\n")
            # Malformed line
            f.write("not valid json\n")
            # Another valid message
            asst_msg = AssistantMessage(content=[TextBlock(text="Hi there")])
            f.write(json.dumps(message_to_dict(asst_msg)) + "\n")

        loaded = Session.resume(session_id)
        assert loaded is not None
        # Should have 2 messages (malformed line skipped)
        assert len(loaded.conversation.messages) == 2
