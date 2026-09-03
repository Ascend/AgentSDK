#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

"""Tests for migrated session persistence."""
# pylint: disable=no-name-in-module, redefined-outer-name, ungrouped-imports

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

import pytest

from src.agent.session import Session
from clawcodex_ext.agent.session import _load_from_enhanced_transcript
from src.bootstrap.state import (
    ModelUsage,
    add_to_tool_duration,
    add_to_total_cost_state,
    add_to_total_duration_state,
    add_to_total_lines_changed,
    reset_state_for_tests,
)
from src.services.cost_restore import restore_cost_state_for_session
from src.services.session_migrate import (
    migrate_all,
    migrate_session,
)
from src.services.session_storage import SessionMetadata, SessionStorage
from src.types.messages import (
    AssistantMessage,
    UserMessage,
    message_to_dict,
)
from src.types.content_blocks import TextBlock, ToolUseBlock


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _HomeFixture:
    """Redirect ``Path.home()`` to a temp dir."""

    def __init__(self, tmp_home: Path) -> None:
        self.tmp_home = tmp_home
        self._patch = mock.patch.object(Path, "home", return_value=tmp_home)

    def __enter__(self):
        self._patch.start()
        return self

    def __exit__(self, *args):
        self._patch.stop()


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect ``Path.home()`` AND ``SESSIONS_DIR`` to ``tmp_path`` and
    reset bootstrap state.

    Both monkeypatches are needed because the production code reaches
    the sessions root via two different paths:

    * :class:`SessionStorage` falls back to the module-level
      :data:`src.services.session_storage.SESSIONS_DIR` constant
      (captured at import time).
    * ``Session.save()`` / ``Session.load()`` resolve the path via
      :func:`Path.home` directly.

    Patching only one of them leaves the other writing to the real
    ``~/.clawcodex`` directory and the test then reads from an empty
    tmp path. Mirrors the pattern used by
    ``tests/services/test_session_resume_unified.py``.
    """
    reset_state_for_tests()
    sessions_root = tmp_path / ".clawcodex" / "sessions"
    monkeypatch.setattr("clawcodex_ext.services.session_storage.SESSIONS_DIR", sessions_root)
    patch = mock.patch.object(Path, "home", return_value=tmp_path)
    patch.start()
    try:
        yield tmp_path
    finally:
        patch.stop()
        reset_state_for_tests()


@pytest.fixture
def sessions_dir(fake_home: Path) -> Path:
    """Return ``fake_home / .clawcodex / /sessions`` and ensure it exists."""
    p = fake_home / ".clawcodex" / "sessions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _prime_cost_state(model: str = "claude-sonnet-4-6", cost: float = 0.05) -> None:
    """Prime bootstrap counters so Session.save() emits a non-zero cost block."""
    usage = ModelUsage(
        input_tokens=200,
        output_tokens=80,
        cache_creation_input_tokens=10,
        cache_read_input_tokens=20,
        cost_usd=cost,
    )
    add_to_total_cost_state(cost, usage, model)
    add_to_total_duration_state(5000, 4500)
    add_to_tool_duration(2000)
    add_to_total_lines_changed(42, 8)


def _write_legacy_session(
    sessions_dir: Path,
    session_id: str,
    *,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
    num_turns: int = 2,
    with_cost: bool = True,
) -> Path:
    """Write a legacy 3-file session: session.json + metadata.json + transcript.jsonl."""
    session_dir = sessions_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # metadata.json
    metadata = {
        "session_id": session_id,
        "start_time": time.time(),
        "model": model,
        "title": f"legacy-{session_id[:8]}",
        "message_count": num_turns * 2,
        "last_updated": time.time(),
        "tags": ["legacy"],
        # Legacy fields that P5-F drops from writes — included here so
        # we can verify from_dict() tolerates them on read.
        "cwd": "/tmp/legacy",
        "total_cost": 0.123,
        "last_user_input": "hello world",
        "agent_name": "general-purpose",
        "cost": {
            "total_cost_usd": 0.123,
            "total_api_duration": 1000,
            "model_usage": {},
        },
    }
    (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # transcript.jsonl — message lines (no session_init marker)
    transcript_lines: list[str] = []
    messages: list = []
    for i in range(num_turns):
        user = UserMessage(content=[TextBlock(text=f"Turn {i} prompt")])
        assistant = AssistantMessage(
            content=[
                TextBlock(text=f"Turn {i} reply"),
                ToolUseBlock(id=f"tu_{i}", name="Read", input={"path": f"file_{i}.py"}),
            ],
        )
        for msg in (user, assistant):
            d = message_to_dict(msg)
            messages.append(d)
            transcript_lines.append(json.dumps(d, ensure_ascii=False) + "\n")
    (session_dir / "transcript.jsonl").write_text("".join(transcript_lines))

    # session.json — full snapshot
    cost_block: dict = {}
    if with_cost:
        cost_block = {
            "total_cost_usd": 0.123,
            "total_api_duration": 1000,
            "total_api_duration_without_retries": 900,
            "total_tool_duration": 500,
            "total_lines_added": 10,
            "total_lines_removed": 2,
            "last_duration": 42.0,
            "model_usage": {
                model: {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cost_usd": 0.123,
                },
            },
        }
    session_data = {
        "session_id": session_id,
        "provider": provider,
        "model": model,
        "conversation": {"messages": messages, "max_history": 2000},
        "created_at": "2026-06-19T09:00:00",
        "updated_at": "2026-06-19T09:30:00",
        "cost": cost_block,
    }
    (session_dir / "session.json").write_text(json.dumps(session_data, indent=2))
    return session_dir


# ---------------------------------------------------------------------------
# P5-F: metadata.json simplification
# ---------------------------------------------------------------------------


class TestMetadataSimplification:
    """P5-F: metadata.json no longer writes the legacy fields."""

    def test_to_dict_excludes_legacy_fields(self, tmp_path: Path) -> None:
        meta = SessionMetadata(
            session_id="sid",
            model="claude-sonnet-4-20250514",
            title="hello",
            cwd="/should/not/serialize",
            total_cost=99.0,
            last_user_input="should not serialize",
            agent_name="should-not-serialize",
            cost={"key": "value"},
        )
        d = meta.to_dict()
        assert "cwd" not in d
        assert "total_cost" not in d
        assert d["last_user_input"] == "should not serialize"
        assert "agent_name" not in d
        assert "cost" not in d
        # List-summary fields ARE written.
        for key in ("session_id", "model", "title", "message_count", "last_updated", "tags"):
            assert key in d

    def test_from_dict_tolerates_legacy_fields(self) -> None:
        """Legacy metadata.json files must still load."""
        data = {
            "session_id": "legacy-sid",
            "model": "claude-opus-4",
            "cwd": "/old/path",
            "total_cost": 1.5,
            "last_user_input": "previous prompt",
            "agent_name": "general-purpose",
            "cost": {"old": "block"},
        }
        meta = SessionMetadata.from_dict(data)
        # Legacy fields stay on the in-memory instance so callers
        # that read .cwd / .cost continue to work.
        assert meta.cwd == "/old/path"
        assert meta.total_cost == 1.5
        assert meta.last_user_input == "previous prompt"
        assert meta.agent_name == "general-purpose"
        assert meta.cost == {"old": "block"}

    def test_roundtrip_drops_legacy_fields(self, tmp_path: Path) -> None:
        storage = SessionStorage(sessions_dir=tmp_path)
        storage.init_metadata(model="claude-opus-4", cwd="/tmp")
        storage.update_metadata(cwd="/elsewhere", total_cost=2.0)

        # ``SessionStorage`` creates ``<sessions_dir>/<session_id>/`` for
        # its files, so ``metadata.json`` lives one level deeper than
        # ``tmp_path`` itself.
        meta_path = storage.session_dir / "metadata.json"
        on_disk = json.loads(meta_path.read_text())
        assert "cwd" not in on_disk
        assert "total_cost" not in on_disk
        # In-memory instance still has the values.
        m = storage.get_metadata()
        assert m is not None
        assert m.cwd == "/elsewhere"
        assert m.total_cost == 2.0

    def test_init_metadata_accepts_cwd_legacy(self, tmp_path: Path) -> None:
        """``init_metadata(cwd=...)`` is accepted for backward compat
        but the value is NOT persisted.
        """
        storage = SessionStorage(sessions_dir=tmp_path)
        storage.init_metadata(model="claude-opus-4", cwd="/legacy/path")
        on_disk = json.loads((storage.session_dir / "metadata.json").read_text())
        assert "cwd" not in on_disk


# ---------------------------------------------------------------------------
# P5-A / P5-B / P5-D: Session.save + load in the new format
# ---------------------------------------------------------------------------


class TestSessionSaveAndLoad:
    """P5-A: Session.save() drops session.json, appends session_snapshot."""

    def test_save_does_not_write_session_json(self, fake_home: Path, sessions_dir: Path) -> None:
        _prime_cost_state()
        sess = Session.create("anthropic", "claude-sonnet-4-20250514")
        sess.save()

        sid = sess.session_id
        session_dir = fake_home / ".clawcodex" / "sessions" / sid
        assert (session_dir / "session.json").exists() is False
        # The transcript exists and contains the snapshot at the tail.
        transcript_path = session_dir / "transcript.jsonl"
        assert transcript_path.exists()
        lines = [line for line in transcript_path.read_text().splitlines() if line.strip()]
        assert lines, "transcript.jsonl should not be empty"
        tail = json.loads(lines[-1])
        assert tail.get("type") == "session_snapshot"
        assert "cost" in tail

    def test_save_then_load_round_trip_preserves_messages(self, fake_home: Path, sessions_dir: Path) -> None:
        """Acceptance scenario #4: save → load → save → load yields identical messages."""
        _prime_cost_state()
        sess = Session.create("anthropic", "claude-sonnet-4-20250514")
        sess.conversation.add_user_message("hello")
        sess.conversation.add_assistant_message("world")
        sess.save()
        sid = sess.session_id

        first = Session.load(sid)
        assert first is not None
        first_msgs = list(first.conversation.messages)
        first_uuids = [m.uuid for m in first_msgs]
        assert len(first_msgs) == 2
        first.save()

        second = Session.load(sid)
        assert second is not None
        second_msgs = list(second.conversation.messages)
        second_uuids = [m.uuid for m in second_msgs]
        assert first_uuids == second_uuids, (
            f"uuids changed across save→load→save→load: first={first_uuids}, second={second_uuids}"
        )

    def test_load_enhanced_transcript_first_line_session_init(self, tmp_path: Path) -> None:
        """P5-B: ``_load_from_enhanced_transcript`` reads session_init as line 1."""
        transcript_path = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "session_init",
                    "session_id": "abc",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "created_at": "2026-06-19T09:00:00",
                }
            ),
            json.dumps(message_to_dict(UserMessage(content=[TextBlock(text="hi")]))),
            json.dumps(message_to_dict(AssistantMessage(content=[TextBlock(text="hello")]))),
            json.dumps({"type": "session_snapshot", "cost": {}}),
        ]
        transcript_path.write_text("\n".join(lines) + "\n")

        loaded = _load_from_enhanced_transcript("abc", transcript_path)
        assert loaded is not None
        assert loaded.provider == "openai"
        assert loaded.model == "gpt-4o"
        assert len(loaded.conversation.messages) == 2

    def test_load_enhanced_transcript_returns_none_for_legacy(self, tmp_path: Path) -> None:
        """P5-B: transcript without session_init marker → return None
        so caller falls back to session.json.
        """
        transcript_path = tmp_path / "transcript.jsonl"
        # Legacy transcript: first line is a message, no init marker.
        msg = message_to_dict(UserMessage(content=[TextBlock(text="hi")]))
        transcript_path.write_text(json.dumps(msg) + "\n")

        loaded = _load_from_enhanced_transcript("legacy", transcript_path)
        assert loaded is None

    def test_load_enhanced_transcript_returns_none_for_empty_file(self, tmp_path: Path) -> None:
        """Regression: an EMPTY transcript.jsonl must also return None.

        Pre-fix, an empty file produced a Session with empty
        provider/model fields, which masked the legacy ``session.json``
        fallback (Session.load() never tried the next branch in the
        chain). The fix: treat "no first non-blank line" the same as
        "first line is not a session_init" — both mean "this
        transcript doesn't carry the new-format anchor, look elsewhere".
        """
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text("")  # zero bytes — common for fresh sessions

        loaded = _load_from_enhanced_transcript("empty", transcript_path)
        assert loaded is None

    def test_load_falls_back_to_session_json_when_transcript_is_empty(
        self, fake_home: Path, sessions_dir: Path
    ) -> None:
        """Regression: legacy session.json must be honored when an empty
        transcript.jsonl sits next to it.

        Common shape for pre-Phase-5 sessions that wrote a snapshot
        before flushing any turn messages.
        """
        session_id = f"empty-trans-{uuid4().hex[:8]}"
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        # Empty transcript alongside a fully-populated session.json.
        (session_dir / "transcript.jsonl").write_text("")
        (session_dir / "session.json").write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "provider": "anthropic",
                    "model": "claude-opus-4",
                    "conversation": {
                        "messages": [
                            message_to_dict(UserMessage(content=[TextBlock(text="recovered")])),
                        ]
                    },
                    "created_at": "2026-06-19",
                    "updated_at": "2026-06-19",
                    "cost": {"total_cost_usd": 0.0, "model_usage": {}},
                }
            )
        )

        loaded = Session.load(session_id)
        assert loaded is not None
        assert loaded.provider == "anthropic"
        assert loaded.model == "claude-opus-4"
        assert len(loaded.conversation.messages) == 1


class TestSessionLoadLegacyFallback:
    """P5-B: Session.load() falls back to session.json when transcript
    lacks session_init marker (pre-Phase-5 saves).
    """

    def test_load_falls_back_to_session_json(self, fake_home: Path, sessions_dir: Path) -> None:
        session_id = f"legacy-{uuid4().hex[:12]}"
        _write_legacy_session(sessions_dir, session_id, num_turns=2)

        loaded = Session.load(session_id)
        assert loaded is not None
        assert loaded.provider == "anthropic"
        assert loaded.model == "claude-sonnet-4-20250514"
        # 2 turns × (user + assistant) = 4 messages from the legacy snapshot.
        assert len(loaded.conversation.messages) == 4

    def test_load_falls_back_to_metadata_when_no_session_json(self, fake_home: Path, sessions_dir: Path) -> None:
        """Pure orchestrator/cron legacy: metadata + transcript, no session.json."""
        session_id = f"orch-{uuid4().hex[:12]}"
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "model": "claude-sonnet-4-20250514",
                    "start_time": time.time(),
                    "last_updated": time.time(),
                    "message_count": 2,
                    "tags": ["orchestrator"],
                }
            )
        )
        msg1 = message_to_dict(UserMessage(content=[TextBlock(text="Q")]))
        msg2 = message_to_dict(AssistantMessage(content=[TextBlock(text="A")]))
        (session_dir / "transcript.jsonl").write_text(json.dumps(msg1) + "\n" + json.dumps(msg2) + "\n")

        loaded = Session.load(session_id)
        assert loaded is not None
        assert loaded.model == "claude-sonnet-4-20250514"
        assert len(loaded.conversation.messages) == 2


# ---------------------------------------------------------------------------
# P5-C: cost_restore reads transcript tail first
# ---------------------------------------------------------------------------


class TestCostRestoreFromTranscriptTail:
    """P5-C: cost_restore prefers the transcript's session_snapshot line."""

    def test_restores_from_session_snapshot_line(self, fake_home: Path, sessions_dir: Path) -> None:
        session_id = f"p5c-{uuid4().hex[:12]}"
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        cost_block = {
            "total_cost_usd": 0.99,
            "total_api_duration": 1234,
            "total_api_duration_without_retries": 1100,
            "total_tool_duration": 800,
            "total_lines_added": 11,
            "total_lines_removed": 3,
            "last_duration": 60.0,
            "model_usage": {
                "claude-sonnet-4-20250514": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cost_usd": 0.99,
                },
            },
        }
        # transcript with session_init + messages + session_snapshot
        transcript = (
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "session_init",
                            "session_id": session_id,
                            "provider": "anthropic",
                            "model": "claude-sonnet-4-20250514",
                        }
                    ),
                    json.dumps(message_to_dict(UserMessage(content=[TextBlock(text="Q")]))),
                    json.dumps({"type": "session_snapshot", "cost": cost_block}),
                ]
            )
            + "\n"
        )
        (session_dir / "transcript.jsonl").write_text(transcript)

        ok = restore_cost_state_for_session(session_id)
        assert ok is True

    def test_restores_from_legacy_cost_block_line(self, fake_home: Path, sessions_dir: Path) -> None:
        """Pre-P5-E transcripts use ``cost_block`` instead of
        ``session_snapshot`` — the reader accepts both.
        """
        session_id = f"p5c-legacy-{uuid4().hex[:12]}"
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        cost_block = {
            "total_cost_usd": 0.42,
            "total_api_duration": 100,
            "total_api_duration_without_retries": 100,
            "total_tool_duration": 50,
            "total_lines_added": 1,
            "total_lines_removed": 0,
            "last_duration": 1.0,
            "model_usage": {},
        }
        transcript = (
            "\n".join(
                [
                    json.dumps(message_to_dict(UserMessage(content=[TextBlock(text="Q")]))),
                    json.dumps({"type": "cost_block", "cost": cost_block}),
                ]
            )
            + "\n"
        )
        (session_dir / "transcript.jsonl").write_text(transcript)

        ok = restore_cost_state_for_session(session_id)
        assert ok is True

    def test_falls_back_to_session_json_when_no_snapshot(self, fake_home: Path, sessions_dir: Path) -> None:
        """Legacy session.json without trailing snapshot line is still honored."""
        session_id = f"p5c-fallback-{uuid4().hex[:12]}"
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        cost_block = {"total_cost_usd": 0.5, "model_usage": {}}
        (session_dir / "session.json").write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "provider": "anthropic",
                    "model": "claude-opus-4",
                    "conversation": {"messages": []},
                    "created_at": "2026-06-19",
                    "updated_at": "2026-06-19",
                    "cost": cost_block,
                }
            )
        )

        ok = restore_cost_state_for_session(session_id)
        assert ok is True

    def test_returns_false_when_nothing_to_restore(self, fake_home: Path, sessions_dir: Path) -> None:
        session_id = f"empty-{uuid4().hex[:12]}"
        ok = restore_cost_state_for_session(session_id)
        assert ok is False


# ---------------------------------------------------------------------------
# P5-E: session_init written by save_to_session_storage
# ---------------------------------------------------------------------------


class TestSessionPersist:
    """P5-E: save_to_session_storage writes session_init as line 1."""

    def test_first_save_writes_session_init_as_line_1(self, fake_home: Path, sessions_dir: Path) -> None:
        sess = Session.create("anthropic", "claude-sonnet-4-20250514")
        sess.conversation.add_user_message("hello")
        sess.save()
        sid = sess.session_id

        transcript_path = sessions_dir / sid / "transcript.jsonl"
        lines = [line for line in transcript_path.read_text().splitlines() if line.strip()]
        assert lines, "transcript.jsonl must not be empty after save"
        first = json.loads(lines[0])
        assert first.get("type") == "session_init"
        assert first.get("provider") == "anthropic"
        assert first.get("model") == "claude-sonnet-4-20250514"
        assert first.get("session_id") == sid

    def test_second_save_does_not_duplicate_session_init(self, fake_home: Path, sessions_dir: Path) -> None:
        """Idempotent: subsequent saves must not write a second init line."""
        sess = Session.create("anthropic", "claude-sonnet-4-20250514")
        sess.conversation.add_user_message("first")
        sess.save()
        sid = sess.session_id

        sess.conversation.add_user_message("second")
        sess.save()

        transcript_path = sessions_dir / sid / "transcript.jsonl"
        init_count = 0
        for line in transcript_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("type") == "session_init":
                init_count += 1
        assert init_count == 1, f"expected exactly 1 session_init, got {init_count}"


# ---------------------------------------------------------------------------
# P5-H: migration tool
# ---------------------------------------------------------------------------


class TestSessionMigration:
    """P5-H: migrate_session() converts legacy 3-file sessions to unified 2-file format."""

    def test_migrate_legacy_session_creates_session_init(self, fake_home: Path, sessions_dir: Path) -> None:
        session_id = f"mig-{uuid4().hex[:12]}"
        _write_legacy_session(sessions_dir, session_id, num_turns=2, with_cost=True)

        result = migrate_session(session_id, remove_legacy=False)
        assert result.migrated is True
        assert result.error == ""
        assert result.source_session_json is True
        assert result.cost_migrated is True
        assert result.messages_migrated == 4
        assert result.removed_session_json is False

        transcript_path = sessions_dir / session_id / "transcript.jsonl"
        lines = [line for line in transcript_path.read_text().splitlines() if line.strip()]
        first = json.loads(lines[0])
        assert first.get("type") == "session_init"
        assert first.get("provider") == "anthropic"
        assert first.get("model") == "claude-sonnet-4-20250514"
        # session_snapshot is the trailing line.
        tail = json.loads(lines[-1])
        assert tail.get("type") == "session_snapshot"
        assert "cost" in tail
        # session.json is still there (remove_legacy=False).
        assert (sessions_dir / session_id / "session.json").exists()

    def test_migrate_with_remove_legacy(self, fake_home: Path, sessions_dir: Path) -> None:
        session_id = f"mig-rm-{uuid4().hex[:12]}"
        _write_legacy_session(sessions_dir, session_id, num_turns=1, with_cost=True)

        result = migrate_session(session_id, remove_legacy=True)
        assert result.migrated is True
        assert result.removed_session_json is True
        assert not (sessions_dir / session_id / "session.json").exists()

    def test_migrate_idempotent(self, fake_home: Path, sessions_dir: Path) -> None:
        """Running migrate twice: second call is a no-op (skipped)."""
        session_id = f"mig-idem-{uuid4().hex[:12]}"
        _write_legacy_session(sessions_dir, session_id, num_turns=1, with_cost=True)

        first = migrate_session(session_id, remove_legacy=False)
        assert first.migrated is True

        second = migrate_session(session_id, remove_legacy=False)
        assert second.migrated is False
        assert "session_init marker" in second.skipped_reason

    def test_migrate_session_then_load_works(self, fake_home: Path, sessions_dir: Path) -> None:
        """Acceptance scenario: after migration Session.load() reads the new format."""
        session_id = f"mig-load-{uuid4().hex[:12]}"
        _write_legacy_session(sessions_dir, session_id, num_turns=2, with_cost=True)

        migrate_session(session_id, remove_legacy=False)
        loaded = Session.load(session_id)
        assert loaded is not None
        assert loaded.provider == "anthropic"
        assert loaded.model == "claude-sonnet-4-20250514"
        assert len(loaded.conversation.messages) == 4

    def test_migrate_session_without_session_json(self, fake_home: Path, sessions_dir: Path) -> None:
        """A session that ONLY has transcript.jsonl + metadata.json is migrated
        to the new format (session_init is prepended).
        """
        session_id = f"mig-nojson-{uuid4().hex[:12]}"
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "start_time": time.time(),
                    "model": "claude-opus-4",
                    "title": "nojson",
                    "last_updated": time.time(),
                    "message_count": 2,
                    "tags": [],
                }
            )
        )
        msg1 = message_to_dict(UserMessage(content=[TextBlock(text="Q")]))
        msg2 = message_to_dict(AssistantMessage(content=[TextBlock(text="A")]))
        (session_dir / "transcript.jsonl").write_text(json.dumps(msg1) + "\n" + json.dumps(msg2) + "\n")

        result = migrate_session(session_id, remove_legacy=False)
        assert result.migrated is True
        assert result.source_session_json is False
        # session_init was still written.
        first_line = (session_dir / "transcript.jsonl").read_text().splitlines()[0]
        assert json.loads(first_line).get("type") == "session_init"

    def test_migrate_missing_session_directory(self, fake_home: Path, sessions_dir: Path) -> None:
        result = migrate_session("does-not-exist", sessions_dir=sessions_dir)
        assert result.migrated is False
        assert result.error

    def test_migrate_all_walks_directory(self, fake_home: Path, sessions_dir: Path) -> None:
        for i in range(3):
            _write_legacy_session(
                sessions_dir,
                f"batch-{i}-{uuid4().hex[:8]}",
                num_turns=1,
                with_cost=False,
            )

        summary = migrate_all(sessions_dir=sessions_dir, remove_legacy=False)
        assert summary.total_sessions == 3
        assert summary.migrated_count == 3
        assert summary.error_count == 0


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


class TestSessionMigrateCLI:
    """The ``session migrate`` subcommand is registered."""

    def test_session_subcommand_registered(self) -> None:
        from clawcodex_ext.cli.subcommand_registry import load_builtin_subcommands, get_subcommand

        load_builtin_subcommands()
        sub = get_subcommand("session")
        assert sub is not None
        assert callable(sub)

    def test_session_migrate_no_args_returns_2(self) -> None:
        from clawcodex_ext.cli.session_migrate_cmd import run_session_command

        # session migrate --from-3-file requires session id or --all
        rc = run_session_command(["migrate", "--from-3-file"])
        assert rc == 2

    def test_session_migrate_all_runs(self, fake_home: Path, sessions_dir: Path, capsys) -> None:
        from clawcodex_ext.cli.session_migrate_cmd import run_session_command

        _write_legacy_session(sessions_dir, f"cli-{uuid4().hex[:8]}", num_turns=1, with_cost=True)
        rc = run_session_command(["migrate", "--from-3-file", "--all"])
        captured = capsys.readouterr()
        assert rc == 0
        # stdout is JSON; "migrated_count" should appear.
        assert "migrated_count" in captured.out

    def test_session_migrate_specific_id(self, fake_home: Path, sessions_dir: Path, capsys) -> None:
        from clawcodex_ext.cli.session_migrate_cmd import run_session_command

        sid = f"cli-id-{uuid4().hex[:8]}"
        _write_legacy_session(sessions_dir, sid, num_turns=1, with_cost=True)
        rc = run_session_command(["migrate", "--from-3-file", sid])
        captured = capsys.readouterr()
        assert rc == 0
        assert sid in captured.out


if __name__ == "__main__":
    unittest.main()
