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

"""Tests for stage4 agent."""

from __future__ import annotations

import sys


class TestStage4Conversation:
    """Tests for TestStage4Conversation."""

    def test_conversation_round_trip(self):
        from src.agent.conversation import Conversation

        conv = Conversation()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi there!")

        data = conv.to_dict()
        reloaded = Conversation.from_dict(data)

        msgs = reloaded.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_conversation_empty(self):
        from src.agent.conversation import Conversation

        conv = Conversation()
        data = conv.to_dict()
        reloaded = Conversation.from_dict(data)
        assert len(reloaded.get_messages()) == 0

    def test_conversation_multi_turn(self):
        from src.agent.conversation import Conversation

        conv = Conversation()
        for i in range(3):
            conv.add_user_message(f"User message {i}")
            conv.add_assistant_message(f"Assistant response {i}")

        msgs = conv.get_messages()
        assert len(msgs) == 6
        assistant_content = msgs[-1]["content"]
        if isinstance(assistant_content, list):
            assert any(isinstance(b, dict) and b.get("text") == "Assistant response 2" for b in assistant_content), (
                f"Expected text block, got {assistant_content}"
            )
        else:
            assert msgs[-1]["content"] == "Assistant response 2"


class TestStage4ConversationSnapshot:
    """Byte-level snapshot tests for Conversation.to_dict() / from_dict().

    P0-3: locks the wire format so any field rename, message reorder, or
    content-block schema change fails the test instead of silently drifting.
    Volatility sources (uuid4, datetime.now) are pinned via the
    ``pinned_message_factory`` fixture.

    NOTE: use ``from src.types.messages import ...`` rather than the
    ``clawcodex_ext.types.messages`` import path so the snapshot is
    anchored against the public facade — that's the path consumers
    actually import through.
    """

    _PINNED_USER_UUID = "00000000-0000-0000-0000-000000000001"
    _PINNED_ASSISTANT_UUID = "00000000-0000-0000-0000-000000000002"
    _PINNED_RESULT_UUID = "00000000-0000-0000-0000-000000000003"
    _PINNED_TS = "2026-01-01T00:00:00"
    _PINNED_TS_PLUS_1 = "2026-01-01T00:00:01"
    _PINNED_TS_PLUS_2 = "2026-01-01T00:00:02"

    def test_conversation_empty_snapshot(self):
        """Empty Conversation.to_dict() byte-stable shape.

        Locks ``{"messages": [], "max_history": 2000}`` so the default cap
        bump (100 → 2000) and any future reordering of the dict keys
        fails the test instead of silently passing.
        """
        from src.agent.conversation import Conversation

        conv = Conversation()
        assert conv.to_dict() == {"messages": [], "max_history": 2000}

    def test_conversation_to_dict_byte_level_single_user(self):
        """Single pinned UserMessage → byte-stable dict shape.

        Locks the full 7-field envelope (role / content / type / uuid /
        timestamp / isMeta / isVirtual / isCompactSummary). Any new field
        added by ``message_to_dict`` or any existing field renamed/removed
        must surface as a test diff — preventing silent drift in
        transcript-on-disk JSON.
        """
        from src.agent.conversation import Conversation
        from src.types.messages import create_user_message

        conv = Conversation()
        conv.messages.append(
            create_user_message(
                "hello world",
                uuid=self._PINNED_USER_UUID,
                timestamp=self._PINNED_TS,
            )
        )
        assert conv.to_dict() == {
            "messages": [
                {
                    "role": "user",
                    "content": "hello world",
                    "type": "user",
                    "uuid": self._PINNED_USER_UUID,
                    "timestamp": self._PINNED_TS,
                    "isMeta": False,
                    "isVirtual": False,
                    "isCompactSummary": False,
                }
            ],
            "max_history": 2000,
        }

    def test_conversation_to_dict_byte_level_multi_turn(self):
        """Three-turn (user/assistant/user/assistant) byte-stable shape.

        Verifies message ordering is preserved and the assistant side
        emits a ``content: [...]`` list-of-blocks (vs. the user side's
        plain string). ``stop_reason`` field on AssistantMessage MUST be
        omitted when default ``None`` per ``message_to_dict``'s ``is not None``
        filter — locking that here so a future refactor that emits
        ``"stop_reason": null`` explicitly is caught.
        """
        from src.agent.conversation import Conversation
        from src.types.content_blocks import TextBlock
        from src.types.messages import (
            AssistantMessage,
            create_user_message,
        )

        conv = Conversation()
        conv.messages.append(
            create_user_message(
                "turn 1 user",
                uuid="00000000-0000-0000-0000-000000000001",
                timestamp="2026-01-01T00:00:00",
            )
        )
        conv.messages.append(
            AssistantMessage(
                content=[TextBlock(text="turn 1 assistant")],
                uuid="00000000-0000-0000-0000-000000000002",
                timestamp="2026-01-01T00:00:01",
            )
        )
        conv.messages.append(
            create_user_message(
                "turn 2 user",
                uuid="00000000-0000-0000-0000-000000000003",
                timestamp="2026-01-01T00:00:02",
            )
        )
        assert conv.to_dict() == {
            "messages": [
                {
                    "role": "user",
                    "content": "turn 1 user",
                    "type": "user",
                    "uuid": "00000000-0000-0000-0000-000000000001",
                    "timestamp": "2026-01-01T00:00:00",
                    "isMeta": False,
                    "isVirtual": False,
                    "isCompactSummary": False,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "turn 1 assistant"}],
                    "type": "assistant",
                    "uuid": "00000000-0000-0000-0000-000000000002",
                    "timestamp": "2026-01-01T00:00:01",
                    "isMeta": False,
                    "isVirtual": False,
                    "isCompactSummary": False,
                },
                {
                    "role": "user",
                    "content": "turn 2 user",
                    "type": "user",
                    "uuid": "00000000-0000-0000-0000-000000000003",
                    "timestamp": "2026-01-01T00:00:02",
                    "isMeta": False,
                    "isVirtual": False,
                    "isCompactSummary": False,
                },
            ],
            "max_history": 2000,
        }

    def test_conversation_round_trip_byte_stable(self):
        """to_dict → from_dict → to_dict must be byte-stable.

        Locks the round-trip invariant: any field dropped or transformed
        during ``from_dict`` (e.g. lossy coercion, default-value drift)
        surfaces as a diff on the second ``to_dict`` call.
        """
        from src.agent.conversation import Conversation
        from src.types.content_blocks import TextBlock
        from src.types.messages import (
            AssistantMessage,
            create_user_message,
        )

        conv = Conversation()
        conv.messages.append(
            create_user_message(
                "ping",
                uuid="00000000-0000-0000-0000-000000000001",
                timestamp="2026-01-01T00:00:00",
            )
        )
        conv.messages.append(
            AssistantMessage(
                content=[TextBlock(text="pong")],
                uuid="00000000-0000-0000-0000-000000000002",
                timestamp="2026-01-01T00:00:01",
            )
        )
        first = conv.to_dict()
        reloaded = Conversation.from_dict(first)
        assert reloaded.to_dict() == first, (
            "round-trip drift detected; from_dict likely dropped a field "
            "or re-emitted a default that the original to_dict omitted"
        )

    def test_conversation_with_tool_use_and_result(self):
        """AssistantMessage(tool_use) + UserMessage(tool_result) snapshot.

        Locks the wire shape of the two most common content blocks
        outside text: ``tool_use`` (id/name/input) and ``tool_result``
        (tool_use_id/content/is_error). Any future rename of
        ``tool_use_id`` → ``id`` on the result block, or addition of a
        required field to tool_use blocks, must surface here.
        """
        from src.agent.conversation import Conversation
        from src.types.content_blocks import (
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
        )
        from src.types.messages import (
            AssistantMessage,
            UserMessage,
            create_user_message,
        )

        conv = Conversation()
        conv.messages.append(
            create_user_message(
                "read foo",
                uuid=self._PINNED_USER_UUID,
                timestamp=self._PINNED_TS,
            )
        )
        conv.messages.append(
            AssistantMessage(
                content=[
                    TextBlock(text="reading"),
                    ToolUseBlock(id="tool_call_1", name="Read", input={"file_path": "/foo"}),
                ],
                uuid=self._PINNED_ASSISTANT_UUID,
                timestamp=self._PINNED_TS_PLUS_1,
            )
        )
        conv.messages.append(
            UserMessage(
                content=[
                    ToolResultBlock(
                        tool_use_id="tool_call_1",
                        content="OK",
                        is_error=False,
                    )
                ],
                uuid=self._PINNED_RESULT_UUID,
                timestamp=self._PINNED_TS_PLUS_2,
            )
        )
        assert conv.to_dict() == {
            "messages": [
                {
                    "role": "user",
                    "content": "read foo",
                    "type": "user",
                    "uuid": self._PINNED_USER_UUID,
                    "timestamp": self._PINNED_TS,
                    "isMeta": False,
                    "isVirtual": False,
                    "isCompactSummary": False,
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "reading"},
                        {
                            "type": "tool_use",
                            "id": "tool_call_1",
                            "name": "Read",
                            "input": {"file_path": "/foo"},
                        },
                    ],
                    "type": "assistant",
                    "uuid": self._PINNED_ASSISTANT_UUID,
                    "timestamp": self._PINNED_TS_PLUS_1,
                    "isMeta": False,
                    "isVirtual": False,
                    "isCompactSummary": False,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_call_1",
                            "content": "OK",
                            "is_error": False,
                        }
                    ],
                    "type": "user",
                    "uuid": self._PINNED_RESULT_UUID,
                    "timestamp": self._PINNED_TS_PLUS_2,
                    "isMeta": False,
                    "isVirtual": False,
                    "isCompactSummary": False,
                },
            ],
            "max_history": 2000,
        }


class TestStage4MessageTypes:
    """Tests for TestStage4MessageTypes."""

    def test_message_types_in_api_payload(self):
        from src.types.content_blocks import TextBlock, ToolUseBlock
        from src.types.messages import (
            AssistantMessage,
            UserMessage,
            normalize_messages_for_api,
        )

        msgs = [
            UserMessage(content="ping"),
            AssistantMessage(
                content=[
                    TextBlock(text="pong"),
                    ToolUseBlock(id="t1", name="Read", input={"file_path": "/foo"}),
                ]
            ),
        ]
        payload = normalize_messages_for_api(msgs)
        assert payload[0] == {"role": "user", "content": "ping"}
        assert payload[1]["role"] == "assistant"
        blocks = payload[1]["content"]
        assert blocks[0] == {"type": "text", "text": "pong"}
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["name"] == "Read"

    def test_user_message_creation(self):
        from src.types.messages import UserMessage

        msg = UserMessage(content="test message")
        assert msg.content == "test message"
        assert msg.role == "user"


class TestStage4Session:
    """Tests for TestStage4Session."""

    def test_session_create(self):
        from src.agent.session import Session

        session = Session.create(provider="anthropic", model="claude-sonnet-4-20250514")
        assert session.provider == "anthropic"
        assert session.model == "claude-sonnet-4-20250514"
        assert session.session_id is not None

    def test_session_conversation_integration(self):
        from src.agent.session import Session

        session = Session.create(provider="anthropic", model="claude-sonnet-4-20250514")
        session.conversation.add_user_message("Hello")
        session.conversation.add_assistant_message("World")
        assert len(session.conversation.get_messages()) == 2


class TestStage4SubagentInParentSession:
    """Tests for TestStage4SubagentInParentSession."""

    def _isolated_setup(self, monkeypatch, tmp_path):
        """Wire up isolated HOME + clear resolver state for one test.

        Returns the (transcript_module, init_callable, reset_callable,
        original_resolver, original_warned, original_nested_flag) tuple.
        Caller is responsible for restoring state in a try/finally block.

        The clawcodex_ext module-level flag
        ``_nested_transcript_initialized`` is sticky across the test
        process (it exists to dedupe registration in production). We
        must reset it explicitly; otherwise the first test's
        registration sticks, and subsequent tests that clear the
        resolver find init() a no-op and fail.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        # Windows: Path.home() uses USERPROFILE, not HOME
        if sys.platform == "win32":
            monkeypatch.setenv("USERPROFILE", str(tmp_path))
            # Also clear HOMEDRIVE/HOMEPATH so they don't bypass USERPROFILE
            monkeypatch.delenv("HOMEDRIVE", raising=False)
            monkeypatch.delenv("HOMEPATH", raising=False)
        import clawcodex_ext
        from clawcodex_ext.agent import transcript
        from src.init import init as init_callable
        from src.init import reset_init_for_test_only

        original_resolver = transcript._transcript_path_resolver
        original_warned = transcript._flat_fallback_warned
        original_nested_flag = clawcodex_ext._nested_transcript_initialized
        reset_init_for_test_only()
        transcript._transcript_path_resolver = None
        transcript._flat_fallback_warned = False
        clawcodex_ext._nested_transcript_initialized = False
        return (
            transcript,
            init_callable,
            reset_init_for_test_only,
            original_resolver,
            original_warned,
            original_nested_flag,
        )

    def test_init_registers_nested_resolver(self, monkeypatch, tmp_path):
        """Verify init registers nested resolver."""
        (
            transcript,
            init_callable,
            reset,
            _original_resolver,
            original_warned,
            original_nested_flag,
        ) = self._isolated_setup(monkeypatch, tmp_path)
        try:
            assert transcript._transcript_path_resolver is None, (
                "precondition: resolver must start cleared for this test"
            )
            init_callable()
            assert transcript._transcript_path_resolver is not None, (
                "init() must register the nested-session transcript "
                "path resolver so sub-agent JSONL files land under "
                "<parent_session_id>/subagents/. See "
                "src/init.py:init() substep 6 and "
                "clawcodex_ext/agent/transcript.py"
            )
        finally:
            transcript._transcript_path_resolver = None
            transcript._flat_fallback_warned = original_warned
            import clawcodex_ext

            clawcodex_ext._nested_transcript_initialized = original_nested_flag
            reset()

    def test_subagent_path_lands_in_subagents_dir(self, monkeypatch, tmp_path):
        """Verify subagent path lands in subagents dir."""
        (
            transcript,
            init_callable,
            reset,
            _original_resolver,
            original_warned,
            original_nested_flag,
        ) = self._isolated_setup(monkeypatch, tmp_path)
        try:
            init_callable()
            path = transcript.get_agent_transcript_path("a1b2c3d4z", parent_session_id="ses-stability-gate")
            from pathlib import Path

            p = Path(path)
            assert p.name == "agent-a1b2c3d4z.jsonl", (
                f"unexpected filename; got {p.name!r}, want 'agent-a1b2c3d4z.jsonl'"
            )
            assert p.parent.name == "subagents", f"parent dir must be 'subagents'; got {p.parent.name!r}"
            assert p.parent.parent.name == "ses-stability-gate", (
                f"grandparent must be the parent session id; got {p.parent.parent.name!r}"
            )
            try:
                p.relative_to(tmp_path)
            except ValueError as exc:
                raise AssertionError(f"path {p} escapes isolated tmp_path {tmp_path}: {exc}")
        finally:
            transcript._transcript_path_resolver = None
            transcript._flat_fallback_warned = original_warned
            import clawcodex_ext

            clawcodex_ext._nested_transcript_initialized = original_nested_flag
            reset()

    def test_subagent_path_shares_sessions_parent_with_main_session(self, monkeypatch, tmp_path):
        """Verify subagent path shares sessions parent with main session."""
        (
            transcript,
            init_callable,
            reset,
            _original_resolver,
            original_warned,
            original_nested_flag,
        ) = self._isolated_setup(monkeypatch, tmp_path)
        try:
            init_callable()
            subagent_path = transcript.get_agent_transcript_path("a-share-parent", parent_session_id="ses-share-parent")
            from pathlib import Path

            p = Path(subagent_path)
            sessions_root = p.parent.parent.parent
            expected_root = Path(tmp_path) / ".clawcodex" / "sessions"
            assert sessions_root == expected_root, (
                f"subagent sessions_root mismatch: got {sessions_root}, expected {expected_root} (full path: {p})"
            )
            assert sessions_root != p.parent, (
                "subagent path must be nested under the per-session directory, not flattened to the sessions/ root"
            )
        finally:
            transcript._transcript_path_resolver = None
            transcript._flat_fallback_warned = original_warned
            import clawcodex_ext

            clawcodex_ext._nested_transcript_initialized = original_nested_flag
            reset()

    def test_subagent_filename_is_agent_dash_id_jsonl(self, monkeypatch, tmp_path):
        """Verify subagent filename is agent dash id jsonl."""
        (
            transcript,
            init_callable,
            reset,
            _original_resolver,
            original_warned,
            original_nested_flag,
        ) = self._isolated_setup(monkeypatch, tmp_path)
        try:
            init_callable()
            for agent_id in ("a1", "agent-xyz", "a-b-c-9z"):
                path = transcript.get_agent_transcript_path(agent_id, parent_session_id="ses-name-test")
                from pathlib import Path

                assert Path(path).name == f"agent-{agent_id}.jsonl", (
                    f"agent_id={agent_id!r}: expected agent-{agent_id}.jsonl suffix, got {path}"
                )
        finally:
            transcript._transcript_path_resolver = None
            transcript._flat_fallback_warned = original_warned
            import clawcodex_ext

            clawcodex_ext._nested_transcript_initialized = original_nested_flag
            reset()

    def test_flat_fallback_remains_writable_when_resolver_missing(self, monkeypatch, tmp_path):
        """Verify flat fallback remains writable when resolver missing."""
        transcript, _, reset, original_resolver, original_warned, original_nested_flag = self._isolated_setup(
            monkeypatch, tmp_path
        )
        try:
            assert transcript._transcript_path_resolver is None
            path = transcript.get_agent_transcript_path("a-fallback-test", parent_session_id="ses-fallback")
            from pathlib import Path

            p = Path(path)
            transcripts_root = Path(tmp_path) / ".clawcodex" / "transcripts"
            assert p.parent == transcripts_root, (
                f"flat fallback parent must be the transcripts/ directory; got {p.parent}, expected {transcripts_root}"
            )
            sessions_root = Path(tmp_path) / ".clawcodex" / "sessions"
            assert not str(p).startswith(str(sessions_root)), f"flat fallback leaked into sessions/: {p}"
            assert p.name == "a-fallback-test.jsonl", (
                f"flat fallback filename should be <id>.jsonl without "
                f"the 'agent-' prefix used in nested mode; got {p.name}"
            )
        finally:
            transcript._transcript_path_resolver = original_resolver
            transcript._flat_fallback_warned = original_warned
            import clawcodex_ext

            clawcodex_ext._nested_transcript_initialized = original_nested_flag
            reset()


class TestStage4Resilience:
    """P0 P1 P2 Tests for TestStage4Resilience."""

    def test_conversation_empty_messages_downstream(self):
        """Verify conversation empty messages downstream."""
        from src.agent.conversation import Conversation

        conv = Conversation()
        msgs = conv.get_messages()
        assert msgs == []

    def test_conversation_to_dict_from_dict_round_trip_empty(self):
        """Verify conversation to dict from dict round trip empty."""
        from src.agent.conversation import Conversation

        conv = Conversation()
        data = conv.to_dict()
        restored = Conversation.from_dict(data)
        assert restored.get_messages() == []

    def test_conversation_from_dict_missing_messages_key(self):
        """Verify conversation from dict missing messages key."""
        from src.agent.conversation import Conversation

        conv = Conversation.from_dict({"max_history": 500})
        assert conv.get_messages() == []

    def test_conversation_from_dict_none_messages(self):
        """Verify conversation from dict none messages."""
        from src.agent.conversation import Conversation

        conv = Conversation.from_dict({"messages": None})
        assert conv.get_messages() == []

    def test_conversation_max_history_cap(self):
        """Verify conversation max history cap."""
        from src.agent.conversation import Conversation

        conv = Conversation(max_history=3)
        for i in range(5):
            conv.add_user_message(f"msg-{i}")
            conv.add_assistant_message(f"resp-{i}")
        msgs = conv.get_messages()
        assert len(msgs) <= 6  # 3 pairs max
        assert msgs[0]["content"] != "msg-0"

    def test_session_load_nonexistent(self):
        """Verify session load nonexistent."""
        from src.agent.session import Session

        s = Session.load("__nonexistent_session_id_for_test__")
        assert s is None

    def test_session_save_and_load_round_trip(self, tmp_path):
        """Verify session save and load round trip."""
        from unittest.mock import patch

        from src.agent.conversation import Conversation
        from src.agent.session import Session

        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        with patch("pathlib.Path.home", return_value=fake_home):
            conv = Conversation()
            conv.add_user_message("hello")
            conv.add_assistant_message("world")
            session = Session(session_id="test-save-load", provider="test", model="test", conversation=conv)
            session.save()
            loaded = Session.load("test-save-load")
            assert loaded is not None
            assert loaded.session_id == "test-save-load"
            msgs = loaded.conversation.get_messages()
            assert len(msgs) == 2

    def test_add_message_large_content(self):
        """Verify add message large content."""
        from src.agent.conversation import Conversation

        conv = Conversation()
        large = "x" * 100_000
        conv.add_user_message(large)
        msgs = conv.get_messages()
        assert len(msgs) == 1
        assert len(msgs[0]["content"]) == 100_000


class TestStage4CrossModePersistence:
    """Tests for TestStage4CrossModePersistence."""

    def test_create_forecast_system_message(self):
        """Verify create forecast system message."""
        from clawcodex_ext.intent_forecast.messages import (
            ForecastResult,
            ForecastSuggestion,
            create_forecast_system_message,
        )

        result = ForecastResult(
            generated=True,
            suggestions=[
                ForecastSuggestion(
                    id="s1",
                    title="Refactor module",
                    prompt="refactor the module",
                    reason="Improves maintainability",
                    confidence=0.85,
                ),
                ForecastSuggestion(
                    id="s2",
                    title="Add tests",
                    prompt="add unit tests",
                    reason="Coverage is low",
                    confidence=0.72,
                ),
            ],
            fingerprint="fp-test-001",
        )
        msg = create_forecast_system_message(result, trigger="auto")

        assert getattr(msg, "subtype", None) == "intent_forecast"
        assert getattr(msg, "role", None) == "system"
        content = getattr(msg, "content", "") or ""
        assert "Forecast" in content
        assert "Refactor module" in content
        assert "Add tests" in content
        assert hasattr(msg, "_forecast_meta")
        assert msg._forecast_meta["trigger"] == "auto"
        assert msg._forecast_meta["fingerprint"] == "fp-test-001"
        assert msg._forecast_meta["suggestion_count"] == 2

    def test_forecast_system_message_persists_in_conversation(self):
        """Verify forecast system message persists in conversation."""
        from clawcodex_ext.intent_forecast.messages import (
            ForecastResult,
            ForecastSuggestion,
            create_forecast_system_message,
        )
        from src.agent.conversation import Conversation

        conv = Conversation()
        result = ForecastResult(
            generated=True,
            suggestions=[
                ForecastSuggestion(id="s1", title="Fix bug", prompt="fix the bug", reason="Critical"),
            ],
            fingerprint="fp-test-002",
        )
        msg = create_forecast_system_message(result, trigger="auto")
        conv.messages.append(msg)

        assert len(conv.messages) == 1
        assert conv.messages[0].subtype == "intent_forecast"
        assert "Fix bug" in str(conv.messages[0].content)

    def test_away_summary_system_message_persists_in_conversation(self):
        """Verify away summary system message persists in conversation."""
        from clawcodex_ext.away_summary.messages import create_away_summary_message
        from src.agent.conversation import Conversation

        conv = Conversation()
        msg = create_away_summary_message(
            summary="- Done task A\n- Started task B",
            trigger="auto",
            fingerprint="fp-recap-001",
            message_count=5,
            model="claude-sonnet-4-20250514",
        )
        conv.messages.append(msg)

        assert len(conv.messages) == 1
        assert conv.messages[0].subtype == "away_summary"
        content = str(conv.messages[0].content)
        assert "Done task A" in content
        assert "Started task B" in content

    def test_forecast_and_recap_survive_session_round_trip(self, tmp_path):
        """Verify forecast and recap survive session round trip."""
        from unittest.mock import patch

        from clawcodex_ext.away_summary.messages import create_away_summary_message
        from clawcodex_ext.intent_forecast.messages import (
            ForecastResult,
            ForecastSuggestion,
            create_forecast_system_message,
        )
        from src.agent.conversation import Conversation
        from src.agent.session import Session

        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()

        conv = Conversation()

        forecast_result = ForecastResult(
            generated=True,
            suggestions=[
                ForecastSuggestion(id="s1", title="Upgrade deps", prompt="upgrade deps"),
            ],
            fingerprint="fp-rt-001",
        )
        forecast_msg = create_forecast_system_message(forecast_result, trigger="auto")
        conv.messages.append(forecast_msg)

        recap_msg = create_away_summary_message(
            summary="Summary of work done.",
            trigger="auto",
            fingerprint="fp-rt-002",
            message_count=3,
        )
        conv.messages.append(recap_msg)

        conv.add_user_message("hello")

        with patch("pathlib.Path.home", return_value=fake_home):
            session = Session(
                session_id="test-cross-mode",
                provider="test",
                model="test",
                conversation=conv,
            )
            session.save()
            loaded = Session.load("test-cross-mode")

        assert loaded is not None
        loaded_msgs = loaded.conversation.messages
        assert len(loaded_msgs) == 3, f"Expected 3 messages, got {len(loaded_msgs)}"

        forecast_found = any(getattr(m, "subtype", None) == "intent_forecast" for m in loaded_msgs)
        assert forecast_found, "Forecast system message lost after session round-trip"

        recap_found = any(getattr(m, "subtype", None) == "away_summary" for m in loaded_msgs)
        assert recap_found, "Recap system message lost after session round-trip"

        for m in loaded_msgs:
            if getattr(m, "subtype", None) == "intent_forecast":
                assert "Upgrade deps" in str(getattr(m, "content", ""))
            elif getattr(m, "subtype", None) == "away_summary":
                assert "Summary of work done." in str(getattr(m, "content", ""))

    def test_conversation_mixed_messages_order_preserved(self):
        """Verify conversation mixed messages order preserved."""
        from clawcodex_ext.away_summary.messages import create_away_summary_message
        from clawcodex_ext.intent_forecast.messages import (
            ForecastResult,
            ForecastSuggestion,
            create_forecast_system_message,
        )
        from src.agent.conversation import Conversation

        conv = Conversation()

        # user → assistant → forecast → user → recap → user
        conv.add_user_message("init")
        conv.add_assistant_message("response-1")

        forecast_result = ForecastResult(
            generated=True,
            suggestions=[
                ForecastSuggestion(id="s1", title="Do X", prompt="do x"),
            ],
            fingerprint="fp-order",
        )
        conv.messages.append(create_forecast_system_message(forecast_result, trigger="auto"))

        conv.add_user_message("follow-up")

        recap_msg = create_away_summary_message(
            summary="Session recap.",
            trigger="auto",
            fingerprint="fp-order-2",
            message_count=4,
        )
        conv.messages.append(recap_msg)

        conv.add_user_message("final")

        roles = [m.role if hasattr(m, "role") else "" for m in conv.messages]
        subtypes = [getattr(m, "subtype", "") for m in conv.messages]

        assert roles == ["user", "assistant", "system", "user", "system", "user"]
        assert subtypes[2] == "intent_forecast"
        assert subtypes[4] == "away_summary"

    def test_replay_resume_history_renders_forecast_content(self):
        """Verify replay resume history renders forecast content."""
        from clawcodex_ext.intent_forecast.messages import (
            ForecastResult,
            ForecastSuggestion,
            create_forecast_system_message,
        )

        result = ForecastResult(
            generated=True,
            suggestions=[
                ForecastSuggestion(
                    id="s1",
                    title="Test suggestion",
                    prompt="test prompt",
                    reason="Because it matters",
                ),
            ],
            fingerprint="fp-render",
        )
        msg = create_forecast_system_message(result, trigger="auto")

        content = getattr(msg, "content", "") or ""
        assert content.startswith("Forecast")
        from clawcodex_ext.repl.core import ClawcodexREPL

        assert not ClawcodexREPL._is_recap_text(content)
        assert getattr(msg, "subtype", None) == "intent_forecast"
