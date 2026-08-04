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

"""Stage 4 — Agent / Conversation 测试（< 3 秒）。

验证：
- Conversation 序列化/反序列化
- Message types 类型构建和 API payload 转换
- Session 创建/保存/加载
- 子 agent transcript 在主 agent session 目录的 subagents/ 下
  （F-49 / S-R4-A：与主 session 共享 ~/.clawcodex/sessions/ 父路径，
   依赖 src.init.init() 注册 nested resolver；resolver 缺失时
   兜底写到 ~/.clawcodex/transcripts/）
- ToolUseBlock / TextBlock 构建
"""

from __future__ import annotations

import sys


class TestStage4Conversation:
    """Conversation 序列化和反序列化测试。"""

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
    """消息类型和 API payload 转换测试。"""

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
    """Session 创建 / 保存 / 加载测试。"""

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
    """子 agent transcript 必须落在主 agent session 目录的 subagents/ 子目录下。

    设计要求（g1 session 治理基本保证，F-49/S-R4-A 演进）：

    - 子 agent 与主 agent 共享父路径 ``~/.clawcodex/sessions/``，方便
      ``list_sessions`` / ``cleanup_sessions`` 一起扫描、一起清理。
    - 子 agent transcript 落在
      ``<parent_session_id>/subagents/agent-<agent_id>.jsonl``，与
      主 session 的 ``<parent_session_id>/transcript.jsonl`` 同根不同枝。
    - 该路径在 ``src.init.init()`` 之后由
      ``clawcodex_ext.agent.transcript.nested_session_path_resolver``
      提供；若 init 被旁路，则落到 flat
      ``~/.clawcodex/transcripts/<agent_id>.jsonl``，不污染主 session。

    5 个测试覆盖 wiring、路径结构、命名约定、HOME 隔离、兜底可写。
    不依赖外部 API，全部使用 stdlib + 代码内构造。
    """

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
        # 清空 init memoize 缓存 + 之前的 resolver 注册 + 嵌套注册 flag，
        # 确保从干净状态起步
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
        """``init()`` 调用后 ``_transcript_path_resolver`` 不再为 ``None``。

        防止以后入口绕开 init()、resolver 永远为 None、子 agent 全部
        落 flat 的回归。
        """
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
        """``get_agent_transcript_path(agent_id, parent_session_id=sid)``
        返回 ``<HOME>/.clawcodex/sessions/<sid>/subagents/agent-<id>.jsonl``。
        """
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
            # 文件名格式
            assert p.name == "agent-a1b2c3d4z.jsonl", (
                f"unexpected filename; got {p.name!r}, want 'agent-a1b2c3d4z.jsonl'"
            )
            # 倒一目录: subagents
            assert p.parent.name == "subagents", f"parent dir must be 'subagents'; got {p.parent.name!r}"
            # 倒二目录: parent_session_id
            assert p.parent.parent.name == "ses-stability-gate", (
                f"grandparent must be the parent session id; got {p.parent.parent.name!r}"
            )
            # HOME 隔离断言: 路径必须在 tmp_path 之下
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
        """子 agent path 与主 session 目录共享父路径
        ``~/.clawcodex/sessions/``，方便统一治理。
        """
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
            # 子 agent: agent-X.jsonl / subagents / <sid> / sessions
            sessions_root = p.parent.parent.parent
            expected_root = Path(tmp_path) / ".clawcodex" / "sessions"
            assert sessions_root == expected_root, (
                f"subagent sessions_root mismatch: got {sessions_root}, expected {expected_root} (full path: {p})"
            )
            # 共享父路径, 但子 agent 在 sessions/ 之下再深一层
            # (sessions/<sid>/subagents/), 不能直接等于 sessions/
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
        """子 agent 文件名遵循 ``agent-<agent_id>.jsonl`` 格式。

        与 ``clawcodex_ext/agent/transcript.py`` 中的字面量
        ``f"agent-{agent_id}.jsonl"`` 同步——任何变更需要两边一起改。
        """
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
        """兜底 flat 路径在 resolver 缺失时仍可写, 不污染主 session 目录。

        模拟某个未来入口漏走 init() 的回归场景——
        ``_transcript_path_resolver`` 保持 ``None``, 调用
        ``get_agent_transcript_path`` 不应抛错、路径应落在
        ``<HOME>/.clawcodex/transcripts/`` 而不是 ``<HOME>/.clawcodex/sessions/``。
        """
        transcript, _, reset, original_resolver, original_warned, original_nested_flag = self._isolated_setup(
            monkeypatch, tmp_path
        )
        try:
            # 显式保持 resolver 为 None, 不调 init, 模拟回归
            assert transcript._transcript_path_resolver is None
            path = transcript.get_agent_transcript_path("a-fallback-test", parent_session_id="ses-fallback")
            from pathlib import Path

            p = Path(path)
            transcripts_root = Path(tmp_path) / ".clawcodex" / "transcripts"
            assert p.parent == transcripts_root, (
                f"flat fallback parent must be the transcripts/ directory; got {p.parent}, expected {transcripts_root}"
            )
            # flat fallback 不应污染 sessions/ 目录
            sessions_root = Path(tmp_path) / ".clawcodex" / "sessions"
            assert not str(p).startswith(str(sessions_root)), f"flat fallback leaked into sessions/: {p}"
            # 文件名应只是 <id>.jsonl (无 agent- 前缀), 与嵌套路径区分
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
    """Conversation / Session 恢复性测试 — P0#5 空/downstream保护, P1#10 损坏恢复, P2#15 并发写安全。"""

    def test_conversation_empty_messages_downstream(self):
        """空的 Conversation.get_messages() 返回 []，下游不炸。"""
        from src.agent.conversation import Conversation

        conv = Conversation()
        msgs = conv.get_messages()
        assert msgs == []

    def test_conversation_to_dict_from_dict_round_trip_empty(self):
        """空 Conversation to_dict → from_dict 不抛异常，messages 为空。"""
        from src.agent.conversation import Conversation

        conv = Conversation()
        data = conv.to_dict()
        restored = Conversation.from_dict(data)
        assert restored.get_messages() == []

    def test_conversation_from_dict_missing_messages_key(self):
        """from_dict 入参缺失 messages key 时不抛异常。"""
        from src.agent.conversation import Conversation

        conv = Conversation.from_dict({"max_history": 500})
        assert conv.get_messages() == []

    def test_conversation_from_dict_none_messages(self):
        """from_dict 入参 messages 为 None 时不抛异常。"""
        from src.agent.conversation import Conversation

        conv = Conversation.from_dict({"messages": None})
        assert conv.get_messages() == []

    def test_conversation_max_history_cap(self):
        """超过 max_history 时旧消息被截断，不爆炸。"""
        from src.agent.conversation import Conversation

        conv = Conversation(max_history=3)
        for i in range(5):
            conv.add_user_message(f"msg-{i}")
            conv.add_assistant_message(f"resp-{i}")
        msgs = conv.get_messages()
        assert len(msgs) <= 6  # 3 pairs max
        assert msgs[0]["content"] != "msg-0"  # 旧消息被弹出

    def test_session_load_nonexistent(self):
        """Session.load 不存在的 session_id 返回 None 而非抛异常。"""
        from src.agent.session import Session

        s = Session.load("__nonexistent_session_id_for_test__")
        assert s is None

    def test_session_save_and_load_round_trip(self, tmp_path):
        """Session.save 后 Session.load 能恢复。"""
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
        """给 Conversation 添加超长字符串不崩溃。"""
        from src.agent.conversation import Conversation

        conv = Conversation()
        large = "x" * 100_000
        conv.add_user_message(large)
        msgs = conv.get_messages()
        assert len(msgs) == 1
        assert len(msgs[0]["content"]) == 100_000


class TestStage4CrossModePersistence:
    """F-103: Recapitulate & Forecast 跨 REPL↔TUI 模式切换时内容保持。

    Recapitulate（away_summary）和 Forecast（intent_forecast）在 REPL 中触发后
    被持久化为 ``SystemMessage(subtype=...)`` 追加到 conversation，切换至 TUI
    后通过重放 history 重新渲染。这些测试验证关键链路的完整性。
    """

    def test_create_forecast_system_message(self):
        """create_forecast_system_message 生成正确的 SystemMessage。"""
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
        """Forecast SystemMessage 追加到 conversation 后通过 messages 属性可读取。

        注：``get_messages()`` 会过滤非 local_command 的 system 消息（API 规格），
        但 replay 代码直接遍历 ``conversation.messages`` 原生列表。
        """
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

        # replay 代码直接遍历 conv.messages（参见 _replay_history_MARKER / _replay_resume_history）
        assert len(conv.messages) == 1
        assert conv.messages[0].subtype == "intent_forecast"
        assert "Fix bug" in str(conv.messages[0].content)

    def test_away_summary_system_message_persists_in_conversation(self):
        """Away Summary（Recapitulate）SystemMessage 追加后可通过 messages 属性读取。"""
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
        """Forecast + Recap system message 经过 Session save/load 后不丢失。"""
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

        # 添加一条 forecast 系统消息
        forecast_result = ForecastResult(
            generated=True,
            suggestions=[
                ForecastSuggestion(id="s1", title="Upgrade deps", prompt="upgrade deps"),
            ],
            fingerprint="fp-rt-001",
        )
        forecast_msg = create_forecast_system_message(forecast_result, trigger="auto")
        conv.messages.append(forecast_msg)

        # 添加一条 away_summary 系统消息
        recap_msg = create_away_summary_message(
            summary="Summary of work done.",
            trigger="auto",
            fingerprint="fp-rt-002",
            message_count=3,
        )
        conv.messages.append(recap_msg)

        # 添加一条普通 user 消息，验证混合场景
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
        # get_messages() 会过滤非 local_command 的 system 消息，直接用 messages 列表
        loaded_msgs = loaded.conversation.messages
        assert len(loaded_msgs) == 3, f"Expected 3 messages, got {len(loaded_msgs)}"

        # 验证 forecast 消息保留
        forecast_found = any(getattr(m, "subtype", None) == "intent_forecast" for m in loaded_msgs)
        assert forecast_found, "Forecast system message lost after session round-trip"

        # 验证 away_summary 消息保留
        recap_found = any(getattr(m, "subtype", None) == "away_summary" for m in loaded_msgs)
        assert recap_found, "Recap system message lost after session round-trip"

        # 验证内容完整
        for m in loaded_msgs:
            if getattr(m, "subtype", None) == "intent_forecast":
                assert "Upgrade deps" in str(getattr(m, "content", ""))
            elif getattr(m, "subtype", None) == "away_summary":
                assert "Summary of work done." in str(getattr(m, "content", ""))

    def test_conversation_mixed_messages_order_preserved(self):
        """System 消息（forecast/recap）与 user/assistant 消息混合时顺序不变。"""
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

        # get_messages() 会过滤非 local_command 的 system 消息，直接用 messages 列表
        roles = [m.role if hasattr(m, "role") else "" for m in conv.messages]
        subtypes = [getattr(m, "subtype", "") for m in conv.messages]

        # 顺序：user / assistant / system(forecast) / user / system(recap) / user
        assert roles == ["user", "assistant", "system", "user", "system", "user"]
        assert subtypes[2] == "intent_forecast"
        assert subtypes[4] == "away_summary"

    def test_replay_resume_history_renders_forecast_content(self):
        """验证 REPL _replay_resume_history 分支能渲染 forecast system 消息。

        不启动完整的 REPL — 验证系统中的 Rending 链关键路径：
        create_forecast_system_message → 消息中包含可渲染的 Markdown 文本。
        """
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
        # _replay_resume_history 通过 msg.content 渲染 Markdown
        assert content.startswith("Forecast")
        # _is_recap_text 不匹配 forecast（只匹配 Recapitulate/Away Summary）
        from clawcodex_ext.repl.core import ClawcodexREPL

        assert not ClawcodexREPL._is_recap_text(content)
        # forecast 在 replay 分支走的是 elif subtype == 'intent_forecast'
        # 它的 subtype 必须是 intent_forecast
        assert getattr(msg, "subtype", None) == "intent_forecast"
