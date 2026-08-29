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

# pylint: disable=no-name-in-module,ungrouped-imports

from __future__ import annotations

from src.agent.conversation import Conversation
from clawcodex_ext.providers.base import ChatResponse
from clawcodex_ext.types.content_blocks import TextBlock
from src.types.messages import Message, normalize_messages_for_api

from clawcodex_ext.away_summary.config import AwaySummaryConfig
from clawcodex_ext.away_summary.fingerprint import last_away_summary_fingerprint
from clawcodex_ext.away_summary.prompt import (
    build_summary_messages,
    infer_response_language,
)
from clawcodex_ext.away_summary.service import (
    AwaySummaryService,
    _normalize_summary_output,
)


class FakeProvider:
    def __init__(
        self,
        content: str = "worked on the feature",
        *,
        reasoning_content: str | None = None,
    ) -> None:
        self.content = content
        self.reasoning_content = reasoning_content
        self.calls: list[dict] = []
        self.model = "fake-model"

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return ChatResponse(
            content=self.content,
            model="fake-model",
            usage={},
            finish_reason="stop",
            reasoning_content=self.reasoning_content,
        )


class FakeSession:
    session_id = "session-1"

    def __init__(self) -> None:
        self.saved = 0

    def save(self) -> None:
        self.saved += 1


def _conversation() -> Conversation:
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="please implement recap"),
        Message(role="assistant", content="I will do that"),
    ]
    return conv


def test_service_generates_and_persists_system_message() -> None:
    conv = _conversation()
    provider = FakeProvider("Summary bullet")
    session = FakeSession()

    result = AwaySummaryService(
        conversation=conv,
        provider=provider,
        model="fake-model",
        session=session,
        config=AwaySummaryConfig(),
    ).generate(trigger="manual")

    assert result.generated is True
    assert "Summary bullet" in result.summary
    assert conv.messages[-1].role == "system"
    assert conv.messages[-1].subtype == "away_summary"
    assert session.saved == 1
    assert provider.calls[0]["tools"] is None


def test_away_summary_does_not_enter_api_messages() -> None:
    conv = _conversation()
    AwaySummaryService(
        conversation=conv,
        provider=FakeProvider(),
        model="fake-model",
        config=AwaySummaryConfig(),
    ).generate(trigger="manual")

    api_messages = normalize_messages_for_api(conv.messages)
    assert api_messages
    assert all("[AWAY SUMMARY]" not in str(m.get("content")) for m in api_messages)


def test_service_suppresses_duplicate_content() -> None:
    conv = _conversation()
    service = AwaySummaryService(
        conversation=conv,
        provider=FakeProvider(),
        model="fake-model",
        config=AwaySummaryConfig(),
    )
    first = service.generate(trigger="manual")
    second = service.generate(trigger="manual")

    assert first.generated is True
    assert second.generated is False
    assert "No new session content" in second.reason
    assert last_away_summary_fingerprint(conv) == first.fingerprint


def test_service_suppresses_duplicate_content_across_triggers() -> None:
    conv = _conversation()
    service = AwaySummaryService(
        conversation=conv,
        provider=FakeProvider(),
        model="fake-model",
        config=AwaySummaryConfig(),
    )
    first = service.generate(trigger="manual")
    second = service.generate(trigger="auto")

    assert first.generated is True
    assert second.generated is False
    assert "No new session content" in second.reason
    assert conv.messages[-1].data["away_summary"]["trigger"] == "manual"


def test_service_min_turns_threshold() -> None:
    conv = Conversation()
    conv.messages = [Message(role="user", content="hello")]

    result = AwaySummaryService(
        conversation=conv,
        provider=FakeProvider(),
        model="fake-model",
        config=AwaySummaryConfig(min_turns=1),
    ).generate(trigger="manual")

    assert result.generated is False
    assert "Not enough conversation" in result.reason


def test_fallback_summary_flattens_content_blocks() -> None:
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="你好"),
        Message(role="assistant", content=[TextBlock(text="你好！看起来我们刚打过招呼。")]),
    ]

    result = AwaySummaryService(
        conversation=conv,
        provider=FakeProvider(""),
        model="fake-model",
        config=AwaySummaryConfig(),
    ).generate(trigger="auto")

    assert result.generated is True
    # Recap should describe the exchange semantically, not surface raw
    # session metadata like "Started with:" or "This session has …".
    assert "Started with:" not in result.summary
    assert "This session has" not in result.summary
    assert "Files mentioned:" not in result.summary
    assert "Actions taken:" not in result.summary
    assert "Latest task:" not in result.summary
    # Must still surface the actual exchange content.
    assert "你好" in result.summary
    # Must not leak SDK internals into the recap.
    assert "TextBlock(" not in result.summary
    assert "type='text'" not in result.summary


def test_fallback_summary_describes_exchanges_semantically() -> None:
    """The LLM-free fallback must read like a natural handoff, not a metadata dump.

    A user who types ``/recap`` after a short greeting exchange should see
    one flowing sentence. When no files were touched and no tools were used,
    the low-value next-step bullet ("Continue with hello") is omitted.
    """
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="Hello! How can I help you today?"),
    ]

    result = AwaySummaryService(
        conversation=conv,
        provider=FakeProvider(""),
        model="fake-model",
        config=AwaySummaryConfig(),
    ).generate(trigger="manual")

    assert result.generated is True
    summary = result.summary
    # English fallback because the user's message is English.
    assert "working on" in summary.lower()
    assert "hello" in summary.lower()
    # No raw session metadata.
    assert "Started with:" not in summary
    assert "This session has" not in summary
    # No fixed labels.
    assert "Current state:" not in summary
    assert "Next step:" not in summary
    # No low-value bullets when there are no files or tools to surface.
    assert "\n- " not in summary


def test_fallback_summary_lists_multiple_user_requests() -> None:
    """The fallback surfaces the most recent user request as the goal and the
    most recent assistant reply as the current task state — but it does NOT
    enumerate every prior turn (that would balloon past the readable budget
    and leak process).
    """
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="Please implement the recap feature."),
        Message(role="assistant", content="Sure, working on it now."),
        Message(role="user", content="Also add a /recap command."),
        Message(role="assistant", content="Got it."),
    ]

    result = AwaySummaryService(
        conversation=conv,
        provider=FakeProvider(""),
        model="fake-model",
        config=AwaySummaryConfig(),
    ).generate(trigger="manual")

    assert result.generated is True
    summary = result.summary
    # English fallback because all messages are English.
    # The LAST user request is the current goal.
    assert "add a /recap command" in summary
    # The LAST assistant reply is the current task state.
    assert "Got it" in summary
    # Earlier turns should NOT be surfaced (would blow the budget).
    assert "implement the recap" not in summary
    # No fixed labels.
    assert "Current state:" not in summary
    assert "Next step:" not in summary


def test_fallback_summary_handles_empty_session() -> None:
    """When there is no user or assistant content, surface a clear empty-state.

    Tested directly against ``_fallback_summary`` because the public
    ``generate()`` path is gated by ``min_turns`` and would short-circuit
    before the fallback ever runs.
    """
    from clawcodex_ext.away_summary.service import _fallback_summary

    conv = Conversation()
    conv.messages = []

    summary = _fallback_summary(conv)
    assert "nothing to recap" in summary.lower() or "刚开始" in summary


def test_fallback_summary_mentions_tools_used() -> None:
    """Tool actions performed by the assistant should appear as plain bullet
    lines in the fallback recap, without fixed labels.
    """
    from clawcodex_ext.away_summary.service import _fallback_summary

    conv = Conversation()
    conv.messages = [
        Message(role="user", content="read /tmp/data.csv"),
        Message(
            role="assistant",
            content=[
                {"type": "text", "text": "Reading the file now."},
                {"type": "tool_use", "name": "Read", "input": {"file_path": "/tmp/data.csv"}},
            ],
        ),
    ]

    summary = _fallback_summary(conv)
    # Sentence summary still surfaces the user request.
    assert "read /tmp/data.csv" in summary
    # No fixed labels.
    assert "Files mentioned:" not in summary
    assert "Actions taken:" not in summary
    assert "Current state:" not in summary
    assert "Next step:" not in summary
    # Plain bullets surface the file and the tool action.
    assert "\n- /tmp/data.csv" in summary
    assert "\n- Read(/tmp/data.csv)" in summary
    assert "\n- Continue with" in summary


def test_fallback_summary_omits_labels_without_tool_calls() -> None:
    """Plain conversation (no tool_use blocks) should produce only the
    sentence summary, with no fixed labels and no low-value next-step bullet.
    """
    from clawcodex_ext.away_summary.service import _fallback_summary

    conv = Conversation()
    conv.messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="Hi! How can I help?"),
    ]

    summary = _fallback_summary(conv)
    # Sentence summary still surfaces the exchange.
    assert "hello" in summary.lower()
    assert "working on" in summary.lower()
    # No bullets when there is nothing concrete to surface.
    assert "\n- " not in summary
    assert "Current state:" not in summary
    assert "Next step:" not in summary
    assert "Files touched:" not in summary
    assert "Actions taken:" not in summary


def test_fallback_summary_separates_sentence_and_labels_with_blank_line() -> None:
    """The fallback is one sentence followed by bullet lines, each separated
    by a newline. With tool calls, extra detail bullets appear before the
    next-step bullet.
    """
    from clawcodex_ext.away_summary.service import _fallback_summary

    conv = Conversation()
    conv.messages = [
        Message(role="user", content="refactor the recap feature"),
        Message(
            role="assistant",
            content=[
                {"type": "text", "text": "On it."},
                {"type": "tool_use", "name": "Read", "input": {"file_path": "/tmp/x.py"}},
                {"type": "tool_use", "name": "Edit", "input": {"file_path": "/tmp/y.py"}},
            ],
        ),
    ]

    summary = _fallback_summary(conv)
    lines = summary.split("\n")
    # Sentence line first.
    assert "working on" in lines[0].lower()
    # Plain bullets for files, actions, and next step.
    assert "\n- /tmp/x.py, /tmp/y.py" in summary
    assert "\n- Read(/tmp/x.py), Edit(/tmp/y.py)" in summary
    assert "\n- Continue with" in summary
    # No fixed labels.
    assert "Files touched:" not in summary
    assert "Actions taken:" not in summary


def test_fallback_summary_chinese_labels() -> None:
    """Chinese sessions get Chinese sentence phrasing and Chinese middle-dot
    bullet markers — with no English fixed labels.
    """
    from clawcodex_ext.away_summary.service import _fallback_summary

    conv = Conversation()
    conv.messages = [
        Message(role="user", content="帮我读一下源码"),
        Message(
            role="assistant",
            content=[
                {"type": "text", "text": "好的。"},
                {"type": "tool_use", "name": "Read", "input": {"file_path": "/tmp/x.py"}},
            ],
        ),
    ]

    summary = _fallback_summary(conv)
    # Chinese sentence + ASCII hyphen bullets.
    assert "我们正在处理" in summary
    assert "- /tmp/x.py" in summary
    assert "- Read(/tmp/x.py)" in summary
    assert "- 继续" in summary
    # English label headers must NOT appear in a Chinese session.
    assert "Current state:" not in summary
    assert "Next step:" not in summary
    assert "Files mentioned:" not in summary
    assert "Actions taken:" not in summary
    # Chinese middle-dot should no longer be used.
    assert "·" not in summary


def test_fallback_summary_caps_label_lists() -> None:
    """The detail bullet lists must never balloon past a readable budget — when
    more than 4 files/actions happened, only the first 4 are shown and a
    trailing ``…`` marker is appended.
    """
    from clawcodex_ext.away_summary.service import _fallback_summary

    tool_uses = [{"type": "tool_use", "name": "Read", "input": {"file_path": f"/tmp/f{i}.py"}} for i in range(10)]
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="scan files"),
        Message(role="assistant", content=[{"type": "text", "text": "scanning"}] + tool_uses),
    ]

    summary = _fallback_summary(conv)
    # 4 files listed, then a marker — not all 10.
    assert "/tmp/f3.py" in summary
    assert "/tmp/f4.py" not in summary
    assert "…" in summary


def test_infer_language_from_chinese_user_message() -> None:
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="请实现一个递归函数"),
        Message(role="assistant", content="好的，以下是递归函数的实现。"),
    ]
    assert infer_response_language(conv) == "Chinese"


def test_infer_language_falls_back_to_assistant_when_user_is_code_heavy() -> None:
    """User turns dominated by English identifiers should not drown out Chinese intent."""
    conv = Conversation()
    conv.messages = [
        Message(
            role="user",
            content="clawcodex_ext/away_summary/prompt.py 的 infer_response_language 改成中文",
        ),
        Message(role="assistant", content="已修改。现在函数会正确检测中文。"),
    ]
    assert infer_response_language(conv) == "Chinese"


def test_infer_language_english_conversation() -> None:
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="fix bug in test_service.py"),
        Message(role="assistant", content="Fixed the bug in test_service.py."),
    ]
    assert infer_response_language(conv) == "English"


def test_infer_language_explicit_override() -> None:
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="fix bug in test_service.py"),
        Message(role="assistant", content="Fixed the bug in test_service.py."),
    ]
    assert infer_response_language(conv, explicit="Chinese") == "Chinese"
    assert infer_response_language(conv, explicit="English") == "English"


def test_summary_prompt_includes_must_language_instruction() -> None:
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="请帮我修改代码"),
        Message(role="assistant", content="好的，我来帮你。"),
    ]
    prompt = build_summary_messages(conv, max_input_tokens=4_000)[0]["content"]
    assert "MUST write the recap in natural Simplified Chinese" in prompt
    assert "MUST be written in the language specified above" in prompt
    assert "Do not switch languages mid-recap" in prompt


def test_summary_prompt_forbids_thinking_preamble() -> None:
    """The Away Summary prompt must explicitly forbid thinking-scaffold leakage.

    A Sapiens AI / Agnes-2.0-Flash style provider was emitting a free-form
    "Here's a thinking process: 1 Analyze 2 Identify 3 Draft Recap 4 Check
    Constraints …" block inside ``content`` instead of the structured
    ``reasoning_content`` field, which previously leaked into the recap.
    The prompt now forbids that scaffold outright.
    """
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="你是谁"),
        Message(role="assistant", content="我是 Agnes-2.0-Flash"),
    ]
    prompt = build_summary_messages(conv, max_input_tokens=4_000)[0]["content"]
    assert "thinking process" in prompt.lower()
    assert "do not output any internal chain-of-thought" in prompt.lower()
    assert "<think>" in prompt


def test_summary_prompt_requires_goal_plus_next_action() -> None:
    """The recap prompt must ask for the goal+state+next-step dimensions
    that the Claude Code /recap canonical implementation uses, while
    forbidding fixed labels so the model can phrase things naturally.
    """
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="请帮我修改代码"),
        Message(role="assistant", content="好的，我来帮你。"),
    ]
    prompt = build_summary_messages(conv, max_input_tokens=4_000)[0]["content"]
    lowered = prompt.lower()
    # 1-2 sentence budget (now described as "short, flowing sentences").
    assert "1-2 short, flowing sentences" in lowered or "1-2 句" in prompt
    # Plain text only, no headings / no bold.
    assert "no headings" in lowered
    assert "no bold" in lowered
    # Goal + current state + next step are covered as guidance.
    assert "high-level goal" in lowered
    assert "where they left off" in lowered
    assert "next action" in lowered
    # Fixed labels are explicitly forbidden.
    assert "fixed section labels" in lowered
    assert "current state:" not in lowered
    assert "next step:" not in lowered
    assert "files mentioned:" not in lowered
    assert "actions taken:" not in lowered


def test_service_strips_thinking_preamble_from_content() -> None:
    """A free-form 'Here's a thinking process:\\n\\n…' preamble inside
    ``content`` must be stripped before the recap is surfaced. When the
    remaining body still looks like a multi-chapter CoT (1 Analyze / 2
    Identify / 3 Draft Recap / 4 Check …) we deliberately fall back to
    the conversation-derived summary rather than surface any of the
    leaked scaffolding.
    """
    leaked = (
        "Here's a thinking process:\n\n"
        " 1 Analyze User Input:\n\n"
        " • Task: Write a short recap\n"
        " • Constraints: 3-6 bullets\n\n"
        " 2 Identify Key Information:\n\n"
        " • The session just started.\n\n"
        " 3 Draft Recap:\n\n"
        " • 会话刚刚开启，目前处于初始问候与身份确认阶段。\n"
        " • 用户仅进行了基础打招呼并询问了助手身份。\n"
        " • 下一步建议：直接说明需要实现的功能。\n"
    )
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="你是谁"),
        Message(role="assistant", content="我是 Agnes"),
    ]
    result = AwaySummaryService(
        conversation=conv,
        provider=FakeProvider(leaked),
        model="fake-model",
        config=AwaySummaryConfig(),
    ).generate(trigger="manual")

    assert result.generated is True
    lowered = result.summary.lower()
    assert "thinking process" not in lowered
    assert "analyze user input" not in lowered
    assert "draft recap" not in lowered
    assert "identify key information" not in lowered
    # We fell back to the conversation-derived recap, not the CoT body.
    assert "你是谁" in result.summary


def test_service_keeps_clean_recap_when_thinking_preamble_present() -> None:
    """Counter-test: a single numbered chapter heading that happens to be
    preceded by a "thinking process" preamble must be allowed to pass
    through, since that's a possible shape for a normally-recap response
    that includes one bullet section.
    """
    leaked = (
        "Here's a thinking process:\n\n"
        "- 会话刚刚开启，目前处于初始问候与身份确认阶段。\n"
        "- 用户仅进行了基础打招呼并询问了助手身份。\n"
        "- 下一步建议：直接说明需要实现的功能。\n"
    )
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="你是谁"),
        Message(role="assistant", content="我是 Agnes"),
    ]
    result = AwaySummaryService(
        conversation=conv,
        provider=FakeProvider(leaked),
        model="fake-model",
        config=AwaySummaryConfig(),
    ).generate(trigger="manual")

    assert result.generated is True
    assert "thinking process" not in result.summary.lower()
    assert "会话刚刚开启" in result.summary
    assert "下一步建议" in result.summary


def test_service_strips_think_xml_envelopes_from_content() -> None:
    """Models that wrap their answer in <think>…</think> should drop the
    envelope while keeping the recap body.
    """
    leaked = (
        "<think>The user greeted me and asked who I am. The session is in its "
        "opening moments. Just summarise that.</think>\n"
        "- 会话刚刚开启，目前处于初始问候阶段。\n"
        "- 用户询问了助手的身份。\n"
        "- 下一步建议：直接说明需要实现的功能。\n"
        "</think>"  # trailing tag without content — must be cleaned up too
    )
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="你是谁"),
        Message(role="assistant", content="我是 Agnes"),
    ]
    result = AwaySummaryService(
        conversation=conv,
        provider=FakeProvider(leaked),
        model="fake-model",
        config=AwaySummaryConfig(),
    ).generate(trigger="manual")

    assert result.generated is True
    assert "<think>" not in result.summary.lower()
    assert "会话刚刚开启" in result.summary
    assert "下一步建议" in result.summary


def test_service_does_not_leak_reasoning_content() -> None:
    """``reasoning_content`` must NEVER be used as a recap — it's an
    internal chain-of-thought, equivalent to the cached stream in
    ``clawcodex_ext/query/query.py``. When the model returns empty
    ``content`` with reasoning populated, fall back to the
    conversation-derived summary.
    """
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="你是谁"),
        Message(role="assistant", content="我是 Agnes-2.0-Flash"),
    ]
    provider = FakeProvider(content="", reasoning_content="internal CoT — never show this")
    result = AwaySummaryService(
        conversation=conv,
        provider=provider,
        model="fake-model",
        config=AwaySummaryConfig(),
    ).generate(trigger="manual")

    assert result.generated is True
    assert "internal CoT" not in result.summary
    assert "never show this" not in result.summary
    assert "你是谁" in result.summary  # we got the fallback summary


def test_service_falls_back_when_content_is_full_cot_transcript() -> None:
    """Some providers emit the entire CoT transcript inside ``content``
    (no preamble, no XML tags). The post-clean CoT-hallmark heuristic
    should detect that and surrender to the conversation-derived summary.
    """
    transcript = (
        "1 Analyze User Input:\n\n"
        " • Task: Write a recap\n"
        " • Constraints: 3-6 bullets\n\n"
        "2 Identify Key Information:\n\n"
        " • The session just started.\n\n"
        "3 Draft Recap (Mental Refinement in Simplified Chinese):\n\n"
        " • 会话刚刚开启。\n\n"
        "4 Check Constraints:\n\n"
        " • Concise recap? Yes.\n"
        " • 3-6 bullets? I have 5 bullets.\n"
        " • Focus areas covered? Yes.\n"
        " • Language: Natural Simplified Chinese? Yes.\n"
        " • No hidden reasoning\n"
    )
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="你是谁"),
        Message(role="assistant", content="我是 Agnes"),
    ]
    result = AwaySummaryService(
        conversation=conv,
        provider=FakeProvider(transcript),
        model="fake-model",
        config=AwaySummaryConfig(),
    ).generate(trigger="manual")

    assert result.generated is True
    lowered = result.summary.lower()
    assert "check constraints" not in lowered
    assert "draft recap (mental refinement" not in lowered
    assert "no hidden reasoning" not in lowered
    # We get the conversation-derived fallback instead of the CoT.
    assert "你是谁" in result.summary


def test_normalize_summary_output_strips_chinese_preamble() -> None:
    """Strip a Chinese recap meta-intro even when the model emits one."""
    raw = "你刚回来，这是之前的会话摘要：\n用户与助手进行了简单的问候。\n- 继续聊天"
    cleaned = _normalize_summary_output(raw)
    assert "你刚回来" not in cleaned
    assert "这是之前的会话摘要" not in cleaned
    assert cleaned.startswith("用户与助手进行了简单的问候。")


def test_normalize_summary_output_replaces_non_hyphen_bullets() -> None:
    """The prompt requires '-' bullets, but models sometimes emit '•', '*',
    '·', or numbered bullets. Normalize them all to '-'.
    """
    raw = "我们刚聊到 greeting。\n• 文件 A\n* 工具 B\n· 继续\n1. 下一步"
    cleaned = _normalize_summary_output(raw)
    assert "•" not in cleaned
    assert "* " not in cleaned
    assert "·" not in cleaned
    assert "1." not in cleaned
    assert "- 文件 A" in cleaned
    assert "- 工具 B" in cleaned
    assert "- 继续" in cleaned
    assert "- 下一步" in cleaned


def test_normalize_summary_output_drops_low_value_greeting_bullets() -> None:
    """Drop bullets that contain only a low-value greeting."""
    raw = "用户与助手打了个招呼。\n- 问候\n- 继续聊天"
    cleaned = _normalize_summary_output(raw)
    assert "- 问候" not in cleaned
    assert "- 继续聊天" in cleaned


def test_normalize_summary_output_strips_inline_markdown() -> None:
    """The prompt forbids markdown beyond bullet markers; strip bold,
    italic, inline code, and heading markers while preserving the text.
    """
    raw = (
        "我们正在处理 recap。\n"
        "## 更新\n"
        "- **当前分支** `feature/example` 上有改动\n"
        "- _最新提交_ 是 asciicast 录制器\n"
        "- 下一步继续"
    )
    cleaned = _normalize_summary_output(raw)
    assert "##" not in cleaned
    assert "**" not in cleaned
    assert "`" not in cleaned
    assert "_最新提交_" not in cleaned
    assert "当前分支 feature/example 上有改动" in cleaned
    assert "最新提交 是 asciicast 录制器" in cleaned
    assert "- 下一步继续" in cleaned


def test_service_normalizes_model_noncompliant_output() -> None:
    """End-to-end: a model that ignores the prompt and emits a preamble plus
    '•' bullets still yields a clean, normalized recap.
    """
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi there"),
    ]
    result = AwaySummaryService(
        conversation=conv,
        provider=FakeProvider("你刚回来，这是之前的会话摘要：\n用户与助手进行了简单的问候。\n• 问候\n- 继续聊天"),
        model="fake-model",
        config=AwaySummaryConfig(),
    ).generate(trigger="manual")

    assert result.generated is True
    assert "你刚回来" not in result.summary
    assert "这是之前的会话摘要" not in result.summary
    assert "•" not in result.summary
    assert "- 继续聊天" in result.summary
    # The low-value greeting bullet is dropped.
    assert "- 问候" not in result.summary
