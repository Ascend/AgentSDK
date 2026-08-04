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

# pylint: disable=no-name-in-module
from __future__ import annotations

from src.agent.conversation import Conversation
from src.types.messages import Message

import pytest

from clawcodex_ext.away_summary import prompt
from clawcodex_ext.away_summary.prompt import (
    AWAY_SUMMARY_INSTRUCTIONS,
    AWAY_SUMMARY_INSTRUCTIONS_AUTO,
    _estimate_tokens,
    _truncate_transcript,
    build_summary_messages,
)


def _conv_with_mixed_messages() -> Conversation:
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="Please implement the recap feature."),
        Message(role="assistant", content="Sure, working on it now."),
    ]
    return conv


def test_manual_trigger_uses_strict_three_part_template() -> None:
    """``/recap`` (manual) keeps the goal+state+next-step guidance, but
    forbids fixed labels so the model can phrase things naturally.
    """
    messages = build_summary_messages(
        _conv_with_mixed_messages(),
        max_input_tokens=4_000,
        trigger="manual",
    )
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert system == AWAY_SUMMARY_INSTRUCTIONS.format(language_instruction="MUST write the recap in natural English.")
    # Relaxed from "1-2 plain sentences" to "1-2 short, flowing sentences"
    assert "1-2 short, flowing sentences" in system
    assert "high-level goal" in system.lower()
    # Bullets are requested, but fixed labels are forbidden.
    assert "bullet list" in system.lower()
    assert "fixed section labels" in system.lower()
    assert "current state:" not in system.lower()
    assert "next step:" not in system.lower()
    assert "Return only the recap" in user


def test_auto_trigger_uses_relaxed_three_sentence_template() -> None:
    """The idle "while you were away" card allows 1-3 flowing sentences plus
    a short bullet list, and explicitly invites weaving in session memory
    when supplied.
    """
    messages = build_summary_messages(
        _conv_with_mixed_messages(),
        max_input_tokens=4_000,
        trigger="auto",
    )
    system = messages[0]["content"]

    assert system == AWAY_SUMMARY_INSTRUCTIONS_AUTO.format(
        language_instruction="MUST write the recap in natural English."
    )
    assert "1-3 short, flowing sentences" in system
    assert "80 words" in system.lower()
    assert "broader session memory" in system.lower()
    # Bullets are requested, but fixed labels are forbidden.
    assert "bullet list" in system.lower()
    assert "fixed section labels" in system.lower()
    assert "current state:" not in system.lower()
    assert "next step:" not in system.lower()
    # Three-part strict ordering must NOT be required for auto.
    assert "in this order" not in system.lower()


def test_auto_user_prompt_does_not_mention_memory_when_absent() -> None:
    """Without memory the auto user prompt stays simple — no dangling marker."""
    messages = build_summary_messages(
        _conv_with_mixed_messages(),
        max_input_tokens=4_000,
        trigger="auto",
        memory=None,
    )
    user = messages[1]["content"]
    assert "Session memory (broader context)" not in user
    assert "Session transcript:" in user
    assert "1-3 short sentences" in user
    assert "1-3 `-` bullets" in user


def test_auto_user_prompt_includes_memory_when_provided() -> None:
    """Memory is injected as a separate block the recap can weave in."""
    messages = build_summary_messages(
        _conv_with_mixed_messages(),
        max_input_tokens=4_000,
        trigger="auto",
        memory="Title: Refactor recap\nGoals:\n- align wording",
    )
    user = messages[1]["content"]
    assert "Session memory (broader context):\nTitle: Refactor recap" in user
    assert "Session transcript:" in user
    # Memory is positioned BEFORE the transcript so the model sees the
    # broader context first.
    assert user.index("Session memory") < user.index("Session transcript")


def test_prompt_contract_has_one_intent_phrase_before_bullets() -> None:
    messages = build_summary_messages(
        _conv_with_mixed_messages(),
        max_input_tokens=4_000,
        trigger="manual",
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "\n后续计划\n- tui/screens/repl.py" in system
    assert "\n- 后续计划\n" not in system
    assert "one intent phrase" in user
    assert "no markdown" not in user.lower()


def test_manual_user_prompt_ignores_memory() -> None:
    """``/recap`` is a focused, transcript-only view — memory is dropped."""
    messages = build_summary_messages(
        _conv_with_mixed_messages(),
        max_input_tokens=4_000,
        trigger="manual",
        memory="Title: should-not-appear",
    )
    user = messages[1]["content"]
    assert "Title: should-not-appear" not in user


def test_unknown_trigger_falls_back_to_manual() -> None:
    """Callers that pass an unrecognised trigger get the manual shape,
    matching the docstring's ``"manual"`` default behaviour.
    """
    messages = build_summary_messages(
        _conv_with_mixed_messages(),
        max_input_tokens=4_000,
        trigger="totally-unknown",
    )
    assert messages[0]["content"] == AWAY_SUMMARY_INSTRUCTIONS.format(
        language_instruction="MUST write the recap in natural English."
    )


def test_chinese_session_uses_chinese_language_instruction() -> None:
    """The relaxed (auto) template still respects the Chinese language gate."""
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="帮我实现 recap 功能"),
        Message(role="assistant", content="好的，我来实现。"),
    ]
    messages = build_summary_messages(conv, max_input_tokens=4_000, trigger="auto")
    system = messages[0]["content"]
    assert "MUST write the recap in natural Simplified Chinese" in system


def test_auto_template_forbids_thinking_preamble() -> None:
    """The relaxed auto template keeps the same anti-CoT guards as manual."""
    messages = build_summary_messages(
        _conv_with_mixed_messages(),
        max_input_tokens=4_000,
        trigger="auto",
    )
    system = messages[0]["content"].lower()
    assert "thinking process" in system
    assert "do not output any internal chain-of-thought" in system
    assert "<think>" in messages[0]["content"]


def test_prompt_forbids_preamble_and_requires_hyphen_bullets() -> None:
    """The recap prompt must explicitly forbid model-added meta-intros and
    require the ASCII hyphen as the only bullet marker.
    """
    system = build_summary_messages(
        _conv_with_mixed_messages(),
        max_input_tokens=4_000,
        trigger="auto",
    )[0]["content"]
    lowered = system.lower()
    assert "你刚回来，这是之前的会话摘要" in system
    assert "do not start the recap with a preamble" in lowered
    assert "use `-` and only `-` as the bullet marker" in lowered
    assert "do not use `•`" in lowered


def test_missing_fingerprint_module_degrades_to_regular_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "clawcodex_ext.away_summary.fingerprint"

    def missing(_name: str):
        exc = ModuleNotFoundError(f"No module named {module_name!r}")
        exc.name = module_name
        raise exc

    monkeypatch.setattr(prompt, "import_module", missing)

    assert prompt.is_away_summary_message(Message(role="user", content="hello")) is False


def test_fingerprint_import_reraises_unrelated_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(_name: str):
        exc = ModuleNotFoundError("No module named 'hash_backend'")
        exc.name = "hash_backend"
        raise exc

    monkeypatch.setattr(prompt, "import_module", missing_dependency)

    with pytest.raises(ModuleNotFoundError, match="hash_backend"):
        prompt.is_away_summary_message(Message(role="user", content="hello"))


def test_transcript_truncation_respects_token_budget() -> None:
    truncated = _truncate_transcript("word " * 2_000, max_input_tokens=256)
    assert _estimate_tokens(truncated) <= 256
    assert "earlier middle of session omitted" in truncated


def test_transcript_truncation_converts_tokens_to_chars_when_marker_is_too_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prompt,
        "_estimate_tokens",
        lambda text: max(1, (len(text) + 3) // 4),
    )

    truncated = _truncate_transcript("x" * 1_000, max_input_tokens=8)

    assert len(truncated) == 32
    assert prompt._estimate_tokens(truncated) <= 8
