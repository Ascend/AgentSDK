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

"""Token counter tests.

Covers the pluggable TokenCounter chain: char, heuristic, tiktoken,
and the FallbackTokenCounter that tries each in order.
"""

from __future__ import annotations

import pytest

from src.services.context_collapse.exceptions import TokenCountUnavailableError
from src.services.context_collapse.tokens import (
    CharTokenCounter,
    FallbackTokenCounter,
    HeuristicTokenCounter,
    TiktokenCounter,
    TokenCounter,
    TokenEstimate,
    heuristic_only,
    tiktoken_first_then_heuristic,
)


# ---------------------------------------------------------------------------
# CharTokenCounter
# ---------------------------------------------------------------------------


def test_char_counter_minimum_one_for_short_input() -> None:
    c = CharTokenCounter()
    assert c.count("") >= 1  # implementation returns max(1, 0) == 1
    assert c.count("a") == 1
    assert c.count("abcd") == 1


def test_char_counter_scales_with_length() -> None:
    c = CharTokenCounter()
    assert c.count("a" * 40) == 10
    assert c.count("a" * 100) == 25


def test_char_counter_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        CharTokenCounter().count(b"bytes")  # type: ignore[arg-type]


def test_char_counter_counts_dict_messages() -> None:
    c = CharTokenCounter()
    msgs = [
        {"role": "user", "content": "x" * 40},
        {"role": "assistant", "content": "y" * 80},
    ]
    # 40//4 + 80//4 = 10 + 20 = 30, plus 4*2 overhead = 38
    assert c.count_messages(msgs) == 38


def test_char_counter_counts_structured_messages() -> None:
    c = CharTokenCounter()

    class FakeMsg:
        def __init__(self, text: str) -> None:
            self.content = [type("B", (), {"text": text})()]

    msgs = [FakeMsg("hello world"), FakeMsg("foo")]
    # "hello world" -> 11//4 = 2; "foo" -> 3//4 = 0 -> max(1,0) = 1
    # Plus 4 overhead per message = 2
    assert c.count_messages(msgs) == 2 + 1 + 4 * 2


# ---------------------------------------------------------------------------
# HeuristicTokenCounter
# ---------------------------------------------------------------------------


def test_heuristic_counter_empty_returns_zero() -> None:
    assert HeuristicTokenCounter().count("") == 0


def test_heuristic_counter_word_count_times_one_point_three() -> None:
    c = HeuristicTokenCounter()
    # 4 words -> round(4 * 1.3) = 5
    assert c.count("one two three four") == 5
    # 10 words -> round(10 * 1.3) = 13
    assert c.count(" ".join(["w"] * 10)) == 13


def test_heuristic_counter_minimum_one_for_nonempty() -> None:
    # 1 word -> round(1.3) = 1
    assert HeuristicTokenCounter().count("hi") == 1


def test_heuristic_counter_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        HeuristicTokenCounter().count(42)  # type: ignore[arg-type]


def test_heuristic_counter_message_count_includes_overhead() -> None:
    c = HeuristicTokenCounter()
    msgs = [
        {"role": "user", "content": "alpha beta gamma"},  # 3 words => 4
        {"role": "assistant", "content": "x"},  # 1 word => 1
    ]
    # (4 + 1) tokens + 4*2 overhead = 13
    assert c.count_messages(msgs) == 13


# ---------------------------------------------------------------------------
# TiktokenCounter (lazy / optional)
# ---------------------------------------------------------------------------


def test_tiktoken_counter_requires_model_or_encoding() -> None:
    with pytest.raises(ValueError):
        TiktokenCounter()


def test_tiktoken_counter_rejects_non_string() -> None:
    c = TiktokenCounter(encoding_name="cl100k_base")
    with pytest.raises(TypeError):
        c.count(123)  # type: ignore[arg-type]


def test_tiktoken_counter_name_reports_encoding_or_model() -> None:
    assert TiktokenCounter(encoding_name="cl100k_base").name == "tiktoken:cl100k_base"
    assert TiktokenCounter(model="gpt-4").name == "tiktoken:gpt-4"


def test_tiktoken_counter_raises_unavailable_when_missing(monkeypatch) -> None:
    """When tiktoken is not importable, raise TokenCountUnavailableError."""
    c = TiktokenCounter(encoding_name="cl100k_base")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken" or name.startswith("tiktoken"):
            raise ImportError("simulated missing tiktoken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(TokenCountUnavailableError):
        c.count("hello")


# ---------------------------------------------------------------------------
# FallbackTokenCounter chain
# ---------------------------------------------------------------------------


def test_fallback_requires_at_least_one_counter() -> None:
    with pytest.raises(ValueError):
        FallbackTokenCounter([])


def test_fallback_uses_first_working_counter() -> None:
    chain = FallbackTokenCounter(
        [
            _RaisingCounter("broken", TokenCountUnavailableError("nope")),
            HeuristicTokenCounter(),
        ]
    )
    est = chain.estimate("hello world")
    assert est.counter_name == "heuristic"
    assert est.tokens >= 1


def test_fallback_raises_when_all_counters_fail() -> None:
    chain = FallbackTokenCounter(
        [
            _RaisingCounter("a", TokenCountUnavailableError("x")),
            _RaisingCounter("b", RuntimeError("y")),
        ]
    )
    with pytest.raises(TokenCountUnavailableError):
        chain.estimate("anything")


def test_fallback_probe_skips_broken_counter_even_with_other_exception() -> None:
    chain = FallbackTokenCounter(
        [
            _RaisingCounter("a", ValueError("bad input")),
            HeuristicTokenCounter(),
        ]
    )
    est = chain.estimate("hello")
    assert est.counter_name == "heuristic"


def test_fallback_add_counter_inserts_at_front() -> None:
    base = FallbackTokenCounter([HeuristicTokenCounter()])
    # First probe will fail, forcing fallback to heuristic.
    base.add_counter(_RaisingCounter("front", TokenCountUnavailableError("x")))
    est = base.estimate("hello")
    # front was inserted at front; it raised and we fell back to heuristic
    assert est.counter_name == "heuristic"


def test_fallback_estimate_messages_uses_first_working() -> None:
    chain = FallbackTokenCounter([HeuristicTokenCounter()])
    est = chain.estimate_messages([{"role": "user", "content": "hello world"}])
    assert est.counter_name == "heuristic"
    assert est.tokens > 0


# ---------------------------------------------------------------------------
# Pre-built chain helpers
# ---------------------------------------------------------------------------


def test_heuristic_only_chain_never_imports_tiktoken() -> None:
    chain = heuristic_only()
    # Should succeed without tiktoken installed.
    assert chain.estimate("hello").tokens >= 1
    assert chain.estimate("a b c d").tokens >= 1


def test_tiktoken_first_chain_handles_missing_tiktoken(monkeypatch) -> None:
    chain = tiktoken_first_then_heuristic(encoding_name="cl100k_base")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken" or name.startswith("tiktoken"):
            raise ImportError("no tiktoken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Should fall back to heuristic without raising.
    est = chain.estimate("hello world")
    assert est.counter_name == "heuristic"


# ---------------------------------------------------------------------------
# Protocol satisfaction (runtime_checkable)
# ---------------------------------------------------------------------------


def test_concrete_counters_satisfy_protocol() -> None:
    for counter in (CharTokenCounter(), HeuristicTokenCounter()):
        assert isinstance(counter, TokenCounter)


def test_token_estimate_is_frozen() -> None:
    est = TokenEstimate(tokens=10, counter_name="heuristic")
    with pytest.raises(Exception):  # FrozenInstanceError is a subclass of AttributeError
        est.tokens = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RaisingCounter:
    """Test double: every count() call raises ``exc``."""

    def __init__(self, name: str, exc: Exception) -> None:
        self.name = name
        self._exc = exc

    def count(self, text: str) -> int:
        raise self._exc

    def count_messages(self, messages) -> int:
        raise self._exc
