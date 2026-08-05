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

"""Summary generator tests.

Covers the deterministic HeadlineSummaryGenerator and the async
LLMSummaryGenerator that delegates to an injected callback with a
timeout + headline fallback.
"""
# pylint: disable=use-implicit-booleaness-not-comparison

from __future__ import annotations

import asyncio

import pytest

from src.services.context_collapse.exceptions import SummaryGeneratorError
from src.services.context_collapse.summary import (
    HeadlineSummaryGenerator,
    LLMSummaryGenerator,
    SummaryGenerator,
    count_words,
    extract_text,
)


# ---------------------------------------------------------------------------
# HeadlineSummaryGenerator
# ---------------------------------------------------------------------------


def test_headline_generator_returns_empty_marker_for_no_messages() -> None:
    out = HeadlineSummaryGenerator().summarize([])
    assert out == "(empty archive)"


def test_headline_generator_one_line_per_message() -> None:
    gen = HeadlineSummaryGenerator()
    msgs = [
        {"role": "user", "content": "firstword rest of message"},
        {"role": "assistant", "content": "secondword rest of message"},
    ]
    out = gen.summarize(msgs)
    assert "[2 archived message(s)]" in out
    # The headline keeps the first whitespace-delimited token only.
    assert "1. firstword" in out
    assert "2. secondword" in out


def test_headline_generator_marks_empty_messages() -> None:
    gen = HeadlineSummaryGenerator()
    out = gen.summarize([{"role": "user", "content": ""}])
    assert "1. (empty)" in out


def test_headline_generator_truncates_headline_to_200_chars() -> None:
    gen = HeadlineSummaryGenerator()
    huge = "x" * 500
    out = gen.summarize([{"role": "user", "content": huge}])
    # 200 chars of headline + "1. " prefix + " " separator; ensure no 500-char run
    assert "x" * 250 not in out
    assert "x" * 200 in out


def test_headline_generator_handles_structured_content_blocks() -> None:
    gen = HeadlineSummaryGenerator()
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "alpha line"},
            {"type": "text", "text": "beta line"},
        ],
    }
    out = gen.summarize([msg])
    # The first non-blank line is "alpha line beta line" (joined with \n)
    # so the headline should at least include "alpha"
    assert "alpha" in out


def test_headline_generator_reports_dropped_lines() -> None:
    gen = HeadlineSummaryGenerator()
    msgs = [
        {"role": "user", "content": "headline\nbody line one\nbody line two"},
        {"role": "user", "content": "headline two\nmore body"},
    ]
    out = gen.summarize(msgs)
    assert "[~" in out and "non-headline line(s) elided]" in out


def test_headline_generator_reports_archived_overflow() -> None:
    gen = HeadlineSummaryGenerator(max_headlines=2)
    msgs = [
        {"role": "user", "content": "m1"},
        {"role": "user", "content": "m2"},
        {"role": "user", "content": "m3"},
        {"role": "user", "content": "m4"},
    ]
    out = gen.summarize(msgs)
    assert "[4 archived message(s)]" in out
    assert "(2 additional message(s) archived)" in out


def test_headline_generator_truncates_long_output() -> None:
    gen = HeadlineSummaryGenerator(max_chars=200)
    msgs = [{"role": "user", "content": "x" * 100} for _ in range(40)]
    out = gen.summarize(msgs)
    assert "[…truncated]" in out


def test_headline_generator_thread_safe() -> None:
    """Many threads concurrently calling summarize() should not corrupt output."""
    import threading

    gen = HeadlineSummaryGenerator()
    msgs = [{"role": "user", "content": f"message {i}"} for i in range(20)]
    results: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        out = gen.summarize(msgs)
        with lock:
            results.append(out)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 20
    # All outputs should be identical (deterministic) and non-empty.
    assert all(r == results[0] for r in results)
    assert results[0]


# ---------------------------------------------------------------------------
# LLMSummaryGenerator (async path + sync fallback)
# ---------------------------------------------------------------------------


def test_llm_summary_generator_sync_path_uses_fallback() -> None:
    """The synchronous summarize() always uses the fallback (no event loop)."""
    called = []

    async def fn(msgs):
        called.append(msgs)
        return "LLM summary"

    gen = LLMSummaryGenerator(fn)
    out = gen.summarize([{"role": "user", "content": "x"}])
    assert called == []  # fn not invoked on sync path
    assert "headline" in out.lower() or "1." in out


def test_llm_summary_generator_rejects_non_callable() -> None:
    with pytest.raises(TypeError):
        LLMSummaryGenerator("not a function")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_llm_summary_generator_uses_callback_when_fast() -> None:
    async def fn(msgs):
        return f"ok:{len(msgs)}"

    gen = LLMSummaryGenerator(fn)
    out = await gen.summarize_async([{"role": "user", "content": "a"}])
    assert out == "ok:1"


@pytest.mark.asyncio
async def test_llm_summary_generator_falls_back_on_timeout() -> None:
    async def slow(msgs):
        await asyncio.sleep(2.0)
        return "never"

    gen = LLMSummaryGenerator(slow, timeout_seconds=0.05)
    out = await gen.summarize_async([{"role": "user", "content": "alpha"}])
    # Fallback produced something; should at least include the headline.
    assert "1. alpha" in out


@pytest.mark.asyncio
async def test_llm_summary_generator_falls_back_on_exception() -> None:
    async def bad(msgs):
        raise RuntimeError("llm exploded")

    gen = LLMSummaryGenerator(bad)
    out = await gen.summarize_async([{"role": "user", "content": "alpha"}])
    assert "1. alpha" in out


@pytest.mark.asyncio
async def test_llm_summary_generator_raises_when_fallback_also_fails() -> None:
    async def bad(msgs):
        raise RuntimeError("llm exploded")

    class BrokenFallback:
        def summarize(self, msgs):
            raise RuntimeError("fallback exploded")

    gen = LLMSummaryGenerator(bad, fallback=BrokenFallback())
    with pytest.raises(SummaryGeneratorError):
        await gen.summarize_async([{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_llm_summary_generator_custom_fallback_respected() -> None:
    async def bad(msgs):
        raise RuntimeError("llm exploded")

    seen: list[int] = []

    class CountingFallback:
        def summarize(self, msgs):
            seen.append(len(msgs))
            return "CUSTOM-FALLBACK"

    gen = LLMSummaryGenerator(bad, fallback=CountingFallback())
    out = await gen.summarize_async([{"role": "user", "content": "a"}])
    assert out == "CUSTOM-FALLBACK"
    assert seen == [1]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def test_extract_text_string_passthrough() -> None:
    assert extract_text("plain") == "plain"


def test_extract_text_dict_with_string_content() -> None:
    assert extract_text({"content": "abc"}) == "abc"


def test_extract_text_dict_with_list_content() -> None:
    assert extract_text({"content": [{"text": "alpha"}, {"text": "beta"}]}) == "alpha\nbeta"


def test_extract_text_object_with_attribute() -> None:
    class M:
        content = "x"

    assert extract_text(M()) == "x"


def test_extract_text_returns_empty_when_unknown() -> None:
    assert extract_text(42) == ""
    assert extract_text(None) == ""


def test_count_words_basic() -> None:
    assert count_words("one two three four") == 4
    assert count_words("hello, world! how are you?") == 5
    assert count_words("") == 0


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_concrete_generators_satisfy_protocol() -> None:
    assert isinstance(HeadlineSummaryGenerator(), SummaryGenerator)
    # LLMSummaryGenerator only implements summarize() at the type level
    # because summarize_async is async; runtime check should still succeed
    # because the sync summarize() method exists.
    gen = LLMSummaryGenerator(lambda msgs: asyncio.sleep(0, result="x"))
    assert isinstance(gen, SummaryGenerator)
