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

"""Tests for P92-B tokenizer.

Covers:
    - LatinProcessor: camelCase, snake_case, kebab-case, stopwords, digits
    - CJKProcessor: bigram fallback, jieba injection, short segments
    - FallbackProcessor: unmatched characters, whitespace, emoji
    - Tokenizer: mixed text, segment dispatch, deduplication, normalization
    - create_default_tokenizer factory
    - register_processor extensibility
"""
# pylint: disable=no-name-in-module

from __future__ import annotations

from clawcodex_ext.services.skill_search.tokenizer import (
    CJKProcessor,
    FallbackProcessor,
    LatinProcessor,
    Tokenizer,
    create_default_tokenizer,
)

# ====================================================================
# FallbackProcessor
# ====================================================================


class TestFallbackProcessor:
    def test_handles_any_char(self):
        proc = FallbackProcessor()
        assert proc.can_handle("a")
        assert proc.can_handle("ф")
        assert proc.can_handle("字")
        assert proc.can_handle("😀")

    def test_keeps_alphanumeric(self):
        proc = FallbackProcessor()
        assert proc.tokenize_segment("abc") == ["a", "b", "c"]
        assert proc.tokenize_segment("фыв") == ["ф", "ы", "в"]

    def test_drops_whitespace(self):
        proc = FallbackProcessor()
        assert proc.tokenize_segment("a b") == ["a", "b"]
        assert proc.tokenize_segment("  \t\n  ") == []

    def test_drops_unprintable(self):
        proc = FallbackProcessor()
        assert "\x00" not in proc.tokenize_segment("a\x00b")

    def test_lowest_priority(self):
        proc = FallbackProcessor()
        assert proc.priority == 9999


# ====================================================================
# LatinProcessor
# ====================================================================


class TestLatinProcessorCamelCase:
    def test_simple_camel(self):
        proc = LatinProcessor()
        assert proc.tokenize_segment("BrowserAutomation") == ["browser", "automation"]

    def test_pascal_case(self):
        proc = LatinProcessor()
        assert proc.tokenize_segment("XMLParser") == ["xml", "parser"]

    def test_acronym_camel(self):
        proc = LatinProcessor()
        assert proc.tokenize_segment("getHTTPResponse") == ["get", "http", "response"]

    def test_standalone_acronym(self):
        proc = LatinProcessor()
        assert proc.tokenize_segment("HTTP") == ["http"]

    def test_number_separation(self):
        proc = LatinProcessor()
        assert "3" in proc.tokenize_segment("python3")
        assert "2" in proc.tokenize_segment("version2")

    def test_lowercase_word(self):
        proc = LatinProcessor()
        assert proc.tokenize_segment("browser") == ["browser"]

    def test_camel_split_disabled(self):
        proc = LatinProcessor(split_camel=False)
        result = proc.tokenize_segment("BrowserAutomation")
        assert result == ["browserautomation"]


class TestLatinProcessorSnakeKebab:
    def test_snake_case(self):
        proc = LatinProcessor()
        assert proc.tokenize_segment("web_browser_skill") == ["web", "browser", "skill"]

    def test_kebab_case(self):
        proc = LatinProcessor()
        assert proc.tokenize_segment("hello-world") == ["hello", "world"]

    def test_mixed_delimiters(self):
        proc = LatinProcessor()
        assert proc.tokenize_segment("ab_bc-cd") == ["ab", "bc", "cd"]

    def test_snake_disabled(self):
        proc = LatinProcessor(split_snake=False)
        assert proc.tokenize_segment("web_browser") == ["web_browser"]

    def test_kebab_disabled(self):
        proc = LatinProcessor(split_kebab=False)
        assert proc.tokenize_segment("hello-world") == ["hello-world"]


class TestLatinProcessorStopwords:
    def test_common_stopwords(self):
        proc = LatinProcessor()
        result = proc.tokenize_segment("the browser is for automation")
        assert "the" not in result
        assert "is" not in result
        assert "for" not in result
        assert "browser" in result
        assert "automation" in result

    def test_all_stopwords(self):
        proc = LatinProcessor()
        result = proc.tokenize_segment("the and or")
        assert result == []

    def test_stopword_as_part_of_camel(self):
        proc = LatinProcessor()
        result = proc.tokenize_segment("CodeBrowser")
        assert result == ["code", "browser"]


class TestLatinProcessorPunctuation:
    def test_punctuation_removed(self):
        proc = LatinProcessor()
        result = proc.tokenize_segment("hello, world!")
        assert result == ["hello", "world"]

    def test_parentheses(self):
        proc = LatinProcessor()
        result = proc.tokenize_segment("test (example)")
        assert result == ["test", "example"]


class TestLatinProcessorCaseFolding:
    def test_case_folding_enabled(self):
        proc = LatinProcessor()
        result = proc.tokenize_segment("Browser BROWSER")
        assert all(t == "browser" for t in result)

    def test_case_folding_disabled(self):
        proc = LatinProcessor(case_fold=False)
        result = proc.tokenize_segment("Browser browser")
        assert "Browser" in result
        assert "browser" in result

    def test_case_folding_disabled_camel_split(self):
        proc = LatinProcessor(case_fold=False)
        result = proc.tokenize_segment("BrowserAutomation")
        assert "Browser" in result
        assert "Automation" in result


# ====================================================================
# CJKProcessor
# ====================================================================


class TestCJKProcessorBigram:
    def test_two_chars(self):
        proc = CJKProcessor()
        result = proc.tokenize_segment("网页")
        assert result == ["网页"]

    def test_three_chars(self):
        proc = CJKProcessor()
        result = proc.tokenize_segment("自动化")
        assert result == ["自动", "动化"]

    def test_four_chars_bigram_count(self):
        proc = CJKProcessor()
        result = proc.tokenize_segment("网页自动化")
        assert len(result) == 4
        assert "网页" in result
        assert "页自" in result
        assert "自动" in result
        assert "动化" in result

    def test_single_char(self):
        proc = CJKProcessor()
        assert proc.tokenize_segment("网") == ["网"]

    def test_empty_string(self):
        proc = CJKProcessor()
        assert proc.tokenize_segment("") == []

    def test_can_handle_cjk(self):
        proc = CJKProcessor()
        assert proc.can_handle("字")
        assert proc.can_handle("本")
        assert not proc.can_handle("a")
        assert not proc.can_handle("1")


class TestCJKProcessorWithTokenizer:
    def test_custom_word_tokenizer(self):
        proc = CJKProcessor(word_tokenizer=lambda s: s.split("|"))
        assert proc.tokenize_segment("网页|自动化") == ["网页", "自动化"]

    def test_none_tokenizer_falls_back(self):
        proc = CJKProcessor(word_tokenizer=None)
        result = proc.tokenize_segment("自动化")
        assert result == ["自动", "动化"]

    def test_priority(self):
        assert CJKProcessor().priority == 20


# ====================================================================
# Tokenizer — segment dispatch
# ====================================================================


class TestTokenizerSegmentDispatch:
    def test_latin_only(self):
        tok = Tokenizer()
        assert tok.tokenize("browser") == ["browser"]

    def test_cjk_only(self):
        tok = Tokenizer()
        result = tok.tokenize("网页自动化")
        assert "网页" in result
        assert "动化" in result

    def test_mixed_latin_cjk(self):
        tok = Tokenizer()
        result = tok.tokenize("Browser 网页 automation")
        assert "browser" in result
        assert "automation" in result
        assert "网页" in result

    def test_fallback_cyrillic(self):
        tok = Tokenizer()
        result = tok.tokenize("привет")
        assert "п" in result
        assert "р" in result
        assert "и" in result
        assert "в" in result
        assert "е" in result
        assert "т" in result

    def test_fallback_mixed(self):
        tok = Tokenizer()
        result = tok.tokenize("hello привет 世界")
        assert "hello" in result
        assert "п" in result
        assert "р" in result
        assert "世" in result or "世界" in result

    def test_fallback_thai(self):
        tok = Tokenizer()
        result = tok.tokenize("สวัสดี")
        assert len(result) > 0

    def test_fallback_emoji_dropped(self):
        tok = Tokenizer()
        result = tok.tokenize("hello 😀 world")
        assert "hello" in result
        assert "world" in result


# ====================================================================
# Tokenizer — normalization
# ====================================================================


class TestTokenizerNormalization:
    def test_min_length(self):
        tok = Tokenizer(min_token_length=2)
        result = tok.tokenize("a ab abc")
        assert "a" not in result
        assert "ab" in result
        assert "abc" in result

    def test_max_length(self):
        tok = Tokenizer(max_token_length=3)
        result = tok.tokenize("go verylongword")
        assert "go" in result
        assert "verylongword" not in result

    def test_length_bounds(self):
        tok = Tokenizer(min_token_length=2, max_token_length=5)
        result = tok.tokenize("a ab abcdef abc")
        assert result == ["ab", "abc"]


# ====================================================================
# Tokenizer — deduplication
# ====================================================================


class TestTokenizerDeduplication:
    def test_deduplicate_keeps_first(self):
        tok = Tokenizer()
        result = tok.tokenize("browser browser automation")
        assert result == ["browser", "automation"]

    def test_deduplicate_disabled(self):
        tok = Tokenizer(deduplicate=False)
        result = tok.tokenize("browser browser")
        assert result == ["browser", "browser"]

    def test_deduplicate_preserves_order(self):
        tok = Tokenizer()
        result = tok.tokenize("c b d b c")
        assert result == ["c", "b", "d"]


# ====================================================================
# Tokenizer — edge cases
# ====================================================================


class TestTokenizerEdgeCases:
    def test_empty_string(self):
        tok = Tokenizer()
        assert tok.tokenize("") == []

    def test_whitespace_only(self):
        tok = Tokenizer()
        assert tok.tokenize("   \t\n  ") == []

    def test_single_character(self):
        tok = Tokenizer()
        assert tok.tokenize("x") == ["x"]

    def test_all_stopwords(self):
        tok = Tokenizer()
        assert tok.tokenize("the and or") == []

    def test_numbers_only(self):
        tok = Tokenizer()
        result = tok.tokenize("123 456")
        assert "123" in result
        assert "456" in result

    def test_tokenize_batch(self):
        tok = Tokenizer()
        results = tok.tokenize_batch(["hello world", "foo bar"])
        assert len(results) == 2
        assert results[0] == ["hello", "world"]
        assert results[1] == ["foo", "bar"]


# ====================================================================
# Processor registration
# ====================================================================


class TestProcessorRegistration:
    def test_register_new_processor(self):
        class DummyProcessor(LatinProcessor):
            @property
            def priority(self):
                return 5

        tok = Tokenizer()
        tok.register_processor(DummyProcessor())
        assert len(tok._processors) == 3

    def test_priority_ordering(self):
        class HighPrio(LatinProcessor):
            @property
            def priority(self):
                return 5

        tok = Tokenizer(processors=[])
        tok.register_processor(LatinProcessor())
        tok.register_processor(HighPrio())
        assert tok._processors[0].priority == 5
        assert tok._processors[1].priority == 10


# ====================================================================
# create_default_tokenizer factory
# ====================================================================


class TestCreateDefaultTokenizer:
    def test_default(self):
        tok = create_default_tokenizer(cjk_word_tokenizer=None)
        result = tok.tokenize("browser automation")
        assert "browser" in result
        assert "automation" in result

    def test_cjk_auto_mode(self):
        tok = create_default_tokenizer(cjk_word_tokenizer="auto")
        result = tok.tokenize("网页自动化")
        assert len(result) > 0

    def test_cjk_custom_tokenizer(self):
        tok = create_default_tokenizer(cjk_word_tokenizer=lambda s: ["网页", "自动化"])
        assert tok.tokenize("网页自动化") == ["网页", "自动化"]

    def test_custom_stopwords(self):
        tok = create_default_tokenizer(
            stopwords=frozenset({"browser"}),
            cjk_word_tokenizer=None,
        )
        result = tok.tokenize("browser automation")
        assert "browser" not in result
        assert "automation" in result

    def test_case_fold_disabled(self):
        tok = create_default_tokenizer(case_fold=False, cjk_word_tokenizer=None)
        result = tok.tokenize("Browser")
        assert "Browser" in result
