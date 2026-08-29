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

"""Tests for P92-C TF-IDF inverted index.

Covers:
    - ``TfIdfSkillIndex.build()``: empty, single, multiple documents
    - ``TfIdfSkillIndex.upsert()``: insert new, update existing
    - ``TfIdfSkillIndex.remove()``: remove existing, no-op for non-existent
    - IDF computation: rare terms have higher idf
    - Field boosts: name/title > tags > description > body
    - Length normalization: shorter docs not penalized, longer docs not inflated
    - Source weight: project > local > mcp
    - Pinned documents: always rank above unpinned
    - Search: min_score filtering, top_k truncation, EmptyQueryError
    - Persistence: save/load roundtrip, version check, corrupt JSON
    - Stats: total_docs, total_terms, total_inverted_entries
"""
# pylint: disable=no-name-in-module

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from clawcodex_ext.services.skill_search.config import SkillSearchConfig
from clawcodex_ext.services.skill_search.document import SkillSearchDocument
from clawcodex_ext.services.skill_search.exceptions import (
    IndexCorruptError,
    EmptyQueryError,
)
from clawcodex_ext.services.skill_search.index import (
    TfIdfSkillIndex,
)
from clawcodex_ext.services.skill_search.tokenizer import (
    Tokenizer,
    create_default_tokenizer,
)

# ============================================================================
# Fixture helpers
# ============================================================================


def create_test_tokenizer() -> Tokenizer:
    return create_default_tokenizer(cjk_word_tokenizer=None)


def create_test_config() -> SkillSearchConfig:
    return SkillSearchConfig(
        enabled=True,
        top_k=8,
        min_score=0.05,
    )


def doc(
    name: str,
    source: str = "local",
    *,
    title: str | None = None,
    description: str = "",
    body: str = "",
    tags: tuple[str, ...] = (),
    weight: float | None = None,
) -> SkillSearchDocument:
    doc_id = SkillSearchDocument.make_id(source, name)
    if weight is None:
        weight = {"project": 1.3, "local": 1.1, "mcp": 0.9}.get(source, 1.0)
    return SkillSearchDocument(
        id=doc_id,
        name=name,
        title=title or name.title(),
        description=description,
        body=body,
        source=source,  # type: ignore
        tags=tags,
        weight=weight,
    )


# ============================================================================
# TestBuild
# ============================================================================


class TestTfIdfSkillIndexBuild:
    def test_build_empty(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        index.build([])

        stats = index.total_stats()
        assert stats.total_docs == 0
        assert stats.total_terms == 0
        assert stats.total_inverted_entries == 0

    def test_build_single_doc(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d = doc("browser_automation", description="browser automation playwright")
        index.build([d])

        assert index.total_docs == 1
        assert "browser" in index.inverted_index
        assert "automation" in index.inverted_index
        assert "playwright" in index.inverted_index
        assert index.doc_freq["browser"] == 1

    def test_build_multiple_docs(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        docs = [
            doc("browser_automation", description="browser automation playwright"),
            doc("git_helper", description="git commit push"),
            doc("file_editor", description="edit file browser"),
        ]
        index.build(docs)

        assert index.total_docs == 3
        assert index.doc_freq["browser"] == 2
        assert index.doc_freq["git"] == 1
        assert "browser" in index.inverted_index
        assert len(index.inverted_index["browser"]) == 2

    def test_token_counts_correct(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d = doc("test", description="one two three")
        index.build([d])

        # token_counts stores the raw (unboosted) token count across all fields:
        # name="test" → 1, title="Test" → 1, description="one two three" → 3
        assert index.token_counts[d.id] == 5

    def test_doc_freq_correct_after_build(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        docs = [
            doc("a", description="common term"),
            doc("b", description="common term rare"),
            doc("c", description="common"),
        ]
        index.build(docs)

        assert index.doc_freq["common"] == 3
        assert index.doc_freq["term"] == 2
        assert index.doc_freq["rare"] == 1

    def test_idf_computed_correct(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        docs = [
            doc("a", description="common term"),
            doc("b", description="common term rare"),
            doc("c", description="common"),
        ]
        index.build(docs)

        # idf = log((3+1)/(df+1)) + 1
        # df=3 → log(4/4) + 1 = 0 + 1 = 1.0
        assert 0.99 <= index.idf["common"] <= 1.01
        # df=2 → log(4/3) + 1 ≈ 0.2877 + 1 = 1.2877
        assert 1.2 <= index.idf["term"] <= 1.3
        # df=1 → log(4/2) + 1 = log(2) + 1 ≈ 0.693 + 1 = 1.693
        assert 1.6 <= index.idf["rare"] <= 1.7

    def test_rare_term_has_higher_idf(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        docs = [doc(f"doc{i}", description="common rare") for i in range(10)]
        docs[-1] = doc("doc_last", description="common")
        index.build(docs)

        # common appears in all 10 docs, rare in 9 docs
        assert index.idf["common"] < index.idf["rare"]


# ============================================================================
# TestUpsert
# ============================================================================


class TestTfIdfSkillIndexUpsert:
    def test_upsert_new_doc(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d = doc("test", description="foo bar")
        index.upsert(d)

        assert index.total_docs == 1
        assert "foo" in index.inverted_index
        assert "bar" in index.inverted_index

    def test_upsert_update_existing(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d1 = doc("test", description="foo bar")
        index.upsert(d1)
        assert index.total_docs == 1
        assert "foo" in index.inverted_index

        d2 = doc("test", description="foo baz")
        index.upsert(d2)
        assert index.total_docs == 1
        assert "foo" in index.inverted_index
        assert "baz" in index.inverted_index
        assert "bar" not in index.inverted_index or index.doc_freq.get("bar", 0) == 0

    def test_upsert_updates_idf(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d1 = doc("a", description="term")
        d2 = doc("b", description="other")
        index.build([d1, d2])
        idf_before = index.idf["term"]

        d3 = doc("c", description="term")
        index.upsert(d3)

        # df increased from 1 to 2, N from 2 to 3, idf should decrease
        idf_after = index.idf["term"]
        assert idf_after < idf_before

    def test_upsert_empty_doc_no_tokens(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d = doc("the", description="")
        index.upsert(d)

        # "the" is a stopword → name and title produce no tokens
        # description is empty → no tokens
        # _tokenize_and_count returns empty → doc not added
        assert index.total_docs == 0
        assert d.id not in index.doc_store


# ============================================================================
# TestRemove
# ============================================================================


class TestTfIdfSkillIndexRemove:
    def test_remove_existing(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d1 = doc("a", description="foo bar")
        d2 = doc("b", description="foo baz")
        index.build([d1, d2])
        assert index.total_docs == 2
        assert index.doc_freq["foo"] == 2

        index.remove(d1.id)

        assert index.total_docs == 1
        assert d1.id not in index.doc_store
        assert index.doc_freq["foo"] == 1

    def test_remove_non_existing_noop(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d = doc("a", description="foo")
        index.upsert(d)
        assert index.total_docs == 1

        index.remove("nonexistent")

        assert index.total_docs == 1
        assert d.id in index.doc_store

    def test_remove_updates_idf(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        docs = [doc(f"doc{i}", description="common term") for i in range(3)]
        docs.append(doc("doc3", description="common"))
        index.build(docs)
        idf_before = index.idf["term"]

        index.remove(docs[0].id)

        # df decreased from 3 to 2, N from 4 to 3, idf should increase
        idf_after = index.idf["term"]
        assert idf_after > idf_before


# ============================================================================
# TestFieldBoost
# ============================================================================


class TestFieldBoost:
    def test_name_match_has_higher_boost(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)

        # doc1: "browser" in name (boost 3.0)
        d1 = doc("browser", body="this is something else")
        # doc2: "browser" in body (boost 1.0)
        d2 = doc("editor", body="this mentions browser here")

        index.build([d1, d2])
        results = index.search("browser")

        # d1 should rank higher
        assert len(results) == 2
        assert results[0].document.id == d1.id
        assert results[1].document.id == d2.id
        assert results[0].score > results[1].score

    def test_multiple_field_boosts_add(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)

        # "browser" in name (3.0) + tags (2.5) → total 5.5
        d1 = doc("browser", tags=("browser", "web"), description="...")
        # "browser" only in description (2.0)
        d2 = doc("other", description="something about browser")

        index.build([d1, d2])
        results = index.search("browser")

        assert len(results) == 2
        assert results[0].document.id == d1.id
        # Score should be significantly higher
        assert results[0].score > 2 * results[1].score

    def test_tags_have_higher_boost_than_description(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)

        # 2.5 boost for tags
        d1 = doc("a", tags=("search",), description="other stuff")
        # 2.0 boost for description
        d2 = doc("b", description="search here")

        index.build([d1, d2])
        results = index.search("search")

        assert results[0].document.id == d1.id
        assert results[0].score > results[1].score

    def test_title_has_same_boost_as_name(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)

        d1 = doc("search_demo", title="Custom Search", description="...")
        d2 = doc("demo", description="this is a custom search demo")

        index.build([d1, d2])
        results = index.search("custom search")

        assert results[0].document.id == d1.id


# ============================================================================
# TestSourceWeight
# ============================================================================


class TestSourceWeight:
    def test_project_has_higher_weight_than_local(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)

        d1 = doc("browser", source="project", description="browser automation")
        d2 = doc("browser", source="local", description="browser automation")

        index.build([d1, d2])
        results = index.search("browser")

        assert len(results) == 2
        assert results[0].document.id == d1.id
        assert results[0].score > results[1].score

    def test_local_has_higher_weight_than_mcp(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)

        d1 = doc("browser", source="local", description="browser")
        d2 = doc("browser", source="mcp", description="browser")

        index.build([d1, d2])
        results = index.search("browser")

        assert results[0].document.id == d1.id
        assert results[0].score > results[1].score


# ============================================================================
# TestLengthNormalization
# ============================================================================


class TestLengthNormalization:
    def test_short_doc_not_penalized(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)

        # Short doc: only "browser"
        d1 = doc("browser", description="browser")
        # Long doc: many words, one of which is "browser"
        long_text = " ".join([f"word{i}" for i in range(100)]) + " browser"
        d2 = doc("other", description=long_text)

        index.build([d1, d2])
        results = index.search("browser")

        # d1's normalized score should be higher
        # boosted_tf / sqrt(raw_tokens): d1: 2.0/√1=2.0, d2: 2.0/√101≈0.199
        assert results[0].document.id == d1.id

    def test_long_doc_not_inflated(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)

        # Query term once in 100 words vs once in 10 words
        d1 = doc("short", description="query " + " ".join([f"w{i}" for i in range(9)]))
        d2 = doc("long", description="query " + " ".join([f"w{i}" for i in range(99)]))

        index.build([d1, d2])
        results = index.search("query")

        # Both have one "query", but d1 is shorter → higher normalized score
        assert results[0].document.id == d1.id


# ============================================================================
# TestSearch
# ============================================================================


class TestSearch:
    def test_search_single_term_single_result(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d = doc("browser", description="browser automation")
        index.build([d])

        results = index.search("browser")
        assert len(results) == 1
        assert results[0].document.id == d.id

    def test_search_multiple_terms(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d1 = doc("browser_automation", description="browser automation playwright")
        d2 = doc("git_helper", description="git commit")
        index.build([d1, d2])

        results = index.search("browser playwright")
        assert len(results) == 1
        assert results[0].document.name == "browser_automation"

    def test_search_ranks_by_relevance(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d1 = doc("a", description="foo bar baz")
        d2 = doc("b", description="foo bar")
        d3 = doc("c", description="foo")
        index.build([d1, d2, d3])

        results = index.search("foo bar")
        scores = [r.score for r in results]
        # d1 matches both terms, d2 matches both terms but shorter → d2 should be top?
        # Actually d1 has more total tokens → length normalization reduces score
        ids = [r.document.id for r in results]
        assert d2.id in ids[:2]
        assert d3.id in ids[1:3]
        # Scores are decreasing
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_min_score(self):
        tokenizer = create_test_tokenizer()
        config = SkillSearchConfig(min_score=1.0)
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d1 = doc("a", description="foo bar")
        d2 = doc("b", description="baz qux")
        index.build([d1, d2])

        results = index.search("foo")
        # With min_score 1.0, what's the actual score?
        # idf = log((2+1)/(1+1)) + 1 = log(1.5) + 1 ≈ 0.405 + 1 = 1.405
        # idf² ≈ 1.976, tf 2.0 / sqrt(2) ≈ 1.414, weight 1.1 → ~ 1.976 * 1.414 * 1.1 ≈ 3.07
        # So it should pass
        assert len(results) == 1

        results = index.search("foo baz")
        # Both terms present but in different docs → each doc has one term, score ~3 vs ~3
        assert len(results) == 2

    def test_search_respects_top_k(self):
        tokenizer = create_test_tokenizer()
        config = SkillSearchConfig(top_k=2)
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        for i in range(5):
            d = doc(f"doc{i}", description="term common")
            index.upsert(d)

        results = index.search("common")
        assert len(results) == 2

    def test_search_respects_pinned(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d1 = doc("a", description="alpha bravo charlie")
        d2 = doc("b", description="bravo charlie delta")
        d3 = doc("c", description="charlie delta echo")
        index.build([d1, d2, d3])

        results = index.search("charlie", pinned_doc_ids={d3.id})

        ids = [r.document.id for r in results]
        # d3 should be first even though it's lowest score for "charlie"
        assert ids[0] == d3.id

    def test_pinned_still_sorted_by_score(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d1 = doc("a", description="query")
        d2 = doc("b", description="something query something")
        index.build([d1, d2])

        results = index.search("query", pinned_doc_ids={d2.id, d1.id})
        ids = [r.document.id for r in results]
        # d1 has higher score, should come first among pinned
        assert ids[0] == d1.id
        assert ids[1] == d2.id

    def test_empty_query_after_tokenization_raises(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d = doc("test", description="test")
        index.build([d])

        try:
            index.search("and or the")
            assert False, "Expected EmptyQueryError"
        except EmptyQueryError:
            pass  # Raising EmptyQueryError is the asserted behavior.

    def test_no_results_returns_empty(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d = doc("test", description="test")
        index.build([d])

        results = index.search("nonexistent")
        assert len(results) == 0

    def test_result_contains_matched_terms(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d = doc("browser_automation", description="browser automation")
        index.build([d])

        results = index.search("browser")
        assert len(results) == 1
        assert "browser" in results[0].matched_terms

    def test_result_contains_term_contributions(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        d = doc("test", description="foo bar")
        index.build([d])

        results = index.search("foo bar")
        assert "foo" in results[0].term_contributions
        assert "bar" in results[0].term_contributions
        assert results[0].term_contributions["foo"] > 0


# ============================================================================
# TestPersistence
# ============================================================================


class TestPersistence:
    def test_save_load_roundtrip(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        docs = [
            doc("browser", description="browser automation", tags=("web",)),
            doc("git", description="git commit push", source="project"),
        ]
        index.build(docs)

        with tempfile.NamedTemporaryFile(encoding="utf-8", mode="w+", suffix=".json", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            saved_path = index.save(tmp_path)
            loaded = TfIdfSkillIndex.load(saved_path, tokenizer, config)

            assert loaded.total_docs == index.total_docs
            assert loaded.total_docs == 2
            assert set(loaded.inverted_index.keys()) == set(index.inverted_index.keys())
            assert loaded.doc_freq == index.doc_freq
            # IDF should be the same (same N, same df)
            assert loaded.idf.keys() == index.idf.keys()
            for term in loaded.idf:
                assert abs(loaded.idf[term] - index.idf[term]) < 1e-9
            # Search should produce same ranking
            results_original = index.search("browser")
            results_loaded = loaded.search("browser")
            assert len(results_original) == len(results_loaded)
            assert results_original[0].document.id == results_loaded[0].document.id
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_corrupt_json_raises(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        with tempfile.NamedTemporaryFile(encoding="utf-8", mode="w+", suffix=".json", delete=False) as f:
            f.write("this is not valid json {")
            tmp_path = Path(f.name)
        try:
            try:
                TfIdfSkillIndex.load(tmp_path, tokenizer, config)
                assert False, "Expected IndexCorruptError"
            except IndexCorruptError:
                pass  # Raising IndexCorruptError is the asserted behavior.
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_version_mismatch_raises(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        with tempfile.NamedTemporaryFile(encoding="utf-8", mode="w+", suffix=".json", delete=False) as f:
            data = {
                "version": 999,
                "total_docs": 0,
                "doc_store": {},
                "token_counts": {},
                "inverted_index": {},
                "doc_freq": {},
                "idf": {},
            }
            json.dump(data, f)
            tmp_path = Path(f.name)
        try:
            try:
                TfIdfSkillIndex.load(tmp_path, tokenizer, config)
                assert False, "Expected IndexCorruptError for wrong version"
            except IndexCorruptError:
                pass  # Raising IndexCorruptError is the asserted behavior.
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_file_not_found_raises(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        missing = Path("/nonexistent/path/index.json")
        try:
            TfIdfSkillIndex.load(missing, tokenizer, config)
            assert False, "Expected IndexCorruptError"
        except IndexCorruptError:
            pass  # Raising IndexCorruptError is the asserted behavior.


# ============================================================================
# TestStats
# ============================================================================


class TestStats:
    def test_total_stats_correct(self):
        tokenizer = create_test_tokenizer()
        config = create_test_config()
        index = TfIdfSkillIndex(tokenizer=tokenizer, config=config)
        docs = [
            doc("a", description="foo bar"),
            doc("b", description="foo baz"),
        ]
        index.build(docs)

        stats = index.total_stats()
        assert stats.total_docs == 2
        # Unique terms: b (from name/title) + foo, bar, baz (from description)
        # "a" is a Latin stopword and gets filtered out
        assert stats.total_terms == 4
        # Inverted entries: b(1) + foo(2) + bar(1) + baz(1)
        assert stats.total_inverted_entries == 5
        assert stats.approximate_bytes > 0
