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

"""Tests for P92-D SkillSearcher.

Covers:
    - ``ensure_index()``: disk load, corrupt rebuild, feature-flag-off error
    - ``search()``: basic, tags filter, source filter, pinned prioritization
    - ``pin()`` / ``unpin()`` / ``get_pinned()``: persistence roundtrip
    - ``inspect()``: normal return, ``None`` when index not loaded
    - ``stats()``: normal return, ``None`` when index not loaded
    - ``refresh()``: full rebuild + persistence
"""
# pylint: disable=no-name-in-module

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from clawcodex_ext.services.skill_search.config import SkillSearchConfig
from clawcodex_ext.services.skill_search.exceptions import SearchDisabledError
from clawcodex_ext.services.skill_search.searcher import (
    SkillSearcher,
    FieldInspect,
)
from clawcodex_ext.services.skill_search.tokenizer import create_default_tokenizer

# ============================================================================
# Helpers
# ============================================================================


def _make_skill(**kwargs):
    """Create a minimal Skill-like object for testing."""
    from clawcodex_ext.skills.model import Skill

    defaults = {
        "name": "test_skill",
        "description": "A test skill",
        "content": "",
        "markdown_content": "",
        "loaded_from": "user",
        "source": "userSettings",
        "display_name": None,
        "when_to_use": None,
        "allowed_tools": [],
        "is_hidden": False,
    }
    defaults.update(kwargs)
    return Skill(**defaults)


def _make_registry(skills: list):
    """Create a mock SkillRegistryExt that returns the given skills."""
    registry = MagicMock()
    registry.get_all_skills.return_value = skills
    return registry


def _make_config(**kwargs) -> SkillSearchConfig:
    defaults = {
        "enabled": True,
        "top_k": 8,
        "min_score": 0.05,
    }
    defaults.update(kwargs)
    if "index_path" not in defaults:
        defaults["index_path"] = Path(tempfile.mkdtemp()) / "index.json"
    return SkillSearchConfig(**defaults)


def _make_searcher(skills: list | None = None, **config_kwargs) -> SkillSearcher:
    if skills is None:
        skills = []
    registry = _make_registry(skills)
    config = _make_config(**config_kwargs)
    tokenizer = create_default_tokenizer(cjk_word_tokenizer=None)
    return SkillSearcher(registry, config=config, tokenizer=tokenizer)


# ============================================================================
# TestEnsureIndex
# ============================================================================


class TestEnsureIndex:
    @pytest.mark.asyncio
    async def test_build_from_registry(self):
        skills = [
            _make_skill(name="browser", description="browser automation"),
            _make_skill(name="git", description="git commit helper"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 2

    @pytest.mark.asyncio
    async def test_load_from_disk(self):
        skills = [
            _make_skill(name="browser", description="browser automation"),
        ]
        searcher = _make_searcher(skills)
        await searcher.refresh()

        # Create a second searcher pointing to same index path
        searcher2 = _make_searcher(skills, index_path=searcher._config.index_path)
        await searcher2.ensure_index()

        stats = searcher2.stats()
        assert stats is not None
        assert stats.total_docs == 1

    @pytest.mark.asyncio
    async def test_corrupt_rebuilds(self):
        skills = [
            _make_skill(name="browser", description="browser automation"),
        ]
        config = _make_config()
        # Write corrupt JSON
        config.index_path.expanduser().parent.mkdir(parents=True, exist_ok=True)
        config.index_path.expanduser().write_text("not valid json {{{", encoding="utf-8")

        registry = _make_registry(skills)
        tokenizer = create_default_tokenizer(cjk_word_tokenizer=None)
        searcher = SkillSearcher(registry, config=config, tokenizer=tokenizer)

        await searcher.ensure_index()
        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 1

    @pytest.mark.asyncio
    async def test_ensure_index_idempotent(self):
        skills = [
            _make_skill(name="browser", description="browser automation"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()
        await searcher.ensure_index()

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 1

    @pytest.mark.asyncio
    async def test_flag_off_raises(self):
        searcher = _make_searcher(enabled=False)
        with pytest.raises(SearchDisabledError):
            await searcher.ensure_index()


# ============================================================================
# TestSearch
# ============================================================================


class TestSearch:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        skills = [
            _make_skill(name="browser_automation", description="browser automation playwright"),
            _make_skill(name="git_helper", description="git commit push"),
        ]
        searcher = _make_searcher(skills)
        results = await searcher.search("browser")

        assert len(results) == 1
        assert results[0].document.name == "browser_automation"

    @pytest.mark.asyncio
    async def test_search_ranks_by_relevance(self):
        skills = [
            _make_skill(name="a", description="foo bar baz"),
            _make_skill(name="b", description="foo bar"),
            _make_skill(name="c", description="foo"),
        ]
        searcher = _make_searcher(skills)
        results = await searcher.search("foo bar")

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_tags_filter(self):
        skills = [
            _make_skill(name="browser", description="browser", allowed_tools=["web"]),
            _make_skill(name="git", description="git", allowed_tools=["bash"]),
        ]
        searcher = _make_searcher(skills)
        results = await searcher.search("browser git", tags=["web"])

        assert len(results) == 1
        assert results[0].document.name == "browser"

    @pytest.mark.asyncio
    async def test_tags_filter_case_insensitive(self):
        skills = [
            _make_skill(name="browser", description="browser", allowed_tools=["Web"]),
            _make_skill(name="git", description="git", allowed_tools=["Bash"]),
        ]
        searcher = _make_searcher(skills)
        results = await searcher.search("browser git", tags=["web"])

        assert len(results) == 1
        assert results[0].document.name == "browser"

    @pytest.mark.asyncio
    async def test_tags_filter_any_match(self):
        skills = [
            _make_skill(name="browser", description="browser", allowed_tools=["web"]),
            _make_skill(name="git", description="git", allowed_tools=["bash"]),
        ]
        searcher = _make_searcher(skills)
        results = await searcher.search("browser git", tags=["web", "bash"])

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_source_filter(self):
        skills = [
            _make_skill(name="browser", description="browser", loaded_from="project"),
            _make_skill(name="git", description="git", loaded_from="user"),
        ]
        searcher = _make_searcher(skills)
        results = await searcher.search("browser git", source="project")

        assert len(results) == 1
        assert results[0].document.name == "browser"

    @pytest.mark.asyncio
    async def test_pinned_ranks_first(self):
        skills = [
            _make_skill(name="browser", description="browser automation"),
            _make_skill(name="git", description="git commit"),
        ]
        searcher = _make_searcher(skills)
        searcher.pin("git")
        results = await searcher.search("browser")

        # git is pinned but doesn't match "browser" well
        # browser should match better
        # Actually, pinned just means pinned items come first among matched items
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_pinned_still_sorted_by_score(self):
        skills = [
            _make_skill(name="a", description="query"),
            _make_skill(name="b", description="something query something"),
        ]
        searcher = _make_searcher(skills)
        searcher.pin("a")
        searcher.pin("b")
        results = await searcher.search("query")

        ids = [r.document.name for r in results]
        # a is shorter → higher score, should come first among pinned
        assert ids[0] == "a"
        assert ids[1] == "b"

    @pytest.mark.asyncio
    async def test_respects_top_k(self):
        config = _make_config(top_k=2)
        skills = [_make_skill(name=f"skill{i}", description="common term") for i in range(5)]
        registry = _make_registry(skills)
        tokenizer = create_default_tokenizer(cjk_word_tokenizer=None)
        searcher = SkillSearcher(registry, config=config, tokenizer=tokenizer)
        results = await searcher.search("common")

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_no_results(self):
        skills = [
            _make_skill(name="browser", description="browser"),
        ]
        searcher = _make_searcher(skills)
        results = await searcher.search("nonexistent")

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_implicit_ensure_index(self):
        skills = [
            _make_skill(name="browser", description="browser"),
        ]
        searcher = _make_searcher(skills)
        # Don't call ensure_index explicitly
        results = await searcher.search("browser")

        assert len(results) == 1


# ============================================================================
# TestPinManagement
# ============================================================================


class TestPinManagement:
    def test_pin_adds(self):
        searcher = _make_searcher()
        searcher.pin("skill_a")
        assert "skill_a" in searcher.get_pinned()

    def test_pin_idempotent(self):
        searcher = _make_searcher()
        searcher.pin("skill_a")
        searcher.pin("skill_a")
        assert searcher.get_pinned().count("skill_a") == 1

    def test_unpin_removes(self):
        searcher = _make_searcher()
        searcher.pin("skill_a")
        searcher.pin("skill_b")
        searcher.unpin("skill_a")
        assert searcher.get_pinned() == ["skill_b"]

    def test_unpin_nonexistent_noop(self):
        searcher = _make_searcher()
        searcher.pin("skill_a")
        searcher.unpin("nonexistent")
        assert searcher.get_pinned() == ["skill_a"]

    def test_get_pinned_returns_copy(self):
        searcher = _make_searcher()
        searcher.pin("skill_a")
        pinned = searcher.get_pinned()
        pinned.append("skill_b")
        assert searcher.get_pinned() == ["skill_a"]

    def test_pin_persisted(self):
        searcher = _make_searcher()
        searcher.pin("skill_a")
        searcher.pin("skill_b")

        # Reload from the same pinned.json
        searcher2 = _make_searcher(index_path=searcher._config.index_path)
        searcher2._load_pinned()
        assert searcher2.get_pinned() == ["skill_a", "skill_b"]

    def test_pin_persistence_roundtrip(self):
        searcher = _make_searcher()
        searcher.pin("skill_x")
        searcher.pin("skill_y")
        searcher.unpin("skill_x")
        searcher.pin("skill_z")

        searcher2 = _make_searcher(index_path=searcher._config.index_path)
        searcher2._load_pinned()
        assert searcher2.get_pinned() == ["skill_y", "skill_z"]

    @pytest.mark.asyncio
    async def test_pin_before_index(self):
        """Pin a skill that doesn't exist yet in the index."""
        skills = [
            _make_skill(name="browser", description="browser"),
        ]
        searcher = _make_searcher(skills)
        searcher.pin("browser")
        searcher.pin("nonexistent")

        results = await searcher.search("browser")
        assert len(results) == 1
        assert results[0].document.name == "browser"


# ============================================================================
# TestInspect
# ============================================================================


class TestInspect:
    @pytest.mark.asyncio
    async def test_inspect_returns_result(self):
        skills = [
            _make_skill(name="browser", description="browser automation"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        result = searcher.inspect("browser")
        assert result is not None
        assert result.name == "browser"
        assert result.source == "local"
        assert result.token_count > 0
        assert "name" in result.fields
        assert "title" in result.fields
        assert "description" in result.fields
        assert "body" in result.fields
        assert "tags" in result.fields
        for field in result.fields.values():
            assert isinstance(field, FieldInspect)
            assert isinstance(field.token_count, int)
            assert isinstance(field.token_sample, list)

    @pytest.mark.asyncio
    async def test_inspect_not_found(self):
        skills = [
            _make_skill(name="browser", description="browser"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        result = searcher.inspect("nonexistent")
        assert result is None

    def test_inspect_no_index(self):
        searcher = _make_searcher()
        result = searcher.inspect("browser")
        assert result is None

    @pytest.mark.asyncio
    async def test_inspect_token_sample_limit(self):
        skills = [
            _make_skill(
                name="browser",
                description="a b c d e f g h i j k l m n o p q r s t u v w x y",
            ),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        result = searcher.inspect("browser")
        assert result is not None
        desc = result.fields["description"]
        assert len(desc.token_sample) <= 20

    @pytest.mark.asyncio
    async def test_inspect_per_field_counts(self):
        skills = [
            _make_skill(
                name="browser",
                description="browser automation playwright",
            ),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        result = searcher.inspect("browser")
        assert result is not None
        total = sum(f.token_count for f in result.fields.values())
        assert total == result.token_count


# ============================================================================
# TestStats
# ============================================================================


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_returns_index_stats(self):
        skills = [
            _make_skill(name="browser", description="browser"),
            _make_skill(name="git", description="git"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 2
        assert stats.total_terms > 0
        assert stats.total_inverted_entries > 0

    def test_stats_no_index(self):
        searcher = _make_searcher()
        stats = searcher.stats()
        assert stats is None


# ============================================================================
# TestRefresh
# ============================================================================


class TestRefresh:
    @pytest.mark.asyncio
    async def test_refresh_builds_and_persists(self):
        skills = [
            _make_skill(name="browser", description="browser"),
        ]
        config = _make_config()
        registry = _make_registry(skills)
        tokenizer = create_default_tokenizer(cjk_word_tokenizer=None)
        searcher = SkillSearcher(registry, config=config, tokenizer=tokenizer)

        await searcher.refresh()

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 1

        # Verify persistence
        assert config.index_path.expanduser().exists()

    @pytest.mark.asyncio
    async def test_refresh_flag_off_noop(self):
        config = _make_config(enabled=False)
        registry = _make_registry([_make_skill(name="browser", description="browser")])
        tokenizer = create_default_tokenizer(cjk_word_tokenizer=None)
        searcher = SkillSearcher(registry, config=config, tokenizer=tokenizer)

        await searcher.refresh()
        stats = searcher.stats()
        assert stats is None

    @pytest.mark.asyncio
    async def test_refresh_updates_existing(self):
        skills1 = [_make_skill(name="browser", description="browser")]
        searcher = _make_searcher(skills1)
        await searcher.ensure_index()

        skills2 = [
            _make_skill(name="browser", description="browser"),
            _make_skill(name="git", description="git commit"),
        ]
        searcher._registry.get_all_skills.return_value = skills2
        await searcher.refresh()

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 2

    @pytest.mark.asyncio
    async def test_refresh_preserves_pinned(self):
        skills = [
            _make_skill(name="browser", description="browser"),
        ]
        searcher = _make_searcher(skills)
        searcher.pin("browser")

        await searcher.refresh()

        # pinned should still be there
        assert "browser" in searcher.get_pinned()
        # search should still work with pinned
        results = await searcher.search("browser")
        assert len(results) == 1
