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

"""Integration tests for P92-G skill search.

Covers end-to-end scenarios:
    - searcher + watcher + registry hook interplay
    - MCP / template skills included in search results
    - Multi-source skill ranking (project > local > template > mcp)
    - Search after incremental updates via watcher
    - Full rebuild after registry changes
    - Feature flag lifecycle: off → on → off
"""
# pylint: disable=no-name-in-module

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from clawcodex_ext.services.skill_search.config import SkillSearchConfig
from clawcodex_ext.services.skill_search.exceptions import SearchDisabledError
from clawcodex_ext.services.skill_search.searcher import SkillSearcher
from clawcodex_ext.services.skill_search.tokenizer import create_default_tokenizer
from clawcodex_ext.services.skill_search.watcher import SkillIndexWatcher

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
    """Create a mock SkillRegistryExt with callback tracking."""
    registry = MagicMock()
    registry.get_all_skills.return_value = skills
    registry._callbacks: list = []
    registry.on_skill_registered = registry._callbacks.append
    registry.off_skill_registered = lambda cb: (registry._callbacks.remove(cb) if cb in registry._callbacks else None)

    def _notify(skill):
        for cb in registry._callbacks:
            cb(skill)

    registry._notify = _notify
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
# TestIntegration: multi-source skill search
# ============================================================================


class TestIntegrationMultiSource:
    """End-to-end: skills from multiple sources are indexed and ranked."""

    @pytest.mark.asyncio
    async def test_multi_source_searches_together(self):
        """Skills from local, project, mcp, and template sources all appear in results."""
        skills = [
            _make_skill(
                name="browser_automation",
                description="browser automation with playwright",
                loaded_from="project",
            ),
            _make_skill(
                name="git_helper",
                description="git commit and push helper",
                loaded_from="user",
            ),
            _make_skill(
                name="dev_browser",
                description="development browser tools",
                loaded_from="mcp",
            ),
            _make_skill(
                name="web_template",
                description="web page template generator",
                loaded_from="template",
            ),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        results = await searcher.search("browser")
        names = [r.document.name for r in results]

        # All browser-related skills should appear
        assert "browser_automation" in names
        assert "dev_browser" in names

        # Project source should rank higher than mcp for same query
        project_idx = names.index("browser_automation")
        mcp_idx = names.index("dev_browser")
        assert project_idx < mcp_idx

    @pytest.mark.asyncio
    async def test_project_beats_local_beats_mcp(self):
        """Source weights: project(1.3) > local(1.1) > template(1.0) > mcp(0.9)."""
        skills = [
            _make_skill(
                name="browser_mcp",
                description="browser automation",
                loaded_from="mcp",
            ),
            _make_skill(
                name="browser_local",
                description="browser automation",
                loaded_from="user",
            ),
            _make_skill(
                name="browser_project",
                description="browser automation",
                loaded_from="project",
            ),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        results = await searcher.search("browser automation")
        scores = {r.document.name: r.score for r in results}

        # All three found
        assert len(results) == 3
        # project beats local
        assert scores["browser_project"] > scores["browser_local"]
        # local beats mcp
        assert scores["browser_local"] > scores["browser_mcp"]

    @pytest.mark.asyncio
    async def test_mcp_skills_searchable(self):
        """MCP-discovered skills are indexed and return in search results."""
        skills = [
            _make_skill(
                name="local_skill",
                description="a local helper",
                loaded_from="user",
            ),
            _make_skill(
                name="mcp_read_file",
                description="read files via MCP",
                loaded_from="mcp",
            ),
            _make_skill(
                name="mcp_write_file",
                description="write files via MCP",
                loaded_from="mcp",
            ),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        results = await searcher.search("mcp file")
        names = [r.document.name for r in results]
        assert "mcp_read_file" in names
        assert "mcp_write_file" in names

    @pytest.mark.asyncio
    async def test_source_filter_mcp(self):
        """Filtering by source=mcp returns only MCP skills."""
        skills = [
            _make_skill(
                name="mcp_tool",
                description="generic mcp tool",
                loaded_from="mcp",
            ),
            _make_skill(
                name="local_tool",
                description="generic local tool",
                loaded_from="user",
            ),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        results = await searcher.search("generic tool", source="mcp")
        assert len(results) == 1
        assert results[0].document.name == "mcp_tool"


# ============================================================================
# TestIntegration: watcher + searcher interplay
# ============================================================================


class TestIntegrationWatcherSearcher:
    @pytest.mark.asyncio
    async def test_watcher_updates_then_search_reflects(self):
        """After watcher upserts, search returns the new skill."""
        skills = [
            _make_skill(name="initial", description="initial skill"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        watcher = SkillIndexWatcher(searcher, searcher._registry, config=searcher._config)
        watcher.start()

        new_skill = _make_skill(
            name="playwright_helper",
            description="browser automation with playwright",
        )
        searcher._registry._notify(new_skill)

        results = await searcher.search("playwright automation")
        assert len(results) >= 1
        assert any(r.document.name == "playwright_helper" for r in results)

    @pytest.mark.asyncio
    async def test_register_many_skills_via_watcher(self):
        """Registering many skills via watcher, then search finds them all."""
        skills = [
            _make_skill(name="base", description="base"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        watcher = SkillIndexWatcher(searcher, searcher._registry, config=searcher._config)
        watcher.start()

        for i in range(20):
            skill = _make_skill(
                name=f"batch_skill_{i}",
                description=f"batch skill number {i}",
            )
            searcher._registry._notify(skill)

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 21  # 1 base + 20 batch

        results = await searcher.search("batch")
        assert len(results) >= 8  # top_k default

    @pytest.mark.asyncio
    async def test_watcher_preserves_existing_results(self):
        """Watcher upserts don't break existing indexed documents."""
        skills = [
            _make_skill(name="alpha", description="alpha description"),
            _make_skill(name="beta", description="beta description"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        watcher = SkillIndexWatcher(searcher, searcher._registry, config=searcher._config)
        watcher.start()

        new_skill = _make_skill(name="gamma", description="gamma")
        searcher._registry._notify(new_skill)

        results = await searcher.search("alpha")
        assert any(r.document.name == "alpha" for r in results)

        results = await searcher.search("gamma")
        assert any(r.document.name == "gamma" for r in results)


# ============================================================================
# TestIntegration: feature flag lifecycle
# ============================================================================


class TestIntegrationFeatureFlag:
    @pytest.mark.asyncio
    async def test_flag_off_then_on(self):
        """Feature flag off → search raises; flag on → search works."""
        skills = [
            _make_skill(name="browser", description="browser"),
        ]
        config = _make_config(enabled=False)
        registry = _make_registry(skills)
        tokenizer = create_default_tokenizer(cjk_word_tokenizer=None)
        searcher = SkillSearcher(registry, config=config, tokenizer=tokenizer)

        # Flag off: search raises
        with pytest.raises(SearchDisabledError):
            await searcher.search("browser")

        # Flag on: works
        enabled_config = _make_config(enabled=True, index_path=config.index_path)
        searcher = SkillSearcher(registry, config=enabled_config, tokenizer=tokenizer)
        results = await searcher.search("browser")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_watcher_flag_off_zero_overhead(self):
        """When config.enabled=False, watcher.start() does not register callbacks."""
        config = _make_config(enabled=False)
        registry = _make_registry([])
        tokenizer = create_default_tokenizer(cjk_word_tokenizer=None)
        searcher = SkillSearcher(registry, config=config, tokenizer=tokenizer)

        watcher = SkillIndexWatcher(searcher, registry, config=config)
        watcher.start()

        assert watcher._active is False
        assert len(registry._callbacks) == 0


# ============================================================================
# TestIntegration: rebuild + stale handling
# ============================================================================


class TestIntegrationRebuild:
    @pytest.mark.asyncio
    async def test_refresh_clears_removed_skills(self):
        """Full refresh rebuilds from registry, removing skills no longer present."""
        skills = [
            _make_skill(name="alpha", description="alpha"),
            _make_skill(name="beta", description="beta"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 2

        # Simulate registry change: beta is gone
        new_skills = [_make_skill(name="alpha", description="alpha")]
        searcher._registry.get_all_skills.return_value = new_skills
        await searcher.refresh()

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 1

        results = await searcher.search("beta")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_refresh_after_watcher_brings_consistency(self):
        """Watcher adds a skill, then refresh confirms it and cleans up."""
        skills = [
            _make_skill(name="alpha", description="alpha"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        watcher = SkillIndexWatcher(searcher, searcher._registry, config=searcher._config)
        watcher.start()

        # Watcher adds beta
        beta = _make_skill(name="beta", description="beta")
        searcher._registry._notify(beta)

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 2

        # Now refresh: registry only has alpha + gamma
        searcher._registry.get_all_skills.return_value = [
            _make_skill(name="alpha", description="alpha"),
            _make_skill(name="gamma", description="gamma"),
        ]
        await searcher.refresh()

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 2

        results = await searcher.search("gamma")
        assert any(r.document.name == "gamma" for r in results)

        results = await searcher.search("beta")
        assert len(results) == 0  # beta was removed by refresh


# ============================================================================
# TestIntegration: pin + search interplay
# ============================================================================


class TestIntegrationPinSearch:
    @pytest.mark.asyncio
    async def test_pinned_appears_first_in_search(self):
        """Pinned skills rank above unpinned in search results."""
        skills = [
            _make_skill(name="browser", description="browser automation"),
            _make_skill(name="git", description="git commit"),
            _make_skill(name="python", description="python scripting"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()
        searcher.pin("python")

        results = await searcher.search("browser git python")
        names = [r.document.name for r in results]
        assert names[0] == "python"

    @pytest.mark.asyncio
    async def test_pin_persists_across_searcher_instances(self):
        """Pinned list survives across searcher instances sharing the same path."""
        skills = [
            _make_skill(name="browser", description="browser"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()
        searcher.pin("browser")

        searcher2 = _make_searcher(skills, index_path=searcher._config.index_path)
        searcher2._load_pinned()
        assert "browser" in searcher2.get_pinned()
