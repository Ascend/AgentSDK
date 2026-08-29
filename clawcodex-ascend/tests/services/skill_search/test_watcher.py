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

"""Tests for P92-E SkillIndexWatcher.

Covers:
    - ``_on_skill_registered``: upserts document into index
    - Hidden skill: silently skipped
    - Update existing: same skill re-registered with different content
    - Cooldown save: multiple registrations batched into single save
    - ``stop()``: no longer responds to callbacks
    - ``start()`` when ``enabled=False``: no callback registered
    - Index not ready: callback does not raise
    - Thread safety: concurrent upserts leave index consistent
    - ``create_watcher()`` factory: returns properly configured watcher
"""
# pylint: disable=no-name-in-module

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from clawcodex_ext.services.skill_search.config import SkillSearchConfig
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
    """Create a mock SkillRegistryExt that returns the given skills.

    The mock also tracks ``on_skill_registered`` / ``off_skill_registered``
    calls so we can trigger callbacks manually in tests.
    """
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
    import tempfile
    from pathlib import Path

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
# TestWatcherUpsert
# ============================================================================


class TestWatcherUpsert:
    @pytest.mark.asyncio
    async def test_upserts_on_register(self):
        """A registered skill is extracted and upserted into the index."""
        skills = [
            _make_skill(name="browser", description="browser automation"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        watcher = SkillIndexWatcher(searcher, searcher._registry, config=searcher._config)
        watcher.start()

        new_skill = _make_skill(
            name="git_helper",
            description="git commit and push helper",
        )
        searcher._registry._notify(new_skill)

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 2

        results = await searcher.search("git commit")
        assert len(results) >= 1
        assert any(r.document.name == "git_helper" for r in results)

    @pytest.mark.asyncio
    async def test_skips_hidden_skill(self):
        """Hidden skills are filtered out by extract_search_document."""
        skills = [
            _make_skill(name="browser", description="browser automation"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        watcher = SkillIndexWatcher(searcher, searcher._registry, config=searcher._config)
        watcher.start()

        hidden = _make_skill(
            name="hidden_skill",
            description="should not be indexed",
            is_hidden=True,
        )
        searcher._registry._notify(hidden)

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 1

    @pytest.mark.asyncio
    async def test_updates_existing(self):
        """Registering the same skill with new content replaces the old entry."""
        skills = [
            _make_skill(name="browser", description="old description"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        watcher = SkillIndexWatcher(searcher, searcher._registry, config=searcher._config)
        watcher.start()

        updated = _make_skill(
            name="browser",
            description="new playwright automation description",
        )
        searcher._registry._notify(updated)

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 1

        results = await searcher.search("playwright")
        assert len(results) >= 1
        assert results[0].document.name == "browser"


# ============================================================================
# TestWatcherCooldownSave
# ============================================================================


class TestWatcherCooldownSave:
    @pytest.mark.asyncio
    async def test_cooldown_save(self):
        """Multiple rapid registrations trigger only one save."""
        skills = [
            _make_skill(name="base", description="base skill"),
        ]
        config = _make_config(save_cooldown_seconds=5)
        registry = _make_registry(skills)
        tokenizer = create_default_tokenizer(cjk_word_tokenizer=None)
        searcher = SkillSearcher(registry, config=config, tokenizer=tokenizer)
        await searcher.ensure_index()

        watcher = SkillIndexWatcher(searcher, registry, config=config)
        watcher.start()

        with patch.object(searcher._index, "save", wraps=searcher._index.save) as mock_save:
            for i in range(10):
                skill = _make_skill(
                    name=f"skill_{i}",
                    description=f"skill number {i}",
                )
                registry._notify(skill)

            # All registrations happen within the same cooldown window
            assert mock_save.call_count == 1

    @pytest.mark.asyncio
    async def test_cooldown_save_multiple_windows(self):
        """Registrations spread across cooldown windows trigger multiple saves."""
        skills = [
            _make_skill(name="base", description="base skill"),
        ]
        config = _make_config(save_cooldown_seconds=0)  # no cooldown
        registry = _make_registry(skills)
        tokenizer = create_default_tokenizer(cjk_word_tokenizer=None)
        searcher = SkillSearcher(registry, config=config, tokenizer=tokenizer)
        await searcher.ensure_index()

        watcher = SkillIndexWatcher(searcher, registry, config=config)
        watcher.start()

        with patch.object(searcher._index, "save", wraps=searcher._index.save) as mock_save:
            for i in range(5):
                skill = _make_skill(
                    name=f"skill_{i}",
                    description=f"skill number {i}",
                )
                registry._notify(skill)

            # With cooldown=0, every registration triggers a save
            assert mock_save.call_count == 5


# ============================================================================
# TestWatcherStop
# ============================================================================


class TestWatcherStop:
    @pytest.mark.asyncio
    async def test_stop_no_longer_updates(self):
        """After stop(), callbacks no longer modify the index."""
        skills = [
            _make_skill(name="browser", description="browser automation"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        watcher = SkillIndexWatcher(searcher, searcher._registry, config=searcher._config)
        watcher.start()
        watcher.stop()

        new_skill = _make_skill(
            name="git_helper",
            description="git commit helper",
        )
        searcher._registry._notify(new_skill)

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 1

    @pytest.mark.asyncio
    async def test_start_stop_idempotent(self):
        """Repeated start/stop calls are no-ops."""
        skills = [
            _make_skill(name="browser", description="browser"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        watcher = SkillIndexWatcher(searcher, searcher._registry, config=searcher._config)
        watcher.start()
        watcher.start()
        watcher.stop()
        watcher.stop()

        # Should not raise and should still work
        assert watcher._active is False


# ============================================================================
# TestWatcherDisabled
# ============================================================================


class TestWatcherDisabled:
    @pytest.mark.asyncio
    async def test_start_when_disabled(self):
        """When config.enabled=False, start() does not register a callback."""
        config = _make_config(enabled=False)
        registry = _make_registry([])
        tokenizer = create_default_tokenizer(cjk_word_tokenizer=None)
        searcher = SkillSearcher(registry, config=config, tokenizer=tokenizer)

        watcher = SkillIndexWatcher(searcher, registry, config=config)
        watcher.start()

        assert watcher._active is False
        assert len(registry._callbacks) == 0


# ============================================================================
# TestWatcherIndexNotReady
# ============================================================================


class TestWatcherIndexNotReady:
    @pytest.mark.asyncio
    async def test_index_not_ready_no_error(self):
        """When the index is not yet loaded, the callback silently skips."""
        skills = [
            _make_skill(name="browser", description="browser"),
        ]
        searcher = _make_searcher(skills)
        # Do NOT call ensure_index() — index is None

        watcher = SkillIndexWatcher(searcher, searcher._registry, config=searcher._config)
        watcher.start()

        # Should not raise
        new_skill = _make_skill(name="git_helper", description="git helper")
        searcher._registry._notify(new_skill)

        stats = searcher.stats()
        assert stats is None


# ============================================================================
# TestWatcherThreadSafety
# ============================================================================


class TestWatcherThreadSafety:
    @pytest.mark.asyncio
    async def test_concurrent_upserts(self):
        """Multiple threads concurrently upserting skills leave index consistent."""
        skills = [
            _make_skill(name="base", description="base skill"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        watcher = SkillIndexWatcher(searcher, searcher._registry, config=searcher._config)
        watcher.start()

        errors: list[Exception] = []

        def register_skills(start: int, count: int):
            try:
                for i in range(start, start + count):
                    skill = _make_skill(
                        name=f"thread_skill_{i}",
                        description=f"thread skill number {i}",
                    )
                    searcher._registry._notify(skill)
            except Exception as e:
                errors.append(e)

        threads = []
        for t in range(4):
            thread = threading.Thread(
                target=register_skills,
                args=(t * 25, 25),
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0

        stats = searcher.stats()
        assert stats is not None
        # 1 base + 4 * 25 = 101
        assert stats.total_docs == 101

    @pytest.mark.asyncio
    async def test_concurrent_same_skill(self):
        """Concurrent updates to the same skill leave it in a consistent state."""
        skills = [
            _make_skill(name="concurrent", description="initial"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        watcher = SkillIndexWatcher(searcher, searcher._registry, config=searcher._config)
        watcher.start()

        def register_variants():
            for i in range(10):
                skill = _make_skill(
                    name="concurrent",
                    description=f"variant {i}",
                )
                searcher._registry._notify(skill)

        threads = [threading.Thread(target=register_variants) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 1


# ============================================================================
# TestCreateWatcher
# ============================================================================


class TestCreateWatcher:
    @pytest.mark.asyncio
    async def test_create_watcher_returns_watcher(self):
        """create_watcher() returns a properly configured SkillIndexWatcher."""
        skills = [
            _make_skill(name="browser", description="browser"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        watcher = searcher.create_watcher()
        assert isinstance(watcher, SkillIndexWatcher)
        assert watcher._searcher is searcher
        assert watcher._registry is searcher._registry
        assert watcher._config is searcher._config
        assert watcher._active is False

    @pytest.mark.asyncio
    async def test_create_watcher_not_started(self):
        """create_watcher() does not auto-start the watcher."""
        skills = [
            _make_skill(name="browser", description="browser"),
        ]
        searcher = _make_searcher(skills)
        await searcher.ensure_index()

        watcher = searcher.create_watcher()
        assert watcher._active is False

        # Registering a skill should NOT affect the index
        new_skill = _make_skill(name="git_helper", description="git helper")
        searcher._registry._notify(new_skill)

        stats = searcher.stats()
        assert stats is not None
        assert stats.total_docs == 1
