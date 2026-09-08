#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Tests for the process-wide shared SkillSearcher access point.

Covers:
    - ``get_default_searcher()`` caches one instance per process
    - The ``/skills`` command surface and the ``SkillSearch`` tool resolve
      the same instance and share one underlying index
    - When the feature gate is enabled, exactly one watcher is started
    - Exit flush: ``flush_default_searcher()`` persists pending updates
"""
# pylint: disable=no-name-in-module

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from clawcodex_ext.services.skill_search import searcher as searcher_mod
from clawcodex_ext.services.skill_search.config import SkillSearchConfig
from clawcodex_ext.services.skill_search.searcher import SkillSearcher


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


def _notifying_registry(skills: list | None = None):
    """Return a mock registry whose watcher callbacks can be triggered."""
    registry = MagicMock()
    registry.get_all_skills.return_value = list(skills or [])
    callbacks: list = []
    registry.on_skill_registered.side_effect = callbacks.append
    registry.off_skill_registered.side_effect = callbacks.remove
    registry._callbacks = callbacks

    def _notify(skill):
        for cb in list(callbacks):
            cb(skill)

    registry._notify = _notify
    return registry


@pytest.fixture(autouse=True)
def _isolate_shared_searcher():
    """Reset the module-level singleton around every test in this file."""
    searcher_mod.reset_default_searcher()
    yield
    searcher_mod.reset_default_searcher()


def _disable_feature_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        SkillSearchConfig,
        "from_feature_gate",
        lambda: SkillSearchConfig(enabled=False),
    )
    monkeypatch.setattr(searcher_mod, "get_default_registry", lambda: _notifying_registry())


class TestSharedSearcherIdentity:
    def test_get_default_searcher_is_cached(self, monkeypatch):
        _disable_feature_gate(monkeypatch)

        first = searcher_mod.get_default_searcher()
        assert isinstance(first, SkillSearcher)
        assert searcher_mod.get_default_searcher() is first

    def test_command_and_tool_surfaces_share_one_instance(self, monkeypatch):
        """/skills commands and the SkillSearch tool resolve the same searcher."""
        _disable_feature_gate(monkeypatch)

        from clawcodex_ext.command_system.builtins import _get_skills_searcher
        from clawcodex_ext.tool_system.tools.skill_search import _get_searcher

        shared = searcher_mod.get_default_searcher()
        assert _get_skills_searcher() is shared
        assert _get_searcher() is shared
        assert _get_skills_searcher() is _get_searcher()

    async def test_surfaces_share_one_underlying_index(self, monkeypatch, tmp_path):
        """An index built through one surface is visible through the other."""
        config = SkillSearchConfig(enabled=True, index_path=tmp_path / "skill_index.json")
        registry = _notifying_registry([_make_skill(name="browser", description="browser automation")])
        monkeypatch.setattr(SkillSearchConfig, "from_feature_gate", lambda: config)
        monkeypatch.setattr(searcher_mod, "get_default_registry", lambda: registry)

        from clawcodex_ext.command_system.builtins import _get_skills_searcher
        from clawcodex_ext.tool_system.tools.skill_search import _get_searcher

        command_searcher = _get_skills_searcher()
        tool_searcher = _get_searcher()
        assert command_searcher is tool_searcher

        await command_searcher.ensure_index()
        stats = tool_searcher.stats()
        assert stats is not None
        assert stats.total_docs == 1


class TestSharedWatcher:
    def test_enabled_shared_searcher_starts_single_watcher(self, monkeypatch, tmp_path):
        config = SkillSearchConfig(enabled=True, index_path=tmp_path / "skill_index.json")
        registry = _notifying_registry()
        monkeypatch.setattr(SkillSearchConfig, "from_feature_gate", lambda: config)
        monkeypatch.setattr(searcher_mod, "get_default_registry", lambda: registry)

        first = searcher_mod.get_default_searcher()
        watcher = searcher_mod._default_watcher
        assert watcher is not None
        assert watcher._active is True
        assert len(registry._callbacks) == 1

        # Repeated access reuses the cache — no second watcher is registered.
        assert searcher_mod.get_default_searcher() is first
        assert len(registry._callbacks) == 1

    def test_reset_stops_watcher_and_clears_cache(self, monkeypatch, tmp_path):
        config = SkillSearchConfig(enabled=True, index_path=tmp_path / "skill_index.json")
        registry = _notifying_registry()
        monkeypatch.setattr(SkillSearchConfig, "from_feature_gate", lambda: config)
        monkeypatch.setattr(searcher_mod, "get_default_registry", lambda: registry)

        searcher_mod.get_default_searcher()
        assert len(registry._callbacks) == 1

        searcher_mod.reset_default_searcher()
        assert searcher_mod._default_searcher is None
        assert searcher_mod._default_watcher is None
        assert len(registry._callbacks) == 0

        # A fresh call builds a new searcher rather than reusing the old one.
        assert searcher_mod.get_default_searcher() is not None


class TestExitFlush:
    async def test_flush_default_searcher_persists_pending_updates(self, monkeypatch, tmp_path):
        """Exit flush writes out watcher updates still inside the cooldown."""
        config = SkillSearchConfig(
            enabled=True,
            save_cooldown_seconds=60,
            index_path=tmp_path / "skill_index.json",
        )
        registry = _notifying_registry([_make_skill(name="base", description="base skill")])
        monkeypatch.setattr(SkillSearchConfig, "from_feature_gate", lambda: config)
        monkeypatch.setattr(searcher_mod, "get_default_registry", lambda: registry)

        searcher = searcher_mod.get_default_searcher()
        await searcher.ensure_index()

        def _persisted_names() -> set[str]:
            data = json.loads(config.index_path.read_text(encoding="utf-8"))
            return {d["name"] for d in data["doc_store"].values()}

        # First watcher save: the initial cooldown window has already expired.
        registry._notify(_make_skill(name="first", description="first arrival"))
        assert "first" in _persisted_names()

        # Second registration lands inside the cooldown window: only held in
        # memory until the exit flush.
        registry._notify(_make_skill(name="late", description="late arrival"))
        assert "late" not in _persisted_names()

        searcher_mod.flush_default_searcher()
        assert "late" in _persisted_names()
