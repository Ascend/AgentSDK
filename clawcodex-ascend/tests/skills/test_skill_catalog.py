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

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType
from typing import Iterator

import pytest

from clawcodex_ext.skills import bundled_skills, loader
from clawcodex_ext.skills.catalog import (
    get_skill_catalog,
    invalidate_skill_catalog,
    resolve,
)
from clawcodex_ext.skills.model import Skill


@pytest.fixture(autouse=True)
def _clean_catalog(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    adapter = ModuleType("extensions.skills_ext")

    def initialize_extensions() -> bool:
        return True

    adapter.init_skill_catalog_extensions = initialize_extensions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "extensions.skills_ext", adapter)
    invalidate_skill_catalog()
    yield
    invalidate_skill_catalog()


def _stub_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loaded: list[Skill],
    bundled: list[Skill] | None = None,
) -> None:
    monkeypatch.setattr(loader, "discover_all_skills", lambda **_kwargs: list(loaded))
    monkeypatch.setattr(
        bundled_skills,
        "get_registered_bundled_skills",
        lambda: list(bundled or []),
    )


def test_snapshot_is_immutable_and_bundled_canonical_name_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disk = Skill(name="reserved", description="disk", loaded_from="project")
    builtin = Skill(name="reserved", description="builtin", loaded_from="bundled")
    _stub_sources(monkeypatch, loaded=[disk], bundled=[builtin])

    snapshot = get_skill_catalog(project_root=tmp_path)

    assert snapshot.skills == (builtin,)
    assert snapshot.resolve("reserved") is builtin
    with pytest.raises(FrozenInstanceError):
        snapshot.version = 99  # type: ignore[misc]
    with pytest.raises(TypeError):
        snapshot.canonical["other"] = disk  # type: ignore[index]


def test_canonical_name_always_wins_over_an_earlier_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias_owner = Skill(
        name="first",
        description="owns alias",
        aliases=["target", "shortcut"],
    )
    canonical = Skill(name="target", description="canonical")
    _stub_sources(monkeypatch, loaded=[alias_owner, canonical])

    snapshot = get_skill_catalog(project_root=tmp_path)

    assert snapshot.resolve("target") is canonical
    assert snapshot.resolve("shortcut") is alias_owner
    assert "target" not in snapshot.aliases


def test_enabled_predicate_is_evaluated_on_every_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"enabled": False, "calls": 0}

    def is_enabled() -> bool:
        state["calls"] += 1
        return bool(state["enabled"])

    skill = Skill(name="gated", description="gated", is_enabled_fn=is_enabled)
    _stub_sources(monkeypatch, loaded=[skill])
    snapshot = get_skill_catalog(project_root=tmp_path)

    assert snapshot.resolve("gated") is None
    state["enabled"] = True
    assert snapshot.resolve("gated") is skill
    assert state["calls"] == 2
    assert snapshot.resolve("gated", include_disabled=True) is skill
    assert state["calls"] == 2


def test_public_resolve_uses_cached_workspace_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    skill = Skill(name="cached", description="cached")

    def load(**_kwargs: object) -> list[Skill]:
        nonlocal calls
        calls += 1
        return [skill]

    monkeypatch.setattr(loader, "discover_all_skills", load)
    monkeypatch.setattr(bundled_skills, "get_registered_bundled_skills", lambda: [])

    first = get_skill_catalog(project_root=tmp_path)
    second = get_skill_catalog(project_root=tmp_path)
    assert first is second
    assert resolve("cached", project_root=tmp_path) is skill
    assert calls == 1

    invalidate_skill_catalog()
    third = get_skill_catalog(project_root=tmp_path)
    assert third is not first
    assert third.version > first.version
    assert calls == 2


def test_legacy_loader_cache_clear_also_invalidates_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_sources(
        monkeypatch,
        loaded=[Skill(name="legacy-clear", description="legacy")],
    )
    first = get_skill_catalog(project_root=tmp_path)

    loader.clear_skill_caches()

    assert get_skill_catalog(project_root=tmp_path) is not first


def test_legacy_loader_facade_exports_private_compatibility() -> None:
    from src.skills.loader import (
        _compile_path_spec,
        _coerce_allowed_tools,
        _skill_registry,
        discover_all_skills,
    )

    assert callable(_compile_path_spec)
    assert callable(_coerce_allowed_tools)
    assert discover_all_skills is loader.discover_all_skills
    assert isinstance(_skill_registry, dict)


def test_catalog_discovery_does_not_replace_legacy_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = Skill(name="legacy-only", description="legacy")
    catalog_only = Skill(name="catalog-only", description="catalog")
    loader.clear_skill_registry()
    loader._skill_registry[legacy.name] = legacy
    _stub_sources(monkeypatch, loaded=[catalog_only])

    snapshot = get_skill_catalog(project_root=tmp_path)

    assert snapshot.resolve("catalog-only") is catalog_only
    assert loader.get_registered_skill("legacy-only") is legacy
    assert loader.get_registered_skill("catalog-only") is None
