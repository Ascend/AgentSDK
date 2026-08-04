#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Lifecycle contracts for the workspace-scoped skill catalog.

These tests intentionally exercise the public context-oriented API.  The
catalog is consumed by long-lived REPL, TUI, and query sessions, so invalidating
one workspace must not throw away immutable snapshots owned by another.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Iterator

import pytest

from clawcodex_ext.command_system import aggregator
from clawcodex_ext.context_system.prompt_assembly import get_system_prompt_cache
from clawcodex_ext.skills import bundled_skills, catalog, loader, mcp_skill_builders
from clawcodex_ext.skills.catalog import get_skill_catalog, invalidate_skill_catalog
from clawcodex_ext.skills.model import Skill


@pytest.fixture(autouse=True)
def _clear_catalog_and_dependent_caches(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    adapter = ModuleType("extensions.skills_ext")

    def initialize_extensions() -> bool:
        return True

    adapter.init_skill_catalog_extensions = initialize_extensions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "extensions.skills_ext", adapter)
    loader.clear_skill_caches()
    catalog._invalidate_catalog_cache_only()
    aggregator.clear_commands_cache()
    get_system_prompt_cache().invalidate_all()
    yield
    loader.clear_skill_caches()
    catalog._invalidate_catalog_cache_only()
    aggregator.clear_commands_cache()
    get_system_prompt_cache().invalidate_all()


def _context(workspace: Path, session_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        workspace_root=workspace,
        cwd=workspace,
        user_skills_dir=None,
        session_id=session_id,
    )


def test_context_catalog_and_targeted_invalidation_are_workspace_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()

    load_calls: list[Path] = []

    def load_for_workspace(**kwargs: object) -> list[Skill]:
        root = Path(str(kwargs["project_root"])).resolve()
        load_calls.append(root)
        return [Skill(name=f"skill-{root.name}", description=root.name)]

    monkeypatch.setattr(loader, "discover_all_skills", load_for_workspace)
    monkeypatch.setattr(bundled_skills, "get_registered_bundled_skills", lambda: [])

    first_a = get_skill_catalog(_context(workspace_a))
    first_b = get_skill_catalog(_context(workspace_b))
    assert first_a.resolve("skill-workspace-a") is not None
    assert first_b.resolve("skill-workspace-b") is not None

    invalidate_skill_catalog("project skill changed", workspace=workspace_a)

    second_a = get_skill_catalog(_context(workspace_a))
    second_b = get_skill_catalog(_context(workspace_b))
    assert second_a is not first_a
    assert second_a.version > first_a.version
    assert second_b is first_b
    assert load_calls.count(workspace_a.resolve()) == 2
    assert load_calls.count(workspace_b.resolve()) == 1


def test_public_invalidation_clears_command_and_skill_prompt_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loader, "discover_all_skills", lambda **_kwargs: [])
    aggregator.get_commands(tmp_path)
    assert aggregator._load_skill_commands_cached.cache_info().currsize > 0

    prompt_cache = get_system_prompt_cache()
    prompt_cache.set("skills:8000", "stale available-skills listing")
    prompt_cache.set("unrelated", "also invalidated by the shared prompt cache")
    assert prompt_cache.get("skills:8000") is not None

    invalidate_skill_catalog("dynamic skill registration", workspace=tmp_path)

    assert aggregator._load_skill_commands_cached.cache_info().currsize == 0
    assert prompt_cache.get("skills:8000") is None
    assert prompt_cache.get("unrelated") is not None


def test_failed_bundled_initialization_is_retried_without_manual_invalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def initialize() -> bool:
        nonlocal attempts
        attempts += 1
        return attempts > 1

    monkeypatch.setattr("clawcodex_ext.skills.bundled.init_bundled_skills", initialize)
    monkeypatch.setattr(loader, "discover_all_skills", lambda **_kwargs: [])
    monkeypatch.setattr(bundled_skills, "get_registered_bundled_skills", lambda: [])

    first = get_skill_catalog(_context(tmp_path))
    assert any("retry is enabled" in item for item in first.diagnostics)

    second = get_skill_catalog(_context(tmp_path))
    assert second is not first
    assert second.diagnostics == ()
    assert attempts == 2


def test_mcp_builder_registration_invalidates_existing_catalog_and_is_copy_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loader, "discover_all_skills", lambda **_kwargs: [])
    monkeypatch.setattr(bundled_skills, "get_registered_bundled_skills", lambda: [])
    monkeypatch.setattr(mcp_skill_builders, "_builders", None)

    first = get_skill_catalog(_context(tmp_path))

    def builder() -> list[Skill]:
        return []

    mcp_skill_builders.register_mcp_skill_builders({"server": builder})
    second = get_skill_catalog(_context(tmp_path))

    assert second is not first
    returned = mcp_skill_builders.get_mcp_skill_builders()
    assert returned == {"server": builder}
    assert returned is not None
    returned.clear()
    assert mcp_skill_builders.get_mcp_skill_builders() == {"server": builder}


def test_catalog_snapshots_are_isolated_by_session_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def discover(**_kwargs: object) -> list[Skill]:
        nonlocal calls
        calls += 1
        return [Skill(name="shared", description="shared")]

    monkeypatch.setattr(loader, "discover_all_skills", discover)
    monkeypatch.setattr(bundled_skills, "get_registered_bundled_skills", lambda: [])

    session_a = get_skill_catalog(_context(tmp_path, "session-a"))
    session_b = get_skill_catalog(_context(tmp_path, "session-b"))

    assert session_a is not session_b
    assert session_a.session_id == "session-a"
    assert session_b.session_id == "session-b"
    assert calls == 2
    assert get_skill_catalog(_context(tmp_path, "session-a")) is session_a


def _write_skill(root: Path, name: str, body: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\ndescription: {name}\n---\n{body}\n",
        encoding="utf-8",
    )
    return directory


def test_dynamic_skill_state_and_invalidation_are_workspace_scoped(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    dynamic_a = tmp_path / "dynamic-a"
    dynamic_b = tmp_path / "dynamic-b"
    _write_skill(dynamic_a, "same-dynamic", "FROM A")
    _write_skill(dynamic_b, "same-dynamic", "FROM B")

    loader.add_skill_directories([str(dynamic_a)], project_root=workspace_a)
    loader.add_skill_directories([str(dynamic_b)], project_root=workspace_b)

    scoped_a = loader.get_dynamic_skills(workspace_a)
    scoped_b = loader.get_dynamic_skills(workspace_b)
    assert [skill.markdown_content.strip() for skill in scoped_a] == ["FROM A"]
    assert [skill.markdown_content.strip() for skill in scoped_b] == ["FROM B"]

    snapshot_a = get_skill_catalog(_context(workspace_a, "session-a"))
    snapshot_b = get_skill_catalog(_context(workspace_b, "session-b"))
    assert snapshot_a.resolve("same-dynamic").markdown_content.strip() == "FROM A"
    assert snapshot_b.resolve("same-dynamic").markdown_content.strip() == "FROM B"

    invalidate_skill_catalog("workspace A changed", workspace=workspace_a)

    assert loader.get_dynamic_skills(workspace_a) == []
    assert [skill.markdown_content.strip() for skill in loader.get_dynamic_skills(workspace_b)] == ["FROM B"]
    assert get_skill_catalog(_context(workspace_b, "session-b")) is snapshot_b


def test_bare_catalog_only_keeps_additional_bundled_and_mcp_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    additional = tmp_path / "additional"
    user = tmp_path / "legacy-user"
    managed = tmp_path / "managed"
    dynamic = tmp_path / "dynamic"

    _write_skill(workspace / ".clawcodex" / "skills", "project-standard", "PROJECT")
    _write_skill(workspace / ".claude" / "skills", "project-legacy", "LEGACY PROJECT")
    _write_skill(
        additional / ".clawcodex" / "skills",
        "explicit-additional",
        "ADDITIONAL",
    )
    _write_skill(user, "legacy-user", "LEGACY USER")
    _write_skill(managed, "managed-extra", "MANAGED")
    _write_skill(dynamic, "dynamic-hidden", "DYNAMIC")

    monkeypatch.setenv("CLAUDE_CODE_BARE_MODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ADDITIONAL_DIRECTORIES", str(additional))
    monkeypatch.setenv("CLAWCODEX_MANAGED_SKILLS_DIR", str(managed))
    monkeypatch.setattr(
        mcp_skill_builders,
        "_builders",
        {"probe": lambda: [Skill(name="mcp-visible", description="mcp", loaded_from="mcp")]},
    )
    loader.add_skill_directories([str(dynamic)], project_root=workspace)

    snapshot = get_skill_catalog(
        _context(workspace, "bare-session"),
        user_skills_dir=user,
    )
    names = {skill.name for skill in snapshot.skills}

    assert {
        "simplify",
        "debug",
        "loop",
        "orchestrator",
        "spec-audit",
        "stuck",
        "verify-content",
        "update-config",
    } <= names
    assert {"explicit-additional", "mcp-visible"} <= names
    assert {
        "project-standard",
        "project-legacy",
        "legacy-user",
        "managed-extra",
        "dynamic-hidden",
    }.isdisjoint(names)
