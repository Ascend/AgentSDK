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
#

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import extensions.skills_ext.cache as cache_module
import extensions.skills_ext.registry_ext as registry_module
from clawcodex_ext.skills.model import Skill
from extensions.skills_ext import init_skills_ext
from extensions.skills_ext.paths import (
    get_clawcodex_user_skills_dirs,
    resolve_skills_paths,
)


def _snapshot(root: Path, skill: Skill, version: int = 1) -> SimpleNamespace:
    def resolve(name: str) -> Skill | None:
        if name == skill.name or name in skill.aliases:
            return skill if skill.is_enabled() else None
        return None

    return SimpleNamespace(
        project_root=str(root),
        diagnostics=(),
        skills=(skill,),
        version=version,
        resolve=resolve,
    )


def test_registry_ext_delegates_to_catalog_without_loader_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    skill = Skill(name="facade-skill", description="test", aliases=["facade"])
    snapshot = _snapshot(tmp_path, skill)
    loader = SimpleNamespace(
        get_all_skills=lambda **kwargs: pytest.fail(f"legacy loader should not be called: {kwargs}")
    )
    calls: list[tuple[Path | None, Path | None]] = []

    def fake_catalog(*, project_root=None, user_skills_dir=None):
        calls.append((project_root, user_skills_dir))
        return snapshot

    monkeypatch.setattr(registry_module, "get_skill_catalog", fake_catalog)

    registry = registry_module.SkillRegistryExt(loader_module=loader, project_root=tmp_path)
    custom_user = tmp_path / "custom-user"
    first = registry.get_all_skills(user_skills_dir=custom_user)
    second = registry.get_all_skills()

    assert first == [skill]
    assert second == [skill]
    assert calls == [(tmp_path, custom_user), (tmp_path, custom_user)]
    assert registry.get_skill("facade") is skill
    assert calls[-1] == (tmp_path, custom_user)
    assert registry.upstream_loader is loader
    assert not hasattr(registry, "_cached_skills")


def test_registry_callbacks_follow_snapshot_versions_and_force_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    skill = Skill(name="facade-skill", description="test")
    current = [_snapshot(tmp_path, skill, version=1)]
    invalidations: list[tuple[str, object | None]] = []

    monkeypatch.setattr(
        registry_module,
        "get_skill_catalog",
        lambda **kwargs: current[0],
    )
    monkeypatch.setattr(
        registry_module,
        "invalidate_skill_catalog",
        lambda reason, workspace=None: invalidations.append((reason, workspace)),
    )

    seen: list[str] = []
    registry = registry_module.SkillRegistryExt(project_root=tmp_path)
    registry.on_skill_registered(lambda registered: seen.append(registered.name))

    registry.get_all_skills()
    registry.get_all_skills()
    assert seen == ["facade-skill"]

    current[0] = _snapshot(tmp_path, skill, version=2)
    registry.get_all_skills(force_refresh=True)

    assert seen == ["facade-skill", "facade-skill"]
    assert invalidations == [("skills_ext force refresh", tmp_path)]


def test_legacy_cache_api_is_stateless_and_forwards_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalidations: list[str] = []
    monkeypatch.setattr(
        cache_module,
        "invalidate_skill_catalog",
        lambda reason: invalidations.append(reason),
    )

    discovery = cache_module.get_discovery_cache()
    registry = cache_module.get_registry_cache()

    assert discovery is registry
    discovery.set("workspace", ["stale"])
    assert discovery.get("workspace") is None
    assert discovery.cleanup_expired() == 0

    discovery.invalidate("workspace")
    cache_module.clear_all_caches()

    assert invalidations == [
        "skills_ext catalog cache key invalidated: workspace",
        "skills_ext clear_all_caches",
    ]


def test_path_facade_projects_canonical_roots_without_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAWCODEX_SKILLS_DIR", "relative-user-skills")
    managed = tmp_path / "managed-skills"
    monkeypatch.setenv("CLAWCODEX_MANAGED_SKILLS_DIR", str(managed))

    clawcodex_user_roots = get_clawcodex_user_skills_dirs()
    assert (tmp_path / "relative-user-skills").resolve() in clawcodex_user_roots

    custom_user = tmp_path / "custom-user"
    paths = resolve_skills_paths(
        project_root=tmp_path,
        user_skills_dir=custom_user,
    )

    assert str(custom_user.resolve()) in paths["user"]
    assert str((tmp_path / ".clawcodex" / "skills").resolve()) in paths["project"]
    assert str((tmp_path / ".claude" / "skills").resolve()) not in paths["project"]
    assert str(managed.resolve()) in paths["managed"]


def test_init_skills_ext_isolates_adapter_failures_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import extensions.skills_ext.bundled as bundled_module

    calls: list[str] = []

    def fail_convert() -> None:
        calls.append("convert")
        raise RuntimeError("invalid converter definition")

    def register_dream() -> None:
        calls.append("dream")

    monkeypatch.setattr(bundled_module, "register_convert_sop_skill", fail_convert)
    monkeypatch.setattr(bundled_module, "register_dream_skill", register_dream)

    with caplog.at_level(logging.WARNING, logger="extensions.skills_ext"):
        init_skills_ext()

    assert calls == ["convert", "dream"]
    assert "failed to register convert-sop-to-agent" in caplog.text
    assert "invalid converter definition" in caplog.text


def test_convert_adapter_uses_canonical_bundled_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extensions.skills_ext.bundled as bundled_module

    definitions: list[object] = []

    def accept(definition: object) -> bool:
        definitions.append(definition)
        return True

    monkeypatch.setattr(bundled_module, "register_bundled_skill", accept)
    bundled_module.register_convert_sop_skill()

    definition = definitions[0]
    assert getattr(definition, "name") == "convert-sop-to-agent"
    assert getattr(definition, "aliases") == ["sop-to-agent"]
    assert getattr(definition, "context") == "inline"
    assert getattr(definition, "disable_model_invocation") is True

    monkeypatch.setattr(
        bundled_module,
        "register_bundled_skill",
        lambda definition: False,
    )
    with pytest.raises(ValueError, match="definition was rejected"):
        bundled_module.register_convert_sop_skill()


def test_catalog_initializes_converter_without_expanding_core_bundled_skills(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from clawcodex_ext.skills import loader
    from clawcodex_ext.skills.bundled import init_bundled_skills
    from clawcodex_ext.skills.bundled_skills import (
        clear_bundled_skills,
        get_bundled_skills,
    )
    from clawcodex_ext.skills.catalog import (
        get_skill_catalog,
        invalidate_skill_catalog,
    )

    core_names = {
        "batch",
        "simplify",
        "debug",
        "loop",
        "orchestrator",
        "stuck",
        "verify-content",
        "update-config",
        "remember",
        "spec-audit",
        "verify",
    }
    clear_bundled_skills()
    invalidate_skill_catalog("extension lifecycle test setup")
    monkeypatch.setattr(loader, "discover_all_skills", lambda **_kwargs: [])

    try:
        assert init_bundled_skills() is True
        assert {skill.name for skill in get_bundled_skills()} == core_names

        workspace_a = tmp_path / "workspace-a"
        workspace_b = tmp_path / "workspace-b"
        workspace_a.mkdir()
        workspace_b.mkdir()

        first_a = get_skill_catalog(project_root=workspace_a)
        converter = first_a.resolve("convert-sop-to-agent")
        assert converter is not None
        assert first_a.resolve("sop-to-agent") is converter
        assert converter.disable_model_invocation is True
        assert {skill.name for skill in first_a.skills if skill.loaded_from == "bundled"} == core_names | {
            "convert-sop-to-agent"
        }

        get_skill_catalog(project_root=workspace_b)
        assert get_skill_catalog(project_root=workspace_a) is first_a

        clear_bundled_skills()
        invalidate_skill_catalog("extension lifecycle reset")
        rebuilt = get_skill_catalog(project_root=workspace_a)
        assert rebuilt.resolve("convert-sop-to-agent") is not None
    finally:
        clear_bundled_skills()
        invalidate_skill_catalog("extension lifecycle test cleanup")


def test_dream_uses_the_common_builtin_command_initialization() -> None:
    from clawcodex_ext.command_system.builtins import (
        get_builtin_commands,
        register_builtin_commands,
    )
    from clawcodex_ext.command_system.registry import CommandRegistry
    from clawcodex_ext.command_system.types import CommandType
    from extensions.skills_ext.bundled.dream import get_dream_command

    builtins = [command for command in get_builtin_commands() if command.name == "dream"]
    assert builtins == [get_dream_command()]

    registry = CommandRegistry()
    register_builtin_commands(registry)

    command = registry.get("dream")
    assert command is get_dream_command()
    assert command.command_type is CommandType.LOCAL


def test_public_compatibility_types_remain_lazy_exports() -> None:
    from extensions.skills_ext import (
        SkillRegistrationCallback,
        SkillRegistryExt,
    )
    from extensions.skills_ext.hooks import (
        SkillRegistrationCallback as CanonicalCallback,
    )

    assert SkillRegistryExt is registry_module.SkillRegistryExt
    assert SkillRegistrationCallback is CanonicalCallback


def test_catalog_records_extension_adapter_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import extensions.skills_ext as extension_module
    from clawcodex_ext.skills import bundled_skills, loader
    from clawcodex_ext.skills.catalog import (
        get_skill_catalog,
        invalidate_skill_catalog,
    )

    invalidate_skill_catalog("extension failure diagnostic setup")
    monkeypatch.setattr(loader, "discover_all_skills", lambda **_kwargs: [])
    monkeypatch.setattr(bundled_skills, "get_registered_bundled_skills", lambda: [])
    monkeypatch.setattr(
        extension_module,
        "init_skill_catalog_extensions",
        lambda: False,
    )

    snapshot = get_skill_catalog(project_root=tmp_path)

    assert any("extension skill adapters failed to initialize" in diagnostic for diagnostic in snapshot.diagnostics)
    invalidate_skill_catalog("extension failure diagnostic cleanup")
