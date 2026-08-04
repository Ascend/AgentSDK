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

"""Public runtime contract for the bundled ``orchestrator`` skill."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from clawcodex_ext.skills.bundled_skills import (
    clear_bundled_skills,
    get_bundled_skill_by_name,
)
from clawcodex_ext.skills.invocation import (
    SkillInvocationOrigin,
    SkillInvocationRequest,
    SkillInvocationService,
)
from clawcodex_ext.tool_system.context import ToolContext


@pytest.fixture(autouse=True)
def _reset_bundled_registry() -> Iterator[None]:
    clear_bundled_skills()
    yield
    clear_bundled_skills()


def test_orchestrator_registers_and_lookup() -> None:
    from clawcodex_ext.skills.bundled.orchestrator import register_orchestrator_skill

    assert register_orchestrator_skill() is True

    skill = get_bundled_skill_by_name("orchestrator")
    assert skill is not None
    assert skill.name == "orchestrator"
    assert skill.source == "bundled"
    assert skill.loaded_from == "bundled"
    assert skill.user_invocable is True
    assert skill.aliases == ["orch"]
    assert skill.allowed_tools == ["Bash", "Read", "Grep", "Glob"]

    # Resolve by alias
    aliased = get_bundled_skill_by_name("orch")
    assert aliased is skill


def test_orchestrator_is_bundled_with_its_complete_portable_resource_tree(
    tmp_path: Path,
) -> None:
    from clawcodex_ext.skills.bundled.orchestrator import register_orchestrator_skill

    register_orchestrator_skill()
    skill = get_bundled_skill_by_name("orchestrator")
    assert skill is not None

    context = ToolContext(workspace_root=tmp_path)
    result = SkillInvocationService(
        resolver=lambda _name, _context: skill,
        recorder=lambda *_args: None,
    ).invoke(
        SkillInvocationRequest(
            "orchestrator",
            args="帮我看看状态",
            origin=SkillInvocationOrigin.USER,
        ),
        context,
    )
    assert result.success is True
    assert result.prompt is not None
    assert result.context_modifier is not None

    assert skill.skill_root is not None
    root = Path(skill.skill_root)
    assert result.prompt.startswith(f"Base directory for this skill: {root}\n\n")
    assert "# /orchestrator" in result.prompt
    assert "编排器管家" in result.prompt
    assert "ARGUMENTS: 帮我看看状态" in result.prompt
    modified_context = result.context_modifier(context)
    assert modified_context.skill_resource_roots == (str(root),)

    required_files = (
        "SKILL.md",
        "references/command-reference.md",
        "references/workflow-config-reference.md",
        "references/workflow.template.md",
        "references/workflow-local.template.md",
        "references/issue-card.template.md",
        "references/workflow.yaml.template",
    )
    for relative_path in required_files:
        assert (root / relative_path).is_file(), relative_path


def test_orchestrator_alias_lookup(tmp_path: Path) -> None:
    from clawcodex_ext.skills.bundled.orchestrator import register_orchestrator_skill

    register_orchestrator_skill()
    skill = get_bundled_skill_by_name("orch")
    assert skill is not None
    assert skill.name == "orchestrator"

    context = ToolContext(workspace_root=tmp_path)
    result = SkillInvocationService(
        resolver=lambda _name, _context: skill,
        recorder=lambda *_args: None,
    ).invoke(
        SkillInvocationRequest(
            "orch",
            args="",
            origin=SkillInvocationOrigin.USER,
        ),
        context,
    )
    assert result.success is True
    assert result.prompt is not None
    assert "编排器管家" in result.prompt


def test_invalid_skill_name_frontmatter_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from clawcodex_ext.skills.bundled import orchestrator as orch_module
    from clawcodex_ext.skills.bundled.resource_loader import load_bundled_text_resources

    original_load = load_bundled_text_resources

    def bad_load(package: str) -> dict[str, str]:
        result = original_load(package)
        result["SKILL.md"] = "---\nname: wrong-name\ndescription: test\n---\n\nbody"
        return result

    monkeypatch.setattr(
        "clawcodex_ext.skills.bundled.orchestrator.load_bundled_text_resources",
        bad_load,
    )
    orch_module._load_portable_skill.cache_clear()

    with pytest.raises(ValueError, match="invalid name frontmatter"):
        orch_module.register_orchestrator_skill()


def test_missing_description_frontmatter_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from clawcodex_ext.skills.bundled import orchestrator as orch_module
    from clawcodex_ext.skills.bundled.resource_loader import load_bundled_text_resources

    original_load = load_bundled_text_resources

    def bad_load(package: str) -> dict[str, str]:
        result = original_load(package)
        result["SKILL.md"] = "---\nname: orchestrator\n---\n\nbody"
        return result

    monkeypatch.setattr(
        "clawcodex_ext.skills.bundled.orchestrator.load_bundled_text_resources",
        bad_load,
    )
    orch_module._load_portable_skill.cache_clear()

    with pytest.raises(ValueError, match="no description frontmatter"):
        orch_module.register_orchestrator_skill()


def test_empty_prompt_body_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from clawcodex_ext.skills.bundled import orchestrator as orch_module
    from clawcodex_ext.skills.bundled.resource_loader import load_bundled_text_resources

    original_load = load_bundled_text_resources

    def bad_load(package: str) -> dict[str, str]:
        result = original_load(package)
        result["SKILL.md"] = "---\nname: orchestrator\ndescription: test\n---\n\n   \n"
        return result

    monkeypatch.setattr(
        "clawcodex_ext.skills.bundled.orchestrator.load_bundled_text_resources",
        bad_load,
    )
    orch_module._load_portable_skill.cache_clear()

    with pytest.raises(ValueError, match="empty prompt body"):
        orch_module.register_orchestrator_skill()


def test_missing_skill_md_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from clawcodex_ext.skills.bundled import orchestrator as orch_module
    from clawcodex_ext.skills.bundled.resource_loader import load_bundled_text_resources

    original_load = load_bundled_text_resources

    def bad_load(package: str) -> dict[str, str]:
        result = original_load(package)
        return {k: v for k, v in result.items() if k != "SKILL.md"}

    monkeypatch.setattr(
        "clawcodex_ext.skills.bundled.orchestrator.load_bundled_text_resources",
        bad_load,
    )
    orch_module._load_portable_skill.cache_clear()

    with pytest.raises(FileNotFoundError, match="does not contain SKILL.md"):
        orch_module.register_orchestrator_skill()
