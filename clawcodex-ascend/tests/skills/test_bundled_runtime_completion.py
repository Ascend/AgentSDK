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

"""Runtime regressions for core bundled-skill registration and resources."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pytest

from clawcodex_ext.command_system.aggregator import clear_commands_cache
from clawcodex_ext.skills.bundled_skills import (
    BundledSkillDefinition,
    clear_bundled_skills,
    get_bundled_skill_by_name,
    get_bundled_skills,
    register_bundled_skill,
)
from clawcodex_ext.skills.loader import (
    clear_dynamic_skills,
    clear_skill_caches,
    clear_skill_registry,
)


@pytest.fixture(autouse=True)
def _isolated_skill_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(home / ".clawcodex"))
    monkeypatch.setenv("CLAWCODEX_MANAGED_CONFIG_DIR", str(tmp_path / "managed"))
    for name in (
        "CLAWCODEX_SKILLS_DIR",
        "CLAUDE_SKILLS_DIR",
        "CLAWCODEX_MANAGED_SKILLS_DIR",
        "CLAUDE_CODE_ADDITIONAL_DIRECTORIES",
        "CLAUDE_CODE_BARE_MODE",
        "CLAUDE_CODE_DISABLE_POLICY_SKILLS",
    ):
        monkeypatch.delenv(name, raising=False)

    clear_commands_cache()
    clear_skill_caches()
    clear_dynamic_skills()
    clear_skill_registry()
    clear_bundled_skills()
    yield
    clear_commands_cache()
    clear_skill_caches()
    clear_dynamic_skills()
    clear_skill_registry()
    clear_bundled_skills()


def _unique_skill_name(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def test_external_registration_before_first_read_does_not_force_core_catalogue() -> None:
    extension_name = _unique_skill_name("extension")
    register_bundled_skill(
        BundledSkillDefinition(
            name=extension_name,
            description="extension-provided bundled skill",
            get_prompt_for_command=lambda args: f"extension body: {args}",
        )
    )

    names = {skill.name for skill in get_bundled_skills()}
    # An explicit registration deliberately seeds an isolated catalogue. Core
    # expansion resumes after clear_bundled_skills() re-arms lazy init.
    assert names == {extension_name}


def test_bundled_definition_maps_runtime_fields_and_enabled_gate() -> None:
    enabled = False
    name = _unique_skill_name("metadata")
    hooks = {"PreToolUse": [{"matcher": {}, "hooks": [{"type": "command"}]}]}

    register_bundled_skill(
        BundledSkillDefinition(
            name=name,
            description="metadata mapping probe",
            aliases=["metadata-alias"],
            when_to_use="when metadata needs verification",
            argument_hint="<target>",
            allowed_tools=["Read", "Grep"],
            model="test-model",
            disable_model_invocation=True,
            user_invocable=False,
            is_enabled=lambda: enabled,
            context="fork",
            agent="Explore",
            hooks=hooks,
            get_prompt_for_command=lambda args: f"metadata body: {args}",
        )
    )

    skill = get_bundled_skill_by_name(name)
    assert skill is not None
    assert skill.aliases == ["metadata-alias"]
    assert skill.when_to_use == "when metadata needs verification"
    assert skill.argument_hint == "<target>"
    assert skill.allowed_tools == ["Read", "Grep"]
    assert skill.model == "test-model"
    assert skill.disable_model_invocation is True
    assert skill.user_invocable is False
    assert skill.is_hidden is True
    assert skill.context == "fork"
    assert skill.agent == "Explore"
    assert skill.hooks == hooks
    assert skill.has_user_specified_description is True
    assert skill.is_enabled() is False

    enabled = True
    assert skill.is_enabled() is True


def test_bundled_files_extract_lazily_prefix_prompt_and_reuse_root() -> None:
    name = _unique_skill_name("assets")
    register_bundled_skill(
        BundledSkillDefinition(
            name=name,
            description="bundled reference files",
            files={
                "references/guide.md": "# Bundled guide\n",
                "scripts/check.txt": "check payload\n",
            },
            get_prompt_for_command=lambda args: f"ASSET BODY: {args}",
        )
    )

    skill = get_bundled_skill_by_name(name)
    assert skill is not None
    assert skill.skill_root is not None
    root = Path(skill.skill_root)
    assert not root.exists()

    first_prompt = skill.get_prompt("first")
    assert root.is_dir()
    assert (root / "references" / "guide.md").read_text(encoding="utf-8") == ("# Bundled guide\n")
    assert (root / "scripts" / "check.txt").read_text(encoding="utf-8") == ("check payload\n")
    assert first_prompt.startswith(f"Base directory for this skill: {root}\n\n")
    assert "ASSET BODY: first" in first_prompt

    second_prompt = skill.get_prompt("second")
    assert Path(skill.skill_root) == root
    assert second_prompt.startswith(f"Base directory for this skill: {root}\n\n")
    assert "ASSET BODY: second" in second_prompt


def test_bundled_file_traversal_failure_degrades_to_unprefixed_prompt() -> None:
    name = _unique_skill_name("traversal")
    escaped_name = f"escaped-{uuid4().hex}.txt"
    register_bundled_skill(
        BundledSkillDefinition(
            name=name,
            description="unsafe bundled file path probe",
            files={f"../{escaped_name}": "must not escape"},
            get_prompt_for_command=lambda _args: "SAFE FALLBACK BODY",
        )
    )

    skill = get_bundled_skill_by_name(name)
    assert skill is not None
    assert skill.skill_root is not None
    root = Path(skill.skill_root)
    escaped_path = root.parent / escaped_name

    prompt = skill.get_prompt("")
    assert prompt == "SAFE FALLBACK BODY"
    assert "Base directory for this skill:" not in prompt
    assert not escaped_path.exists()
