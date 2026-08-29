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

"""Security and compatibility contracts for bundled-skill resources."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import stat
from typing import Iterator
from uuid import uuid4

import pytest

from clawcodex_ext.skills.bundled_skills import (
    BundledSkillDefinition,
    clear_bundled_skills,
    get_bundled_skill_by_name,
    get_bundled_skill_extract_dir,
    get_bundled_skills,
    get_bundled_skills_root,
    is_bundled_skill_path,
    register_bundled_skill,
)
from clawcodex_ext.skills.invocation import (
    SkillInvocationErrorCode,
    SkillInvocationOrigin,
    SkillInvocationRequest,
    SkillInvocationService,
)
from clawcodex_ext.tool_system.context import ToolContext


@pytest.fixture(autouse=True)
def _clean_bundled_registry() -> Iterator[None]:
    clear_bundled_skills()
    yield
    clear_bundled_skills()


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def test_same_name_replaces_definition_and_context_builder_is_supported() -> None:
    name = _unique_name("replace")
    context = object()
    register_bundled_skill(
        BundledSkillDefinition(
            name=name,
            description="old definition",
            get_prompt_for_command=lambda args: f"old: {args}",
        )
    )
    register_bundled_skill(
        BundledSkillDefinition(
            name=name,
            description="new definition",
            get_prompt_for_command=lambda args, ctx: f"new: {args}; context={ctx is context}",
        )
    )

    matches = [skill for skill in get_bundled_skills() if skill.name == name]
    assert len(matches) == 1
    assert matches[0].description == "new definition"
    assert matches[0].get_prompt_for_command is not None
    assert matches[0].get_prompt_for_command("value", context) == ("new: value; context=True")


def test_public_path_helpers_enforce_process_root_and_preexisting_dir_fails_open() -> None:
    name = _unique_name("collision")
    process_root = Path(get_bundled_skills_root())
    extract_dir = Path(get_bundled_skill_extract_dir(name))
    assert is_bundled_skill_path(process_root)
    assert is_bundled_skill_path(extract_dir)
    assert not is_bundled_skill_path(process_root.parent)

    register_bundled_skill(
        BundledSkillDefinition(
            name=name,
            description="pre-created extraction directory probe",
            files={"guide.txt": "expected content"},
            get_prompt_for_command=lambda _args: "UNPREFIXED BODY",
        )
    )
    skill = get_bundled_skill_by_name(name)
    assert skill is not None
    assert skill.skill_root is not None
    extract_dir = Path(skill.skill_root)
    assert is_bundled_skill_path(extract_dir)

    extract_dir.mkdir()
    (extract_dir / "guide.txt").write_text("expected content", encoding="utf-8")
    (extract_dir / "injected.txt").write_text("untrusted extra file", encoding="utf-8")

    assert skill.get_prompt("") == "UNPREFIXED BODY"


def test_concurrent_first_invocations_share_one_extraction() -> None:
    name = _unique_name("concurrent")
    register_bundled_skill(
        BundledSkillDefinition(
            name=name,
            description="concurrent extraction probe",
            files={"references/guide.md": "thread-safe content"},
            get_prompt_for_command=lambda args: f"BODY {args}",
        )
    )
    skill = get_bundled_skill_by_name(name)
    assert skill is not None
    assert skill.skill_root is not None

    with ThreadPoolExecutor(max_workers=16) as pool:
        prompts = list(pool.map(skill.get_prompt, (str(index) for index in range(64))))

    prefix = f"Base directory for this skill: {skill.skill_root}\n\n"
    assert all(prompt.startswith(prefix) for prompt in prompts)
    assert (Path(skill.skill_root) / "references" / "guide.md").read_text(encoding="utf-8") == ("thread-safe content")


def test_resource_skill_same_name_reregistration_gets_fresh_private_root() -> None:
    name = _unique_name("resource-replace")
    register_bundled_skill(
        BundledSkillDefinition(
            name=name,
            description="first resources",
            files={"guide.txt": "first"},
            get_prompt_for_command=lambda _args: "FIRST",
        )
    )
    first = get_bundled_skill_by_name(name)
    assert first is not None and first.skill_root is not None
    assert "FIRST" in first.get_prompt("")
    first_root = Path(first.skill_root)
    assert (first_root / "guide.txt").read_text(encoding="utf-8") == "first"

    register_bundled_skill(
        BundledSkillDefinition(
            name=name,
            description="replacement resources",
            files={"guide.txt": "second"},
            get_prompt_for_command=lambda _args: "SECOND",
        )
    )
    second = get_bundled_skill_by_name(name)
    assert second is not None and second.skill_root is not None
    second_root = Path(second.skill_root)

    assert second_root != first_root
    assert "SECOND" in second.get_prompt("")
    assert (second_root / "guide.txt").read_text(encoding="utf-8") == "second"
    assert (first_root / "guide.txt").read_text(encoding="utf-8") == "first"


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "../escape.txt",
        "nested/../../escape.txt",
        "/absolute.txt",
        r"C:\absolute.txt",
        r"\\server\share\escape.txt",
    ),
)
def test_unsafe_resource_paths_fail_open_without_exposing_a_root(
    unsafe_path: str,
) -> None:
    name = _unique_name("unsafe-path")
    assert register_bundled_skill(
        BundledSkillDefinition(
            name=name,
            description="unsafe resource path probe",
            files={unsafe_path: "must never be written"},
            get_prompt_for_command=lambda _args: "SAFE BODY",
        )
    )
    skill = get_bundled_skill_by_name(name)
    assert skill is not None and skill.skill_root is not None

    assert skill.get_prompt("") == "SAFE BODY"
    assert not Path(skill.skill_root).exists()
    builder = skill.get_prompt_for_command
    assert builder is not None
    assert getattr(builder, "_bundled_resource_root") is None
    assert "continuing without a base directory" in getattr(
        builder,
        "_bundled_resource_diagnostic",
    )

    records: list[tuple[object, ...]] = []
    result = SkillInvocationService(
        resolver=lambda _name, _context: skill,
        recorder=lambda *record: records.append(record),
    ).invoke(
        SkillInvocationRequest(name, origin=SkillInvocationOrigin.USER),
        ToolContext(workspace_root=Path.cwd()),
    )
    assert result.success is True
    assert records[0][1] == f"bundled:{name}"
    assert any("continuing without a base directory" in item for item in result.diagnostics)


def test_required_resource_extraction_failure_blocks_invocation() -> None:
    name = _unique_name("required-resource")
    assert register_bundled_skill(
        BundledSkillDefinition(
            name=name,
            description="required resource failure probe",
            files={"../escape.txt": "must never be written"},
            requires_resources=True,
            get_prompt_for_command=lambda _args: "MUST NOT BE RETURNED",
        )
    )
    skill = get_bundled_skill_by_name(name)
    assert skill is not None

    result = SkillInvocationService(
        resolver=lambda _name, _context: skill,
        recorder=lambda *_args: None,
    ).invoke(
        SkillInvocationRequest(name, origin=SkillInvocationOrigin.USER),
        ToolContext(workspace_root=Path.cwd()),
    )

    assert result.success is False
    assert result.prompt is None
    assert result.error is not None
    assert result.error.code is SkillInvocationErrorCode.PROMPT_BUILD_FAILED
    assert f"required bundled resources for skill '{name}'" in result.error.message.lower()


def test_symlinked_extraction_root_is_rejected(tmp_path: Path) -> None:
    name = _unique_name("symlink-root")
    assert register_bundled_skill(
        BundledSkillDefinition(
            name=name,
            description="symlink extraction root probe",
            files={"guide.txt": "trusted"},
            get_prompt_for_command=lambda _args: "SAFE BODY",
        )
    )
    skill = get_bundled_skill_by_name(name)
    assert skill is not None and skill.skill_root is not None
    root = Path(skill.skill_root)
    target = tmp_path / "outside"
    target.mkdir()
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    assert skill.get_prompt("") == "SAFE BODY"
    assert not (target / "guide.txt").exists()


def test_extracted_files_use_private_posix_modes() -> None:
    if os.name == "nt":
        pytest.skip("POSIX mode bits are not meaningful on Windows")

    name = _unique_name("private-modes")
    assert register_bundled_skill(
        BundledSkillDefinition(
            name=name,
            description="private mode probe",
            files={"references/guide.txt": "trusted"},
            get_prompt_for_command=lambda _args: "BODY",
        )
    )
    skill = get_bundled_skill_by_name(name)
    assert skill is not None and skill.skill_root is not None
    skill.get_prompt("")

    root = Path(skill.skill_root)
    guide = root / "references" / "guide.txt"
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(guide.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(guide.stat().st_mode) == 0o600
