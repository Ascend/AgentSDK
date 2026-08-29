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

"""Bundled reference files are readable without becoming writable."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pytest

from clawcodex_ext.permissions.check import has_permissions_to_use_tool
from clawcodex_ext.permissions.types import ToolPermissionContext
from clawcodex_ext.skills.bundled_skills import (
    BundledSkillDefinition,
    clear_bundled_skills,
    get_bundled_skill_by_name,
    register_bundled_skill,
)
from clawcodex_ext.skills.invocation import (
    SkillInvocationOrigin,
    SkillInvocationRequest,
    SkillInvocationService,
    apply_skill_context_modifier,
)
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.errors import ToolPermissionError
from clawcodex_ext.tool_system.tools import GlobTool, GrepTool, ReadTool, WriteTool


@pytest.fixture(autouse=True)
def _clean_bundled_registry() -> Iterator[None]:
    clear_bundled_skills()
    yield
    clear_bundled_skills()


def test_bundled_resources_are_readable_searchable_but_not_writable(
    tmp_path: Path,
) -> None:
    name = f"resource-permissions-{uuid4().hex}"
    marker = f"bundled-marker-{uuid4().hex}"
    register_bundled_skill(
        BundledSkillDefinition(
            name=name,
            description="bundled resource permission probe",
            files={"references/guide.md": f"# Guide\n\n{marker}\n"},
            get_prompt_for_command=lambda _args: "RESOURCE BODY",
        )
    )
    skill = get_bundled_skill_by_name(name)
    assert skill is not None
    assert skill.skill_root is not None
    context = ToolContext(
        workspace_root=tmp_path,
        cwd=tmp_path,
        permission_context=ToolPermissionContext(mode="default"),
    )
    service = SkillInvocationService(
        resolver=lambda _name, _context: skill,
        recorder=lambda *_args: None,
    )
    result = service.invoke(
        SkillInvocationRequest(name, origin=SkillInvocationOrigin.USER),
        context,
    )
    assert result.success is True
    assert result.context_modifier is not None
    assert "Base directory for this skill:" in (result.prompt or "")
    apply_skill_context_modifier(context, result.context_modifier)

    root = Path(skill.skill_root)
    guide = root / "references" / "guide.md"
    assert context.skill_resource_roots == (str(root),)

    # Stale roots are ignored independently; they must not mask the active root.
    missing_root = root.parent / "missing-resource-root"
    context.skill_resource_roots = (str(missing_root), str(root))

    permission_inputs = (
        (ReadTool, {"file_path": str(guide)}),
        (GrepTool, {"pattern": marker, "path": str(root)}),
        (GlobTool, {"pattern": "**/*.md", "path": str(root)}),
    )
    for tool, tool_input in permission_inputs:
        decision = has_permissions_to_use_tool(
            tool,
            tool_input,
            context.permission_context,
            tool_use_context=context,
        )
        assert decision.behavior == "allow"

    other_name = f"other-resource-{uuid4().hex}"
    register_bundled_skill(
        BundledSkillDefinition(
            name=other_name,
            description="inactive resource permission probe",
            files={"secret.txt": "not active"},
            get_prompt_for_command=lambda _args: "OTHER BODY",
        )
    )
    other_skill = get_bundled_skill_by_name(other_name)
    assert other_skill is not None
    assert other_skill.skill_root is not None
    other_skill.get_prompt("")
    other_file = Path(other_skill.skill_root) / "secret.txt"
    other_decision = has_permissions_to_use_tool(
        ReadTool,
        {"file_path": str(other_file)},
        context.permission_context,
        tool_use_context=context,
    )
    assert other_decision.behavior == "ask"

    outside_path = tmp_path.parent / f"outside-{uuid4().hex}.txt"
    outside_path.write_text("not harness-owned", encoding="utf-8")
    outside_decision = has_permissions_to_use_tool(
        ReadTool,
        {"file_path": str(outside_path)},
        context.permission_context,
        tool_use_context=context,
    )
    assert outside_decision.behavior == "ask"

    read_output = ReadTool.call({"file_path": str(guide)}, context).output
    assert marker in read_output["file"]["content"]

    grep_output = GrepTool.call(
        {
            "pattern": marker,
            "path": str(root),
            "output_mode": "content",
        },
        context,
    ).output
    assert marker in str(grep_output)

    glob_output = GlobTool.call(
        {"pattern": "**/*.md", "path": str(root)},
        context,
    ).output
    assert any(str(path).endswith("guide.md") for path in glob_output["filenames"])

    with pytest.raises(ToolPermissionError):
        WriteTool.call(
            {"file_path": str(guide), "content": "tampered"},
            context,
        )
    assert guide.read_text(encoding="utf-8").endswith(f"{marker}\n")
