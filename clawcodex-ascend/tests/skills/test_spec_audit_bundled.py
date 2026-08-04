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

"""Public runtime contract for the bundled ``spec-audit`` skill."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
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


def test_spec_audit_is_bundled_with_its_complete_portable_resource_tree(
    tmp_path: Path,
) -> None:
    skill = get_bundled_skill_by_name("spec-audit")

    assert skill is not None
    assert skill.source == "bundled"
    assert skill.loaded_from == "bundled"
    assert skill.user_invocable is True

    context = ToolContext(workspace_root=tmp_path)
    result = SkillInvocationService(
        resolver=lambda _name, _context: skill,
        recorder=lambda *_args: None,
    ).invoke(
        SkillInvocationRequest(
            "spec-audit",
            args="--spec /tmp/requirements.md",
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
    assert "# Spec Audit" in result.prompt
    assert "ARGUMENTS: --spec /tmp/requirements.md" in result.prompt
    modified_context = result.context_modifier(context)
    assert modified_context.skill_resource_roots == (str(root),)

    required_files = (
        "SKILL.md",
        "agents/openai.yaml",
        "assets/templates/dossier.md",
        "assets/templates/report.md",
        "references/audit-protocol.md",
        "references/finding-protocol.md",
        "references/host-adaptation.md",
        "references/report-contract.md",
        "scripts/inventory.py",
        "scripts/lint_report.py",
        "scripts/prepare_audit.py",
    )
    for relative_path in required_files:
        assert (root / relative_path).is_file(), relative_path

    for helper_name in ("inventory.py", "lint_report.py", "prepare_audit.py"):
        completed = subprocess.run(
            [sys.executable, "-I", "-S", str(root / "scripts" / helper_name), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
