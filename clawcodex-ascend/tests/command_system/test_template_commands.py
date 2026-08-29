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

from pathlib import Path

from clawcodex_ext.command_system.engine import create_command_context
from clawcodex_ext.command_system.template_commands import template_command_call
from src.services.templates import reset_default_template_registry


def test_template_list_includes_built_ins(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("CLAWCODEX_MANAGED_CONFIG_DIR", str(tmp_path / "managed"))
    reset_default_template_registry()
    context = create_command_context(tmp_path)
    result = template_command_call("list --kind agent", context)
    assert "general-purpose" in result.value
    assert "agent" in result.value


def test_template_preview_renders_variables(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "cfg" / "templates"
    cfg.mkdir(parents=True)
    (cfg / "skill.yml").write_text(
        """
id: skill-demo
title: Skill Demo
kind: skill
variables:
  - name: skill_name
    description: Skill name
fields:
  output_path_template: ".claude/skills/{{ skill_name }}/SKILL.md"
  content_template: "# {{ skill_name }}"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("CLAWCODEX_MANAGED_CONFIG_DIR", str(tmp_path / "managed"))
    reset_default_template_registry()
    context = create_command_context(tmp_path)
    result = template_command_call("preview skill-demo --var skill_name=demo", context)
    assert "# demo" in result.value
    assert str(tmp_path / ".claude" / "skills" / "demo" / "SKILL.md") in result.value


def test_template_create_skill_writes_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("CLAWCODEX_MANAGED_CONFIG_DIR", str(tmp_path / "managed"))
    reset_default_template_registry()
    context = create_command_context(tmp_path)
    result = template_command_call(
        'create skill --name browser --description "Browser automation"',
        context,
    )
    target = tmp_path / ".claude" / "skills" / "browser" / "SKILL.md"
    assert "Created skill" in result.value
    assert target.read_text(encoding="utf-8") == "# browser\n\nBrowser automation\n"
