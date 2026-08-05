#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.templates import (
    Template,
    TemplateCompatibilityError,
    TemplateRenderError,
    TemplateRenderer,
    TemplateUnsafePathError,
)


def _template() -> Template:
    return Template(
        id="skill-demo",
        title="Skill Demo",
        fields={
            "content_template": "# {{ skill_name }}\n\n{{ description }}\n{{ token }}",
        },
        metadata={
            "kind": "skill",
            "output_path_template": ".claude/skills/{{ skill_name }}/SKILL.md",
            "variables": [
                {
                    "name": "skill_name",
                    "description": "Skill name",
                    "pattern": r"^[A-Za-z0-9_-]+$",
                },
                {"name": "description", "description": "Description"},
                {"name": "token", "description": "Secret token", "secret": True},
            ],
        },
    )


def test_renderer_substitutes_simple_placeholders(tmp_path: Path) -> None:
    rendered = TemplateRenderer().render(
        _template(),
        {"skill_name": "browser", "description": "Automate browsers", "token": "s3cr3t"},
        workspace_root=tmp_path,
    )
    assert rendered.content == "# browser\n\nAutomate browsers\ns3cr3t"
    assert rendered.output_path == tmp_path / ".claude" / "skills" / "browser" / "SKILL.md"
    assert rendered.variables_used == {
        "skill_name": "browser",
        "description": "Automate browsers",
    }


def test_renderer_missing_required_variable_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(TemplateRenderError, match="description"):
        TemplateRenderer().render(
            _template(),
            {"skill_name": "browser", "token": "s3cr3t"},
            workspace_root=tmp_path,
        )


def test_renderer_rejects_pattern_mismatch(tmp_path: Path) -> None:
    with pytest.raises(TemplateRenderError, match="pattern"):
        TemplateRenderer().render(
            _template(),
            {"skill_name": "../bad", "description": "x", "token": "s3cr3t"},
            workspace_root=tmp_path,
        )


def test_renderer_does_not_execute_jinja_expressions(tmp_path: Path) -> None:
    template = Template(
        id="prompt",
        title="Prompt",
        fields={"content_template": "{{ name }} {{ issue.title }} {% if x %}"},
        metadata={"kind": "prompt"},
    )
    rendered = TemplateRenderer().render(template, {"name": "hello"}, workspace_root=tmp_path)
    assert rendered.content == "hello {{ issue.title }} {% if x %}"


def test_renderer_rejects_path_traversal(tmp_path: Path) -> None:
    template = Template(
        id="bad-path",
        title="Bad Path",
        fields={"content_template": "x"},
        metadata={"output_path_template": "../outside.txt"},
    )
    with pytest.raises(TemplateUnsafePathError):
        TemplateRenderer().render(template, {}, workspace_root=tmp_path)


def test_renderer_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    template = Template(
        id="future",
        title="Future",
        fields={"content_template": "x"},
        metadata={"schema_version": "99"},
    )
    with pytest.raises(TemplateCompatibilityError):
        TemplateRenderer().render(template, {}, workspace_root=tmp_path)
