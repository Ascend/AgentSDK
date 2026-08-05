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

import pytest

from src.services.templates import (
    Template,
    TemplateCatalogue,
    TemplateNotFoundError,
    TemplateRegistry,
)


def _registry() -> TemplateRegistry:
    registry = TemplateRegistry()
    registry.register(
        Template(
            id="python-fix",
            title="Python Fix",
            description="Fix Python bugs",
            metadata={"kind": "agent", "tags": ["python", "fix"], "category": "edit"},
            source="built-in",
        )
    )
    registry.register(
        Template(
            id="skill-browser",
            title="Browser Skill",
            description="Create browser automation skills",
            metadata={"kind": "skill", "tags": ["browser", "automation"]},
            source="project",
        )
    )
    return registry


def test_catalogue_filters_by_kind_source_and_tags() -> None:
    catalogue = TemplateCatalogue(_registry())
    assert [t.id for t in catalogue.list(kind="skill")] == ["skill-browser"]
    assert [t.id for t in catalogue.list(source="built-in")] == ["python-fix"]
    assert [t.id for t in catalogue.list(tags=["python"])] == ["python-fix"]


def test_catalogue_searches_title_description_tags() -> None:
    catalogue = TemplateCatalogue(_registry())
    assert [t.id for t in catalogue.search("browser automation")] == ["skill-browser"]


def test_catalogue_describe_returns_manifest() -> None:
    manifest = TemplateCatalogue(_registry()).describe("python-fix")
    assert manifest.kind == "agent"
    assert manifest.tags == ("python", "fix")


def test_catalogue_not_found_includes_suggestion() -> None:
    with pytest.raises(TemplateNotFoundError, match="python-fix"):
        TemplateCatalogue(_registry()).describe("python-fux")
