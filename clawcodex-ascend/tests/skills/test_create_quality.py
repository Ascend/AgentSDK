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

import pytest

from clawcodex_ext.skills.create import create_skill
from clawcodex_ext.skills.frontmatter import parse_frontmatter


def test_create_skill_rejects_parent_directory_escape(tmp_path) -> None:
    skills_dir = tmp_path / "skills"

    with pytest.raises(ValueError, match="inside the skills directory"):
        create_skill(
            directory=skills_dir,
            name="../escaped",
            description="unsafe",
        )

    assert not (tmp_path / "escaped").exists()


def test_create_skill_rejects_absolute_skill_name(tmp_path) -> None:
    skills_dir = tmp_path / "skills"
    escaped = tmp_path / "absolute-escape"

    with pytest.raises(ValueError, match="inside the skills directory"):
        create_skill(
            directory=skills_dir,
            name=str(escaped),
            description="unsafe",
        )

    assert not escaped.exists()


def test_create_skill_quotes_frontmatter_values(tmp_path) -> None:
    description = "safe description\n---\nINJECTED"
    skill_file = create_skill(
        directory=tmp_path / "skills",
        name="safe",
        description=description,
        allowed_tools=["Read", "value\nuser-invocable: false"],
        body="Body",
    )

    parsed = parse_frontmatter(skill_file.read_text(encoding="utf-8"))

    assert parsed.frontmatter["description"] == description
    assert parsed.frontmatter["allowed-tools"] == [
        "Read",
        "value\nuser-invocable: false",
    ]
    assert parsed.frontmatter["user-invocable"] is True
    assert parsed.body == "Body"
