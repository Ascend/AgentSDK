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

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.skills.create import create_skill
from src.skills.frontmatter import parse_frontmatter
from src.skills.loader import clear_skill_registry, get_all_skills
from src.tool_system.context import ToolContext
from src.tool_system.tools import SkillTool

from ._assertions import rendered_prompt


class SkillSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self) -> None:
        clear_skill_registry()
        self.tmp.cleanup()


class TestSkillCreate(SkillSystemTests):
    def test_create_writes_skill_md(self) -> None:
        skills_dir = self.root / "skills"
        p = create_skill(
            directory=skills_dir,
            name="demo",
            description="demo skill",
            when_to_use="use it when testing",
            allowed_tools=["Read", "Grep"],
            arguments=["foo", "bar"],
            body="Hello $foo $1",
        )
        self.assertTrue(p.exists())
        parsed = parse_frontmatter(p.read_text(encoding="utf-8"))
        self.assertEqual(parsed.frontmatter["description"], "demo skill")
        self.assertEqual(parsed.frontmatter["when_to_use"], "use it when testing")

    def test_parse_frontmatter_supports_inline_lists(self) -> None:
        parsed = parse_frontmatter("---\narguments: [name]\nallowed-tools: [Read, Grep]\n---\nHello $name\n")
        self.assertEqual(parsed.frontmatter["arguments"], ["name"])
        self.assertEqual(parsed.frontmatter["allowed-tools"], ["Read", "Grep"])


class TestSkillRegister(SkillSystemTests):
    def test_register_loads_skill_from_dir(self) -> None:
        skills_dir = self.root / "skills"
        create_skill(
            directory=skills_dir,
            name="hello",
            description="say hello",
            body="Hello",
        )
        with patch.dict(os.environ, {"CLAWCODEX_SKILLS_DIR": str(skills_dir)}):
            skills = get_all_skills(project_root=self.root)
            by_name = {s.name: s for s in skills}
            self.assertIn("hello", by_name)
            self.assertEqual(by_name["hello"].description, "say hello")
            self.assertEqual(by_name["hello"].loaded_from, "user")


class TestSkillUse(SkillSystemTests):
    def test_use_substitutes_arguments(self) -> None:
        skills_dir = self.root / "skills"
        create_skill(
            directory=skills_dir,
            name="hello",
            description="say hello",
            arguments=["name"],
            body="Hello $name ($0) / $ARGUMENTS",
        )
        ctx = ToolContext(workspace_root=self.root)
        with patch.dict(os.environ, {"CLAWCODEX_SKILLS_DIR": str(skills_dir)}):
            result = SkillTool.call({"skill": "hello", "args": 'bob "the builder"'}, ctx)
            out = result.output
            prompt = rendered_prompt(result)
            self.assertTrue(out["success"])
            self.assertIn("Hello bob (bob)", prompt)
            self.assertIn('bob "the builder"', prompt)
