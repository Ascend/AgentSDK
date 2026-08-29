#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Tests for SOP bundle skill registration and prompts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from extensions.sop_converter.bundle_skills import register_bundle_skills
from extensions.sop_converter.sop_prompts import agent_type_to_skill_name


class TestSopPrompts(unittest.TestCase):
    def test_agent_type_to_skill_name(self) -> None:
        self.assertEqual(
            agent_type_to_skill_name("harness_merged-agent"),
            "harness_merged-skill",
        )


class TestBundleSkills(unittest.TestCase):
    def test_loads_flat_skill_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            bundle = ws / "JiuwenAgent_tool_test"
            skill_dir = ws / "skills" / "JiuwenAgent_tool_test"
            skill_dir.mkdir(parents=True)
            bundle.mkdir(parents=True)

            (skill_dir / "harness_merged-skill.md").write_text(
                "---\n"
                "name: harness_merged-skill\n"
                "description: harness domain\n"
                "user-invocable: true\n"
                "---\n\n"
                "# harness\n",
                encoding="utf-8",
            )

            names = register_bundle_skills(bundle, ws)
            self.assertIn("harness_merged-skill", names.skill_names)

            from src.skills.loader import get_all_skills, get_registered_skill

            get_all_skills(project_root=ws)
            skill = get_registered_skill("harness_merged-skill")
            self.assertIsNotNone(skill)
            assert skill is not None
            self.assertEqual(skill.name, "harness_merged-skill")

    def test_flat_skill_allowed_tools_line_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            bundle = ws / "JiuwenAgent_tool_test"
            skill_dir = ws / "skills" / "JiuwenAgent_tool_test"
            skill_dir.mkdir(parents=True)
            bundle.mkdir(parents=True)

            (skill_dir / "memory-skill.md").write_text(
                "---\n"
                "name: memory-skill\n"
                "description: memory domain\n"
                "allowed-tools:\n"
                "  - openjiuwen-agent-teams-team-memory-dir\n"
                "  - openjiuwen-agent-teams-memory-sharedmemorymanager-ensure-dir\n"
                "---\n\n"
                "# memory\n",
                encoding="utf-8",
            )

            load = register_bundle_skills(bundle, ws)
            self.assertIn(
                "openjiuwen-agent-teams-team-memory-dir",
                load.tool_names,
            )
            self.assertIn(
                "openjiuwen-agent-teams-memory-sharedmemorymanager-ensure-dir",
                load.tool_names,
            )


if __name__ == "__main__":
    unittest.main()
