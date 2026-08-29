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

"""Tests for overview-agent macro intent routing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from extensions.sop_converter.agent_md_writer import AgentComponentInfo
from extensions.sop_converter.runtime.macros.models import MacroDefinition, MacroRoute
from extensions.sop_converter.runtime.macros.overview_intent import (
    assign_macros_to_owner_skills,
    format_overview_macro_intent_block,
    pick_owner_skill,
    resolve_macro_delegate_agent,
)
from extensions.sop_converter.skill_grouper import SkillSpec
from extensions.sop_converter.sop_prompts import (
    SOP_OVERVIEW_ROUTING,
    append_sop_overview_routing,
)


class TestResolveMacroDelegateAgent(unittest.TestCase):
    def test_prefers_agent_matching_covered_tool_name(self) -> None:
        agent = resolve_macro_delegate_agent(
            target_tool="text-processing-pipeline",
            covered_tools=["skills-skill-handlers-execute-pipeline"],
            agent_names=[
                "skills-agent",
                "skill_handlers-agent",
                "operations-agent",
            ],
            agent_tools={
                "skills-agent": {"text-processing-pipeline"},
                "skill_handlers-agent": {
                    "text-processing-pipeline",
                    "skills-skill-handlers-execute-pipeline",
                },
                "operations-agent": {"text-processing-pipeline"},
            },
        )
        self.assertEqual(agent, "skill_handlers-agent")

    def test_falls_back_to_macro_owner_when_no_name_match(self) -> None:
        agent = resolve_macro_delegate_agent(
            target_tool="text-processing-pipeline",
            covered_tools=["unrelated-atomic-tool"],
            agent_names=["ops-agent", "core-agent"],
            agent_tools={
                "ops-agent": {"text-processing-pipeline"},
                "core-agent": set(),
            },
        )
        self.assertEqual(agent, "ops-agent")


class TestAssignMacrosToOwnerSkills(unittest.TestCase):
    def test_macro_lands_on_single_owner_only(self) -> None:
        skills = [
            SkillSpec(
                name="skills",
                description="skills root",
                allowed_tools=["skills-other-tool"],
            ),
            SkillSpec(
                name="skill_handlers",
                description="handlers",
                allowed_tools=["skills-skill-handlers-execute-pipeline"],
            ),
            SkillSpec(
                name="operations",
                description="ops",
                allowed_tools=["operations-foo"],
            ),
        ]
        macros = [
            MacroDefinition(
                name="text-processing-pipeline",
                description="text cleanup",
                routing=MacroRoute(
                    phrases=["用手写宏处理文本数据"],
                    target_tool="text-processing-pipeline",
                    covered_tools=["skills-skill-handlers-execute-pipeline"],
                ),
            ),
            MacroDefinition(
                name="image-processing-pipeline",
                description="image cleanup",
                routing=MacroRoute(
                    phrases=["用手写宏处理图像数据"],
                    target_tool="image-processing-pipeline",
                    covered_tools=["skills-skill-handlers-execute-pipeline"],
                ),
            ),
        ]
        ownership = assign_macros_to_owner_skills(skills, macros)
        self.assertEqual(
            ownership,
            {
                "text-processing-pipeline": "skill_handlers",
                "image-processing-pipeline": "skill_handlers",
            },
        )
        owners = [s for s in skills if "text-processing-pipeline" in s.allowed_tools]
        self.assertEqual([s.name for s in owners], ["skill_handlers"])
        self.assertIn("image-processing-pipeline", skills[1].allowed_tools)
        self.assertNotIn("text-processing-pipeline", skills[0].allowed_tools)
        self.assertNotIn("text-processing-pipeline", skills[2].allowed_tools)

    def test_pick_owner_skill_by_covered_tools(self) -> None:
        skills = [
            SkillSpec(name="core", description="", allowed_tools=["core-a"]),
            SkillSpec(
                name="skill_handlers",
                description="",
                allowed_tools=["skills-skill-handlers-execute-pipeline"],
            ),
        ]
        owner = pick_owner_skill(
            skills,
            target_tool="text-processing-pipeline",
            covered_tools=["skills-skill-handlers-execute-pipeline"],
        )
        self.assertIsNotNone(owner)
        assert owner is not None
        self.assertEqual(owner.name, "skill_handlers")


class TestOverviewMacroIntentBlock(unittest.TestCase):
    def _write_macro(self, bundle: Path, name: str, phrases: list[str], covered: list[str]) -> None:
        macros = bundle / ".clawcodex" / "macros"
        macros.mkdir(parents=True, exist_ok=True)
        phrases_yaml = "\n".join(f"    - {p}" for p in phrases)
        covered_yaml = "\n".join(f"    - {c}" for c in covered)
        (macros / f"{name}.yaml").write_text(
            f"""version: 1
name: {name}
description: Demo macro for {name}
enabled: true
routing:
  phrases:
{phrases_yaml}
  target_tool: {name}
  priority: 100
  covered_tools:
{covered_yaml}
""",
            encoding="utf-8",
        )

    def test_format_block_routes_text_macro_to_skill_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            self._write_macro(
                bundle,
                "text-processing-pipeline",
                ["用手写宏处理文本数据", "处理文本数据", "手写宏"],
                ["skills-skill-handlers-execute-pipeline"],
            )
            agents = [
                AgentComponentInfo(
                    name="skills-agent",
                    description="skills",
                    capabilities=["skills-other-tool"],
                ),
                AgentComponentInfo(
                    name="skill_handlers-agent",
                    description="handlers",
                    capabilities=["skills-skill-handlers-execute-pipeline"],
                ),
            ]
            block = format_overview_macro_intent_block(bundle, component_agents=agents)
            self.assertIn("宏工具意图", block)
            self.assertIn("用手写宏处理文本数据", block)
            self.assertIn("text-processing-pipeline", block)
            self.assertIn("@skill_handlers-agent", block)
            self.assertIn("禁止", block)
            self.assertIn("Explore", block)

    def test_append_overview_includes_macro_block_before_sdk_overview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            self._write_macro(
                bundle,
                "text-processing-pipeline",
                ["用手写宏处理文本数据"],
                ["skills-skill-handlers-execute-pipeline"],
            )
            (bundle / "SDK_OVERVIEW.md").write_text("# SDK\n", encoding="utf-8")
            agents = [
                AgentComponentInfo(
                    name="skill_handlers-agent",
                    description="handlers",
                    capabilities=["skills-skill-handlers-execute-pipeline"],
                ),
            ]
            body = append_sop_overview_routing(
                "",
                bundle_path=bundle,
                component_agents=agents,
            )
            self.assertIn("用手写宏处理文本数据", body)
            self.assertIn("@skill_handlers-agent", body)
            macro_at = body.index("宏工具意图")
            sdk_at = body.index("SDK_OVERVIEW.md")
            self.assertLess(macro_at, sdk_at)

    def test_sop_overview_routing_mentions_macro_intent(self) -> None:
        self.assertIn("宏工具意图", SOP_OVERVIEW_ROUTING)
        self.assertIn("手写宏", SOP_OVERVIEW_ROUTING)


if __name__ == "__main__":
    unittest.main()
