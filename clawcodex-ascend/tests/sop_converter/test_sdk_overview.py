# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Tests for SDK overview generation during sop convert."""

from __future__ import annotations

import unittest
from pathlib import Path

from extensions.sop_converter.runtime.sdk_overview import (
    format_sdk_overview_block,
    generate_sdk_overview_markdown,
    write_sdk_overview,
)
from extensions.sop_converter.runtime.skill_grouper import GroupStrategy, SkillSpec
from extensions.sop_converter.core.source_parser import ParamSpec, SourceComponent, SourceOperation


class TestSdkOverview(unittest.TestCase):
    def test_generate_markdown_lists_skills_and_modules(self) -> None:
        comp = SourceComponent(
            name="openjiuwen.agent_teams",
            file_path="openjiuwen/agent_teams/core.py",
            description="Team agent orchestration",
            operations=[
                SourceOperation(
                    name="init_team_chat",
                    description="Initialize team conversation session.",
                    file_stem="cli",
                    has_docstring=True,
                )
            ],
        )
        skills = [
            SkillSpec(
                name="agent-teams",
                description="Jiuwen team agents",
                allowed_tools=["openjiuwen-agent-teams-init-team-chat"],
            )
        ]
        md = generate_sdk_overview_markdown([comp], skills=skills, sdk_source_dir="/tmp/sdk")
        self.assertIn("SDK 模块总览", md)
        self.assertIn("agent-teams-agent", md)
        self.assertIn("init_team_chat", md)

    def test_write_and_prompt_block(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            path = write_sdk_overview(
                bundle,
                [
                    SourceComponent(
                        name="pkg.mod",
                        file_path="pkg/mod.py",
                        description="Demo",
                        operations=[],
                    )
                ],
            )
            self.assertTrue(path.is_file())
            block = format_sdk_overview_block(bundle)
            self.assertIn("SDK_OVERVIEW.md", block)
            self.assertIn("SDK 模块总览", block)
            self.assertIn("Read", block)
            self.assertNotIn("域 Agent 速查", block)
            inlined = format_sdk_overview_block(bundle, inline_content=True)
            self.assertIn("域 Agent 速查", inlined)


class TestSkillForComponentMatching(unittest.TestCase):
    """Verify ``_skill_for_component`` adapts to existing naming conventions.

    Skill names follow ``<domain>_merged-skill`` or ``<domain>-skill`` patterns
    (see skill_grouper / bundle_skills). Component names are full module paths
    like ``JiuwenAgent/openjiuwen/harness/task_loop``. The matcher must bridge
    these two naming schemes without changing either side.
    """

    def _skill(self, name: str) -> SkillSpec:
        return SkillSpec(name=name, description="", allowed_tools=[])

    def test_merged_skill_matches_path_segment(self) -> None:
        """``harness_merged-skill`` should match ``.../harness/task_loop``."""
        from extensions.sop_converter.runtime.sdk_overview import _skill_for_component

        skills = [self._skill("harness_merged-skill")]
        result = _skill_for_component("JiuwenAgent/openjiuwen/harness/task_loop", skills)
        self.assertEqual(result, "harness_merged-skill-agent")

    def test_merged_skill_matches_short_path(self) -> None:
        """``core_merged-skill`` should match ``.../core/agents``."""
        from extensions.sop_converter.runtime.sdk_overview import _skill_for_component

        skills = [self._skill("core_merged-skill")]
        result = _skill_for_component("JiuwenAgent/openjiuwen/core/agents", skills)
        self.assertEqual(result, "core_merged-skill-agent")

    def test_plain_skill_matches_path_segment(self) -> None:
        """``foundation-skill`` should match ``.../foundation``."""
        from extensions.sop_converter.runtime.sdk_overview import _skill_for_component

        skills = [self._skill("foundation-skill")]
        result = _skill_for_component("JiuwenAgent/openjiuwen/core/foundation", skills)
        self.assertEqual(result, "foundation-skill-agent")

    def test_multi_word_skill_matches_compound_segment(self) -> None:
        """``agent_evolving_merged-skill`` matches ``.../agent_evolving/...``."""
        from extensions.sop_converter.runtime.sdk_overview import _skill_for_component

        skills = [self._skill("agent_evolving_merged-skill")]
        result = _skill_for_component(
            "JiuwenAgent/openjiuwen/agent_evolving/rl_offline_coordinator",
            skills,
        )
        self.assertEqual(result, "agent_evolving_merged-skill-agent")

    def test_hyphenated_skill_matches_underscore_component(self) -> None:
        """``agent-teams`` should match dotted ``openjiuwen.agent_teams``.

        Regression for review feedback: hyphenated skill base names must
        match underscore/dotted component names (``-`` vs ``_`` notation).
        """
        from extensions.sop_converter.runtime.sdk_overview import _skill_for_component

        skills = [self._skill("agent-teams")]
        result = _skill_for_component("openjiuwen.agent_teams", skills)
        self.assertEqual(result, "agent-teams-agent")

    def test_hyphenated_skill_matches_dotted_module(self) -> None:
        """``task_loop-skill`` should match dotted ``openjiuwen.harness.task_loop``."""
        from extensions.sop_converter.runtime.sdk_overview import _skill_for_component

        skills = [self._skill("task_loop-skill")]
        result = _skill_for_component("openjiuwen.harness.task_loop", skills)
        self.assertEqual(result, "task_loop-skill-agent")

    def test_hyphenated_skill_does_not_match_unrelated_component(self) -> None:
        """``agent-teams`` must not match unrelated dotted modules."""
        from extensions.sop_converter.runtime.sdk_overview import _skill_for_component

        skills = [self._skill("agent-teams")]
        result = _skill_for_component("openjiuwen.other.module", skills)
        self.assertIsNone(result)

    def test_no_match_returns_none(self) -> None:
        """Unrelated component returns None (rendered as ``—`` in overview)."""
        from extensions.sop_converter.runtime.sdk_overview import _skill_for_component

        skills = [self._skill("harness_merged-skill")]
        result = _skill_for_component("JiuwenAgent/openjiuwen/agent_teams/external", skills)
        self.assertIsNone(result)

    def test_first_match_wins_when_multiple_skills_match(self) -> None:
        """When two skills could match, the first in the list wins."""
        from extensions.sop_converter.runtime.sdk_overview import _skill_for_component

        skills = [
            self._skill("harness_merged-skill"),
            self._skill("task_loop-skill"),
        ]
        # Both could match ``.../harness/task_loop``; first one wins.
        result = _skill_for_component("JiuwenAgent/openjiuwen/harness/task_loop", skills)
        self.assertEqual(result, "harness_merged-skill-agent")

    def test_overview_markdown_shows_agent_for_matched_component(self) -> None:
        """Integration: generated markdown should show ``@<skill>-agent``."""
        comp = SourceComponent(
            name="openjiuwen.harness.task_loop",
            file_path="openjiuwen/harness/task_loop/loop_coordinator.py",
            description="Loop coordinator state machine",
            operations=[],
        )
        skills = [
            SkillSpec(
                name="harness_merged-skill",
                description="Harness task loop",
                allowed_tools=["openjiuwen-harness-task-loop-reset"],
            )
        ]
        md = generate_sdk_overview_markdown([comp], skills=skills)
        self.assertIn("harness_merged-skill-agent", md)
        self.assertNotIn("路由 Agent**: —", md)


class TestIoSdkOverview(unittest.TestCase):
    def _components(self) -> list[SourceComponent]:
        harness_ops = [
            SourceOperation(
                name="run_loop",
                description="Run the task loop coordinator.",
                parameters=[ParamSpec(name="session", type_hint="SessionHandle")],
                file_stem="loop_coordinator",
                has_docstring=True,
            ),
        ]
        core_ops = [
            SourceOperation(
                name="create_session",
                description="Create a new session handle.",
                parameters=[ParamSpec(name="config", type_hint="dict")],
                class_name="SessionFactory",
                has_docstring=True,
            ),
        ]
        return [
            SourceComponent(
                name="openjiuwen.harness.task_loop",
                file_path="openjiuwen/harness/task_loop/loop_coordinator.py",
                description="Harness task loop",
                operations=harness_ops,
            ),
            SourceComponent(
                name="openjiuwen.core.session",
                file_path="openjiuwen/core/session/factory.py",
                description="Session factory",
                operations=core_ops,
            ),
        ]

    def test_io_overview_uses_tool_routing_not_module_path(self) -> None:
        skills = [
            SkillSpec(
                name="io_group_sessionhandle_dict",
                description="Operations sharing types: sessionhandle, dict",
                allowed_tools=[
                    "openjiuwen.harness.task_loop.run_loop",
                    "SessionFactory.create_session",
                ],
            ),
        ]
        md = generate_sdk_overview_markdown(
            self._components(),
            skills=skills,
            group_strategy=GroupStrategy.IO_RELATION,
        )
        self.assertIn("SDK 类型路由总览 (IO 分组)", md)
        self.assertIn("工具 → Agent", md)
        self.assertIn("入口 API → Agent", md)
        self.assertIn("禁止", md)
        self.assertNotIn("## 模块 → 能力", md)
        self.assertIn("openjiuwen.harness.task_loop.run_loop", md)
        self.assertIn("@io_group_sessionhandle_dict-agent", md)

    def test_io_overview_prompt_block_header(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            write_sdk_overview(
                bundle,
                self._components(),
                skills=[
                    SkillSpec(
                        name="io_group_str",
                        description="Operations sharing types: str",
                        allowed_tools=["openjiuwen.harness.task_loop.run_loop"],
                    )
                ],
                group_strategy=GroupStrategy.IO_RELATION,
            )
            block = format_sdk_overview_block(bundle)
            self.assertIn("IO 分组", block)
            self.assertIn("禁止", block)
            self.assertIn("工具 → Agent", block)
            self.assertIn("Read", block)
            # Default is pointer-only; full tables stay in SDK_OVERVIEW.md
            self.assertNotIn("## 工具 → Agent", block)

    def test_path_strategy_still_uses_module_section(self) -> None:
        comp = SourceComponent(
            name="openjiuwen.harness.task_loop",
            file_path="openjiuwen/harness/task_loop/loop_coordinator.py",
            description="Harness task loop",
            operations=[],
        )
        skills = [
            SkillSpec(
                name="harness_merged-skill",
                description="Harness APIs",
                allowed_tools=[],
            )
        ]
        md = generate_sdk_overview_markdown(
            [comp],
            skills=skills,
            group_strategy=GroupStrategy.COMPONENT_GROUP,
        )
        self.assertIn("SDK 模块总览", md)
        self.assertIn("## 模块 → 能力", md)
        self.assertNotIn("SDK 类型路由总览", md)
