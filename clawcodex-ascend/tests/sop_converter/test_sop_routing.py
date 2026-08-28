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

"""Tests for SOP bundle routing helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from extensions.sop_converter.sop_prompts import agent_type_to_skill_name
from extensions.sop_converter.sop_routing import (
    check_bundle_agent_delegation,
    looks_like_direct_sdk_execution,
    refresh_domain_agent_sop_prompts,
    requested_agent_types_in_prompt,
)
from extensions.sop_converter.workflow_project import (
    is_prefixed_stage_agent,
    read_workflow_first_stage_skill_name,
    read_workflow_project_name,
)
from src.command_system.input_processing import strip_agent_mentions


class TestSopRouting(unittest.TestCase):
    def test_looks_like_direct_sdk_execution(self) -> None:
        self.assertTrue(looks_like_direct_sdk_execution("Skill(openjiuwen_merged-skill) then team-memory-dir"))
        self.assertFalse(looks_like_direct_sdk_execution("summarize the conversation"))

    def test_requested_agent_types_in_prompt(self) -> None:
        prompt = "@agent-AutoResearchClaw-topic-init-agent explain output contract"
        self.assertEqual(
            requested_agent_types_in_prompt(prompt),
            ["AutoResearchClaw-topic-init-agent"],
        )

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_blocks_general_purpose_when_user_named_stage_agent(self, mock_bundle) -> None:
        mock_bundle.return_value = SimpleNamespace(
            bundle_name="AutoResearchClaw3",
            bundle_path=Path("/tmp/bundle"),
        )
        agents = [
            SimpleNamespace(agent_type="AutoResearchClaw-topic-init-agent"),
            SimpleNamespace(agent_type="core_pipeline-agent"),
        ]
        err = check_bundle_agent_delegation(
            subagent_type="general-purpose",
            prompt="@agent-AutoResearchClaw-topic-init-agent list output contract",
            agent_definitions=agents,
        )
        self.assertIsNotNone(err)
        self.assertIn("AutoResearchClaw-topic-init-agent", err or "")

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_blocks_general_purpose_for_sdk_prompt(self, mock_bundle) -> None:
        mock_bundle.return_value = SimpleNamespace(bundle_name="JiuwenAgent_tool_test")
        agents = [
            SimpleNamespace(agent_type="openjiuwen_merged-agent"),
            SimpleNamespace(agent_type="memory-agent"),
        ]
        err = check_bundle_agent_delegation(
            subagent_type="general-purpose",
            prompt="Skill(openjiuwen_merged-skill) and openjiuwen-agent-teams-team-memory-dir",
            agent_definitions=agents,
        )
        self.assertIsNotNone(err)
        self.assertIn("openjiuwen_merged-agent", err or "")

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_allows_domain_agent(self, mock_bundle) -> None:
        mock_bundle.return_value = SimpleNamespace(bundle_name="JiuwenAgent_tool_test")
        agents = [SimpleNamespace(agent_type="memory-agent")]
        err = check_bundle_agent_delegation(
            subagent_type="memory-agent",
            prompt="ensure-dir with Skill(memory-skill)",
            agent_definitions=agents,
        )
        self.assertIsNone(err)

    @patch("extensions.sop_converter.bundle_context.get_active_bundle")
    def test_refresh_injects_sop_into_prefixed_stage_agents(self, mock_bundle) -> None:
        import tempfile

        from clawcodex_ext.agent.agent_definitions import AgentDefinition

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "AutoResearchClaw3"
            bundle.mkdir()
            (bundle / "workflow.yaml").write_text("name: AutoResearchClaw\n", encoding="utf-8")
            mock_bundle.return_value = SimpleNamespace(
                bundle_name="AutoResearchClaw3",
                bundle_path=bundle,
                sdk_source_dir=None,
            )

            stage_body = "# Hybrid Agent\n\n## 输出契约\n- `goal.md`"
            stage = AgentDefinition(
                agent_type="AutoResearchClaw-topic-init-agent",
                when_to_use="stage",
                source="project",
                base_dir=str(bundle),
                tools=[
                    "researchclaw-pipeline-execute-stage",
                    "autoresearchclaw-execute-stage",
                ],
                get_system_prompt=lambda **_k: stage_body,
            )
            coarse = AgentDefinition(
                agent_type="core_pipeline-agent",
                when_to_use="coarse",
                source="project",
                base_dir=str(bundle),
                get_system_prompt=lambda **_k: "coarse original",
            )

            refreshed = refresh_domain_agent_sop_prompts([stage, coarse])
            stage_prompt = refreshed[0].get_system_prompt()
            self.assertIn("SOP 工作流", stage_prompt)
            self.assertIn("默认用户指令", stage_prompt)
            self.assertIn('Skill(skill="topic-init-skill")', stage_prompt)
            self.assertIn("输出契约", stage_prompt)
            self.assertIn("Skill", refreshed[0].tools or [])
            self.assertIn("ToolSearch", refreshed[0].tools or [])
            self.assertIn(
                "researchclaw-pipeline-execute-stage",
                refreshed[0].tools or [],
            )
            self.assertIn("SOP 工作流", refreshed[1].get_system_prompt())

    def test_agent_type_to_skill_name_strips_project_prefix(self) -> None:
        self.assertEqual(
            agent_type_to_skill_name(
                "AutoResearchClaw-topic-init-agent",
                project_prefix="AutoResearchClaw",
            ),
            "topic-init-skill",
        )
        self.assertEqual(
            agent_type_to_skill_name("core_pipeline-agent"),
            "core_pipeline-skill",
        )

    def test_strip_agent_mentions(self) -> None:
        text = "@agent-AutoResearchClaw-topic-init-agent 只根据 agent 定义说明输出契约"
        self.assertNotIn("@agent-", strip_agent_mentions(text))
        self.assertIn("输出契约", strip_agent_mentions(text))

    def test_find_unknown_agent_mentions(self) -> None:
        from src.command_system.input_processing import (
            find_unknown_agent_mentions,
            format_unknown_agent_mention_error,
        )

        agents = [SimpleNamespace(agent_type="AutoResearchClaw-topic-init-agent")]
        text = "@agent-AutoResearchClaw-topic-init-agent-wrong 说明契约"
        unknown = find_unknown_agent_mentions(text, agents)
        self.assertEqual(unknown, ["AutoResearchClaw-topic-init-agent-wrong"])
        err = format_unknown_agent_mention_error(unknown, agents)
        self.assertIn("Unknown agent", err)
        self.assertIn("AutoResearchClaw-topic-init-agent", err)

    def test_is_prefixed_stage_agent(self) -> None:
        self.assertTrue(is_prefixed_stage_agent("AutoResearchClaw-topic-init-agent", "AutoResearchClaw"))
        self.assertFalse(is_prefixed_stage_agent("core_pipeline-agent", "AutoResearchClaw"))

    def test_read_workflow_first_stage_skill_name(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "workflow.yaml").write_text(
                "name: AutoResearchClaw\n"
                "stages:\n"
                "  - id: 1\n"
                "    phase: topic-init\n"
                "    agent_config:\n"
                "      agent: AutoResearchClaw-topic-init-agent\n",
                encoding="utf-8",
            )
            self.assertEqual(
                read_workflow_first_stage_skill_name(root),
                "topic-init-skill",
            )

    def test_read_workflow_project_name(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "workflow.yaml").write_text(
                "name: AutoResearchClaw\nversion: '1.0'\n",
                encoding="utf-8",
            )
            self.assertEqual(read_workflow_project_name(root), "AutoResearchClaw")


if __name__ == "__main__":
    unittest.main()
