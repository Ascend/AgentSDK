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

"""Phase 0 SOP fixes: Skill errors + ToolSearch discovery."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from pathlib import Path

from clawcodex_ext.agent.tool_authoring.factory import build_tool_from_spec
from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
from clawcodex_ext.tool_system.build_tool import build_tool
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.registry import ToolRegistry
from clawcodex_ext.tool_system.tools.skill import _skill_map_result_to_api
from clawcodex_ext.tool_system.tools.tool_search import make_tool_search_tool
from clawcodex_ext.tool_system.tool_search import (
    extract_discovered_tool_names,
    filter_tools_for_request,
)
from extensions.sop_converter.sop_prompts import domain_agent_sop_body


class TestSkillErrorDisplay(unittest.TestCase):
    def test_error_shows_message_not_unknown(self) -> None:
        block = _skill_map_result_to_api(
            {"error": "skill not found: openjiuwen_merged-skill"},
            "tu-1",
        )
        self.assertIn("Skill error:", block["content"])
        self.assertIn("skill not found", block["content"])
        self.assertNotIn("unknown", block["content"])

    def test_success_shows_skill_name(self) -> None:
        block = _skill_map_result_to_api(
            {"success": True, "commandName": "openjiuwen_merged-skill"},
            "tu-2",
        )
        self.assertEqual(block["content"], "Launching skill: openjiuwen_merged-skill")

    def test_missing_command_name_is_error(self) -> None:
        block = _skill_map_result_to_api({"success": True}, "tu-3")
        self.assertIn("Skill error:", block["content"])
        self.assertIn("missing commandName", block["content"])


class TestToolSearchDiscovery(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(
            build_tool(
                name="Read",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="Read files",
            )
        )
        deferred = build_tool_from_spec(
            AgentToolSpec(
                name="openjiuwen-agent-teams-get-openjiuwen-home",
                description="Get openjiuwen home directory",
                input_schema={"type": "object", "properties": {}},
                call_type="bash",
                call_impl="echo {}",
                source="pos-converter",
            )
        )
        self.registry.register(deferred)
        self.registry.register(make_tool_search_tool(self.registry))
        self.ctx = ToolContext(workspace_root=Path("."))

    def test_map_result_includes_tool_reference(self) -> None:
        tool_search = self.registry.get("ToolSearch")
        assert tool_search is not None
        result = tool_search.call({"query": "get-openjiuwen-home"}, self.ctx)
        block = tool_search.map_result_to_api(result.output, "ts-1")
        content = block["content"]
        self.assertIsInstance(content, list)
        refs = [b for b in content if b.get("type") == "tool_reference"]
        self.assertEqual(
            [r["tool_name"] for r in refs],
            ["openjiuwen-agent-teams-get-openjiuwen-home"],
        )

    def test_discovered_deferred_tool_included_in_next_request(self) -> None:
        tool_search = self.registry.get("ToolSearch")
        assert tool_search is not None
        result = tool_search.call({"query": "get-openjiuwen-home"}, self.ctx)
        block = tool_search.map_result_to_api(result.output, "ts-2")
        messages = [
            {
                "type": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "ts-2",
                        "content": block["content"],
                    }
                ],
            }
        ]
        tools = [
            self.registry.get("Read"),
            tool_search,
            self.registry.get("openjiuwen-agent-teams-get-openjiuwen-home"),
        ]
        tools = [t for t in tools if t is not None]
        with patch.dict(
            os.environ,
            {"ENABLE_TOOL_SEARCH": "true", "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": ""},
        ):
            filtered = filter_tools_for_request(tools, "claude-sonnet-4-6", messages=messages)
        names = [t.name for t in filtered]
        self.assertIn("openjiuwen-agent-teams-get-openjiuwen-home", names)

    def test_json_string_tool_result_fallback(self) -> None:
        payload = json.dumps(
            {
                "matches": ["openjiuwen-agent-teams-get-openjiuwen-home"],
                "query": "get-openjiuwen-home",
            }
        )
        discovered = extract_discovered_tool_names(
            [
                {
                    "type": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "legacy",
                            "content": payload,
                        }
                    ],
                }
            ]
        )
        self.assertIn("openjiuwen-agent-teams-get-openjiuwen-home", discovered)


class TestSopPromptsPhase0(unittest.TestCase):
    def test_domain_body_documents_select_and_no_repeat_skill(self) -> None:
        body = domain_agent_sop_body(
            agent_type="openjiuwen_merged-agent",
            description="test",
            skill_name="openjiuwen_merged-skill",
        )
        self.assertIn("任务指南", body)
        self.assertIn("同义词", body)
        self.assertIn("禁止 Skill 成功后再次调用 Skill", body)
        self.assertIn("禁止要求用户提供具体工具名", body)

    def test_domain_body_no_kebab_requirement(self) -> None:
        body = domain_agent_sop_body(
            agent_type="openjiuwen_merged-agent",
            description="test",
            skill_name="openjiuwen_merged-skill",
        )
        self.assertNotIn("kebab 片段", body)

    def test_domain_body_is_sdk_agnostic(self) -> None:
        body = domain_agent_sop_body(
            agent_type="dev_tools_merged-agent",
            description="test",
            skill_name="dev_tools_merged-skill",
        )
        self.assertNotIn("JiuwenAgent", body)
        # Domain prompts may mention platform config dirs (.clawcodex); they
        # must not hard-code a specific SDK product name.
        self.assertIn("SDK 源码树", body)

    def test_domain_body_forbids_post_toolsearch_exploration(self) -> None:
        body = domain_agent_sop_body(
            agent_type="openjiuwen_merged-agent",
            description="test",
            skill_name="openjiuwen_merged-skill",
        )
        self.assertIn("agent-tools/", body)
        self.assertIn("msg", body)
        self.assertIn("suggestions", body)
        self.assertIn("Explore", body)

    def test_domain_body_uses_example_params_without_reasking(self) -> None:
        body = domain_agent_sop_body(
            agent_type="openjiuwen_merged-agent",
            description="test",
            skill_name="openjiuwen_merged-skill",
        )
        self.assertIn("示例参数时直接使用", body)
        self.assertIn("不要向用户重复确认", body)

    def test_domain_body_includes_interactive_terminal_stop_loss(self) -> None:
        body = domain_agent_sop_body(
            agent_type="openjiuwen_merged-agent",
            description="test",
            skill_name="openjiuwen_merged-skill",
        )
        self.assertIn("交互式终端停损", body)
        self.assertIn("交互式终端 / TUI / REPL", body)
        self.assertIn("tests/", body)
        self.assertIn("fixtures", body)

    def test_stop_loss_forbids_fabricated_cli(self) -> None:
        body = domain_agent_sop_body(
            agent_type="openjiuwen_merged-agent",
            description="test",
            skill_name="openjiuwen_merged-skill",
        )
        self.assertIn("禁止编造", body)
        self.assertIn("call_impl", body)
        self.assertIn("--flag", body)
        self.assertIn("没有终端入口命令", body)


if __name__ == "__main__":
    unittest.main()
