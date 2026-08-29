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

"""layered macro / atomic ToolSearch acceptance tests."""

from __future__ import annotations


# pylint: disable=E0611
import unittest

from clawcodex_ext.tool_system.build_tool import build_tool
from clawcodex_ext.tool_system.context import ToolContext, ToolUseOptions
from clawcodex_ext.tool_system.protocol import ToolCall, ToolResult
from clawcodex_ext.tool_system.registry import ToolRegistry
from clawcodex_ext.tool_system.tools.tool_search import make_tool_search_tool
from clawcodex_ext.tool_system.tools.tool_search_matching import rank_tool_matches
from extensions.sop_converter.runtime.macros.models import MacroRoute
from extensions.sop_converter.runtime.macros.routing import MacroRouteCatalog
from extensions.sop_converter.tool_retrieval import index_from_routes


MACRO = "invoke-existing-agent"
ATOMIC_INVOKE = "openjiuwen-core-application-llm-agent-llmagent-invoke"
ATOMIC_SEND = "openjiuwen-core-controller-legacy-basecontroller-send-to-agent"


def _result_tool(name: str, prompt: str, *, enabled: bool = True):
    return build_tool(
        name=name,
        input_schema={"type": "object", "properties": {}},
        call=lambda _i, _c: ToolResult(name=name, output={"ok": True}),
        prompt=prompt,
        is_enabled=lambda: enabled,
    )


class TestLayeredToolSearchIntegration(unittest.TestCase):
    def _runtime(self, *, macro_enabled: bool = True):
        registry = ToolRegistry()
        registry.register(_result_tool(MACRO, "invoke existing agent", enabled=macro_enabled))
        registry.register(_result_tool(ATOMIC_INVOKE, "llmagent invoke agent reply"))
        registry.register(_result_tool(ATOMIC_SEND, "send message to agent reply"))
        registry.register(make_tool_search_tool(registry))
        context = ToolContext(
            workspace_root=".",
            tool_registry=registry,
            options=ToolUseOptions(tools=list(registry.list_tools())),
        )
        return registry, context

    def test_verified_exclusive_hides_covered_atomics(self) -> None:
        registry, context = self._runtime()
        result = registry.get("ToolSearch").call(  # type: ignore[union-attr]
            {"query": "用 verify-bot 回复 ping"},
            context,
        )
        self.assertEqual(result.output["matches"], [MACRO])
        retrieval = result.output["retrieval"]
        self.assertEqual(retrieval["selection"], "exclusive")
        self.assertEqual(retrieval["intent_key"], "agent.invoke_existing")
        self.assertEqual(
            set(retrieval["suppressed_tools"]),
            {ATOMIC_INVOKE, ATOMIC_SEND},
        )
        exposed = {tool.name for tool in context.options.tools}
        self.assertIn(MACRO, exposed)
        self.assertNotIn(ATOMIC_INVOKE, exposed)
        self.assertNotIn(ATOMIC_SEND, exposed)
        self.assertEqual(context.retrieval_metrics["macro_exclusive_commit_count"], 1)
        self.assertEqual(context.retrieval_metrics["atomic_suppressed_count"], 2)

    def test_shadow_guard_blocks_stale_atomic_reference_then_macro_restores(self) -> None:
        registry, context = self._runtime()
        registry.get("ToolSearch").call(  # type: ignore[union-attr]
            {"query": "用已创建的 agent 回复"},
            context,
        )

        blocked = registry.dispatch(ToolCall(name=ATOMIC_INVOKE, input={}), context)
        self.assertTrue(blocked.is_error)
        self.assertEqual(blocked.output["error_code"], "tool_shadowed_by_macro")
        self.assertEqual(blocked.output["recommended_tool"], MACRO)
        self.assertEqual(context.retrieval_metrics["shadowed_atomic_call_count"], 1)

        macro_result = registry.dispatch(ToolCall(name=MACRO, input={}), context)
        self.assertFalse(macro_result.is_error)
        self.assertIsNone(context.retrieval_plan)
        restored = {tool.name for tool in context.options.tools}
        self.assertIn(ATOMIC_INVOKE, restored)
        self.assertIn(ATOMIC_SEND, restored)

    def test_macro_preflight_failure_restores_atomics_same_search(self) -> None:
        registry, context = self._runtime(macro_enabled=False)
        result = registry.get("ToolSearch").call(  # type: ignore[union-attr]
            {"query": "llmagent invoke verify-bot ping"},
            context,
        )
        self.assertNotIn(MACRO, result.output["matches"])
        self.assertEqual(
            result.output["matches"][:2],
            [ATOMIC_INVOKE, ATOMIC_SEND],
        )
        self.assertEqual(result.output["retrieval"]["preflight"], "unavailable")
        self.assertIn("atomic_restore", result.output["retrieval"]["reason_codes"])
        self.assertIsNone(context.retrieval_plan)
        self.assertEqual(context.retrieval_metrics["macro_preflight_failure_count"], 1)
        self.assertEqual(context.retrieval_metrics["atomic_restore_count"], 2)

    def test_new_search_restores_previous_hidden_tools(self) -> None:
        registry, context = self._runtime()
        tool_search = registry.get("ToolSearch")
        tool_search.call({"query": "用已创建的 agent 回复"}, context)  # type: ignore[union-attr]
        self.assertNotIn(ATOMIC_INVOKE, {tool.name for tool in context.options.tools})

        tool_search.call({"query": "select:" + ATOMIC_INVOKE}, context)  # type: ignore[union-attr]
        self.assertIn(ATOMIC_INVOKE, {tool.name for tool in context.options.tools})
        self.assertIsNone(context.retrieval_plan)

    def test_adjacent_management_intents_do_not_commit_invoke_exclusive(self) -> None:
        for query in ("创建 agent", "配置 verify-bot", "删除 verify-bot", "列出已有 agent"):
            with self.subTest(query=query):
                registry, context = self._runtime()
                result = registry.get("ToolSearch").call(  # type: ignore[union-attr]
                    {"query": query},
                    context,
                )
                self.assertIsNone(context.retrieval_plan)
                retrieval = result.output.get("retrieval")
                if retrieval is not None:
                    self.assertNotEqual(retrieval.get("selection"), "exclusive")


class TestLayeredNormalScoring(unittest.TestCase):
    def test_same_semantic_tier_prefers_macro_over_covered_atomic(self) -> None:
        route = MacroRoute(
            target_tool="z-agent-workflow",
            selection="prefer",
            intent_key="agent.invoke_existing",
            covered_tools=["a-agent-invoke"],
        )
        index = index_from_routes(
            [route],
            ["z-agent-workflow", "a-agent-invoke"],
            require_unique=True,
        )
        tools = [
            _result_tool("a-agent-invoke", "invoke existing agent"),
            _result_tool("z-agent-workflow", "invoke existing agent"),
        ]
        matches = rank_tool_matches(
            "invoke existing agent",
            tools,
            max_results=5,
            macro_route_catalog=MacroRouteCatalog(),
            retrieval_index=index,
        )
        self.assertEqual(matches[:2], ["z-agent-workflow", "a-agent-invoke"])

    def test_prefer_route_keeps_atomic_candidates(self) -> None:
        catalog = MacroRouteCatalog()
        catalog.register_route(
            MacroRoute(
                keywords=["agent", "invoke"],
                target_tool="agent-workflow",
                selection="prefer",
                intent_key="agent.invoke_existing",
                covered_tools=["agent-invoke"],
            )
        )
        tools = [
            _result_tool("agent-workflow", "agent invoke workflow"),
            _result_tool("agent-invoke", "agent invoke atomic"),
        ]
        matches = rank_tool_matches(
            "agent invoke",
            tools,
            max_results=5,
            macro_route_catalog=catalog,
        )
        self.assertEqual(matches[0], "agent-workflow")
        self.assertIn("agent-invoke", matches)


if __name__ == "__main__":
    unittest.main()
