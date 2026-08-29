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

"""Tests for ToolSearch token-aware matching (Plan B + P1 ranking)."""

from __future__ import annotations


# pylint: disable=E0611,W0404,W0621
import unittest
from pathlib import Path

from clawcodex_ext.agent.tool_authoring.factory import build_tool_from_spec
from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
from clawcodex_ext.tool_system.build_tool import build_tool
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.registry import ToolRegistry
from clawcodex_ext.tool_system.tools.tool_search import make_tool_search_tool
from clawcodex_ext.tool_system.tools.tool_search_matching import (
    rank_tool_matches,
    rank_tools_by_lifecycle,
    resolve_select_tool_names,
    score_tool_match,
    tokenize_query,
    tool_search_document,
)
from extensions.sop_converter.dependency import (
    IntentGroup,
    PriorityRoute,
    ToolDependency,
    ToolDependencyGraph,
)
from extensions.sop_converter.core.search_tags import generate_search_tags
from extensions.sop_converter.core.source_parser import SourceOperation
from extensions.sop_converter.tool_registry_bridge import operation_to_spec


class TestTokenizeQuery(unittest.TestCase):
    def test_splits_on_spaces_and_punctuation(self) -> None:
        self.assertEqual(
            tokenize_query("loop coordinator iteration"),
            ["loop", "coordinator", "iteration"],
        )


class TestScoreToolMatch(unittest.TestCase):
    def test_name_substring_is_tier_zero(self) -> None:
        key = score_tool_match(
            "read",
            tool_name="Read",
            document="read file",
        )
        self.assertEqual(key, (0, 0, 0, "Read"))

    def test_kebab_phrase_in_name_is_tier_zero(self) -> None:
        key = score_tool_match(
            "run team cli",
            tool_name="openjiuwen-agent-teams-cli-run-team-cli",
            search_hint="run team cli team cli",
            document="openjiuwen-agent-teams-cli-run-team-cli",
        )
        self.assertEqual(key[0], 0)

    def test_full_phrase_in_tags_is_tier_one(self) -> None:
        key = score_tool_match(
            "run team cli",
            tool_name="other-tool",
            search_hint="run team cli team cli run",
            document="description only",
        )
        self.assertEqual(key, (1, 0, 0, "other-tool"))

    def test_token_overlap_uses_tier_four_with_word_boundaries(self) -> None:
        key = score_tool_match(
            "loop coordinator iteration",
            tool_name="openjiuwen-harness-task-loop-loopcoordinator-current-iteration",
            document=(
                "openjiuwen-harness-task-loop-loopcoordinator-current-iteration\n"
                "loop coordinator current iteration should continue"
            ),
        )
        self.assertIsNotNone(key)
        assert key is not None
        self.assertEqual(key[0], 4)
        self.assertEqual(key[1], -3)

    def test_team_token_does_not_match_teams_segment(self) -> None:
        key = score_tool_match(
            "team",
            tool_name="openjiuwen-agent-teams-set-session-id",
            search_hint="teams set session agent teams",
            document="openjiuwen-agent-teams-set-session-id",
        )
        self.assertIsNone(key)


class TestRankToolMatches(unittest.TestCase):
    def _loop_tool(self):
        op = SourceOperation(
            name="current_iteration",
            description="Return the current loop iteration index.",
            class_name="LoopCoordinator",
        )
        tags = generate_search_tags(op, comp_name="openjiuwen.harness.task_loop")
        return build_tool_from_spec(
            AgentToolSpec(
                name="openjiuwen-harness-task-loop-loopcoordinator-current-iteration",
                description=op.description,
                input_schema={"type": "object", "properties": {}},
                call_type="bash",
                call_impl="echo {}",
                tags=tags,
                source="pos-converter",
            )
        )

    def _team_cli_tool(self):
        op = SourceOperation(
            name="run_team_cli",
            description="Bring up the Team CLI against YAML team specs.",
        )
        tags = generate_search_tags(op, comp_name="openjiuwen.agent_teams.cli")
        return build_tool_from_spec(
            AgentToolSpec(
                name="openjiuwen-agent-teams-cli-run-team-cli",
                description=op.description,
                input_schema={
                    "type": "object",
                    "properties": {
                        "yaml_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": ("Paths to YAML team spec files used to initialize the team conversation"),
                        },
                    },
                },
                call_type="bash",
                call_impl="echo {}",
                tags=tags,
                source="pos-converter",
            )
        )

    def _generic_teams_tool(self):
        return build_tool_from_spec(
            AgentToolSpec(
                name="openjiuwen-agent-teams-set-session-id",
                description="Set the current session id.",
                input_schema={"type": "object", "properties": {}},
                call_type="bash",
                call_impl="echo {}",
                tags=("set", "session", "id", "teams", "agent teams"),
                source="pos-converter",
            )
        )

    def test_multi_word_query_matches_tags(self) -> None:
        tools = [self._loop_tool()]
        matches = rank_tool_matches(
            "loop coordinator iteration",
            tools,
            max_results=5,
        )
        self.assertEqual(
            matches,
            ["openjiuwen-harness-task-loop-loopcoordinator-current-iteration"],
        )

    def test_phrase_query_matches_run_team_cli_tags(self) -> None:
        tools = [self._team_cli_tool()]
        matches = rank_tool_matches("run team cli", tools, max_results=5)
        self.assertEqual(matches, ["openjiuwen-agent-teams-cli-run-team-cli"])

    def test_glued_spacing_run_teamcli_matches(self) -> None:
        tools = [self._team_cli_tool()]
        matches = rank_tool_matches("run teamcli", tools, max_results=5)
        self.assertEqual(matches, ["openjiuwen-agent-teams-cli-run-team-cli"])

    def test_glued_spacing_beats_teams_noise(self) -> None:
        tools = [self._generic_teams_tool(), self._team_cli_tool()]
        matches = rank_tool_matches("run teamcli", tools, max_results=5)
        self.assertEqual(matches[0], "openjiuwen-agent-teams-cli-run-team-cli")

    def test_team_cli_subphrase_ranks_run_team_cli_first(self) -> None:
        tools = [self._generic_teams_tool(), self._team_cli_tool()]
        matches = rank_tool_matches("team cli", tools, max_results=5)
        self.assertEqual(matches[0], "openjiuwen-agent-teams-cli-run-team-cli")

    def test_team_token_does_not_rank_teams_only_tools(self) -> None:
        tools = [self._generic_teams_tool(), self._team_cli_tool()]
        matches = rank_tool_matches("team", tools, max_results=5)
        self.assertEqual(matches, ["openjiuwen-agent-teams-cli-run-team-cli"])

    def test_phrase_beats_single_token_among_many_teams_tools(self) -> None:
        noise = [
            build_tool_from_spec(
                AgentToolSpec(
                    name=f"openjiuwen-agent-teams-noise-{idx}",
                    description="Teams utility helper.",
                    input_schema={"type": "object", "properties": {}},
                    call_type="bash",
                    call_impl="echo {}",
                    tags=("teams", "agent teams", "utility"),
                    source="pos-converter",
                )
            )
            for idx in range(10)
        ]
        tools = noise + [self._team_cli_tool()]
        matches = rank_tool_matches("run team cli", tools, max_results=5)
        self.assertEqual(matches[0], "openjiuwen-agent-teams-cli-run-team-cli")

    def test_schema_param_descriptions_indexed_in_document(self) -> None:
        tool = self._team_cli_tool()
        document = tool_search_document(tool)
        self.assertIn("yaml team spec", document)
        self.assertIn("initialize", document)
        self.assertIn("conversation", document)

    def test_yaml_team_spec_query_matches_via_schema_descriptions(self) -> None:
        tools = [self._generic_teams_tool(), self._team_cli_tool()]
        matches = rank_tool_matches("yaml team spec", tools, max_results=5)
        self.assertEqual(matches[0], "openjiuwen-agent-teams-cli-run-team-cli")

    def test_initialize_conversation_query_matches_via_schema_descriptions(self) -> None:
        tools = [self._generic_teams_tool(), self._team_cli_tool()]
        matches = rank_tool_matches("initialize conversation", tools, max_results=5)
        self.assertEqual(matches[0], "openjiuwen-agent-teams-cli-run-team-cli")


class TestTeamMemoryToolDisambiguation(unittest.TestCase):
    def setUp(self) -> None:
        self.team_dir = build_tool_from_spec(
            AgentToolSpec(
                name="openjiuwen-agent-teams-team-memory-dir",
                description="Return the per-team shared memory directory.",
                input_schema={
                    "type": "object",
                    "properties": {"team_name": {"type": "string"}},
                    "required": ["team_name"],
                },
                call_type="bash",
                call_impl="echo {}",
                tags=(
                    "team_memory_dir",
                    "team memory dir",
                    "team memory directory path",
                    "get team memory dir",
                ),
                source="pos-converter",
            )
        )
        self.ensure_dir = build_tool_from_spec(
            AgentToolSpec(
                name="openjiuwen-agent-teams-memory-sharedmemorymanager-ensure-dir",
                description="Ensure team-memory/ directory exists.",
                input_schema={"type": "object", "properties": {}},
                call_type="bash",
                call_impl="echo {}",
                tags=("ensure_dir", "ensure dir", "team memory", "ensure team memory"),
                source="pos-converter",
            )
        )
        self.tools = [self.team_dir, self.ensure_dir]

    def test_path_query_ranks_team_memory_dir_first(self) -> None:
        matches = rank_tool_matches(
            "team memory directory path",
            self.tools,
            max_results=2,
        )
        self.assertEqual(matches[0], "openjiuwen-agent-teams-team-memory-dir")

    def test_select_team_memory_dir_suffix(self) -> None:
        matches = resolve_select_tool_names("team-memory-dir", self.tools)
        self.assertEqual(matches, ["openjiuwen-agent-teams-team-memory-dir"])

    def test_ensure_query_ranks_ensure_dir_first(self) -> None:
        matches = rank_tool_matches(
            "ensure team memory directory",
            self.tools,
            max_results=2,
        )
        self.assertEqual(
            matches[0],
            "openjiuwen-agent-teams-memory-sharedmemorymanager-ensure-dir",
        )


class TestToolSearchLifecycleRanking(unittest.TestCase):
    def _lifecycle_graph(self) -> ToolDependencyGraph:
        return ToolDependencyGraph(
            dependencies=[
                ToolDependency(
                    from_tool="demo.build-agent",
                    to_tool="demo.run-agent",
                    shared_params=["agent_id"],
                    lifecycle="create -> invoke",
                )
            ],
            intent_groups=[
                IntentGroup(
                    name="agent_lifecycle",
                    description="agent lifecycle",
                    tools=["demo.build-agent", "demo.run-agent"],
                    primary_entry="demo.build-agent",
                )
            ],
            priority_routes=[
                PriorityRoute(
                    keywords=["create agent", "build agent"],
                    intent_group="agent_lifecycle",
                    entry_first=True,
                ),
                PriorityRoute(
                    keywords=["run agent", "invoke agent"],
                    intent_group="agent_lifecycle",
                    entry_first=False,
                ),
            ],
        )

    def _tools(self):
        return [
            build_tool(
                name="demo-build-agent",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="create agent run agent lifecycle",
                search_hint="create agent build agent run agent lifecycle",
            ),
            build_tool(
                name="demo-run-agent",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="run agent create agent lifecycle",
                search_hint="run agent invoke agent create agent lifecycle",
            ),
            build_tool(
                name="noise-agent-helper",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="agent helper lifecycle",
                search_hint="agent helper lifecycle",
            ),
        ]

    def test_rank_tools_by_lifecycle_orders_create_and_run_routes(self) -> None:
        graph = self._lifecycle_graph()
        matches = ["noise-agent-helper", "demo-run-agent", "demo-build-agent"]
        self.assertEqual(
            rank_tools_by_lifecycle(matches, "create agent", graph)[:2],
            ["demo-build-agent", "demo-run-agent"],
        )
        self.assertEqual(
            rank_tools_by_lifecycle(matches, "run agent", graph)[:2],
            ["demo-run-agent", "demo-build-agent"],
        )

    def test_lifecycle_chain_query_returns_dependency_chain(self) -> None:
        matches = rank_tool_matches(
            "lifecycle-chain agent",
            self._tools(),
            max_results=5,
            lifecycle_graph=self._lifecycle_graph(),
        )
        self.assertEqual(matches[:2], ["demo-build-agent", "demo-run-agent"])
        self.assertNotIn("noise-agent-helper", matches[:2])


class TestToolSearchBundleAllowlist(unittest.TestCase):
    def setUp(self) -> None:
        from extensions.sop_converter.bundle_context import set_active_bundle

        self._set_active_bundle = set_active_bundle

    def tearDown(self) -> None:
        self._set_active_bundle(None)

    def test_exact_allowlisted_name_without_registered_tool(self) -> None:
        from extensions.sop_converter.bundle_context import build_bundle_context

        registry = ToolRegistry()
        for name in ("Agent", "Glob", "Grep", "Read"):
            registry.register(
                build_tool(
                    name=name,
                    input_schema={"type": "object", "properties": {}},
                    call=lambda _i, _c: None,
                    prompt=name.lower(),
                )
            )
        registry.register(make_tool_search_tool(registry))

        bundle = build_bundle_context(
            bundle_path=Path("/tmp/JiuwenAgent_tool_test"),
            skill_names=["openjiuwen_merged-skill"],
            skill_dirs=[],
            tool_names=["openjiuwen-agent-teams-team-memory-dir"],
        )
        self._set_active_bundle(bundle)
        ctx = ToolContext(workspace_root=".", bundle_context=bundle)

        tool_search = registry.get("ToolSearch")
        assert tool_search is not None
        result = tool_search.call(
            {"query": "openjiuwen-agent-teams-team-memory-dir"},
            ctx,
        )
        self.assertEqual(
            result.output["matches"],
            ["openjiuwen-agent-teams-team-memory-dir"],
        )

    def test_select_prefix_resolves_allowlisted_name(self) -> None:
        from extensions.sop_converter.bundle_context import build_bundle_context

        registry = ToolRegistry()
        registry.register(
            build_tool(
                name="Read",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="read",
            )
        )
        registry.register(make_tool_search_tool(registry))

        bundle = build_bundle_context(
            bundle_path=Path("/tmp/JiuwenAgent_tool_test"),
            skill_names=["memory-skill"],
            skill_dirs=[],
            tool_names=[
                "openjiuwen-agent-teams-memory-sharedmemorymanager-ensure-dir",
            ],
        )
        ctx = ToolContext(workspace_root=".", bundle_context=bundle)

        tool_search = registry.get("ToolSearch")
        assert tool_search is not None
        result = tool_search.call(
            {
                "query": ("select:openjiuwen-agent-teams-memory-sharedmemorymanager-ensure-dir"),
            },
            ctx,
        )
        self.assertEqual(
            result.output["matches"],
            ["openjiuwen-agent-teams-memory-sharedmemorymanager-ensure-dir"],
        )

    def test_select_registers_persisted_tool_for_next_turn(self) -> None:
        """ToolSearch must load bundle specs into registry + options.tools."""
        import os
        import tempfile

        from clawcodex_ext.agent.tool_authoring.persistence import bundle_tool_dir, save_spec
        from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
        from clawcodex_ext.query.query import _resolve_effective_tools, QueryParams
        from clawcodex_ext.tool_system.context import ToolContext, ToolUseOptions
        from extensions.sop_converter.bundle_context import build_bundle_context
        from unittest.mock import MagicMock

        spec_name = "openjiuwen-agent-teams-team-memory-dir"
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "JiuwenAgent_tool_test"
            bundle_path.mkdir(parents=True)
            save_spec(
                AgentToolSpec(
                    name=spec_name,
                    description="Return team memory directory path",
                    input_schema={
                        "type": "object",
                        "properties": {"team_name": {"type": "string"}},
                    },
                    call_type="bash",
                    call_impl='python3 -c "print({})"',
                    source="pos-converter",
                    bundle_id=bundle_path.name,
                ),
                tool_dir=bundle_tool_dir(bundle_path),
            )

            registry = ToolRegistry()
            registry.register(
                build_tool(
                    name="Skill",
                    input_schema={"type": "object", "properties": {}},
                    call=lambda _i, _c: None,
                    prompt="skill",
                )
            )
            registry.register(make_tool_search_tool(registry))

            bundle = build_bundle_context(
                bundle_path=bundle_path,
                skill_names=["openjiuwen_merged-skill"],
                skill_dirs=[],
                tool_names=[spec_name],
            )
            self._set_active_bundle(bundle)
            ctx = ToolContext(
                workspace_root=Path(tmp),
                bundle_context=bundle,
                tool_registry=registry,
                options=ToolUseOptions(tools=list(registry.list_tools())),
            )

            tool_search = registry.get("ToolSearch")
            assert tool_search is not None
            self.assertIsNone(registry.get(spec_name))

            result = tool_search.call(
                {"query": f"select:{spec_name}"},
                ctx,
            )
            self.assertEqual(result.output["matches"], [spec_name])
            self.assertIsNotNone(registry.get(spec_name))
            self.assertIn(spec_name, {t.name for t in ctx.options.tools})

            block = tool_search.map_result_to_api(result.output, "ts-select")
            messages = [
                {
                    "type": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "ts-select",
                            "content": block["content"],
                        }
                    ],
                }
            ]
            os.environ["ENABLE_TOOL_SEARCH"] = "true"
            provider = MagicMock()
            provider.model = "claude-sonnet-4-6"
            params = QueryParams(
                messages=[],
                system_prompt="test",
                tools=list(ctx.options.tools),
                tool_registry=registry,
                tool_use_context=ctx,
                provider=provider,
                abort_controller=MagicMock(),
            )
            effective = _resolve_effective_tools(params, ctx, messages)
            self.assertIn(spec_name, {t.name for t in effective})

    def test_toolsearch_reads_bundle_lifecycle_chain_query(self) -> None:
        import tempfile

        from extensions.sop_converter.bundle_context import build_bundle_context
        from extensions.sop_converter.dependency import write_tool_dependencies

        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "bundle"
            bundle_path.mkdir(parents=True)
            graph = ToolDependencyGraph(
                dependencies=[
                    ToolDependency(
                        from_tool="demo.build-agent",
                        to_tool="demo.run-agent",
                        shared_params=["agent_id"],
                        lifecycle="create -> invoke",
                    )
                ],
                intent_groups=[
                    IntentGroup(
                        name="agent_lifecycle",
                        description="agent lifecycle",
                        tools=["demo.build-agent", "demo.run-agent"],
                        primary_entry="demo.build-agent",
                    )
                ],
                priority_routes=[
                    PriorityRoute(
                        keywords=["create agent", "run agent"],
                        intent_group="agent_lifecycle",
                        entry_first=True,
                    )
                ],
            )
            write_tool_dependencies(
                graph,
                bundle_path / ".clawcodex" / "tool-dependencies.yaml",
            )

            registry = ToolRegistry()
            registry.register(
                build_tool(
                    name="demo-build-agent",
                    input_schema={"type": "object", "properties": {}},
                    call=lambda _i, _c: None,
                    prompt="create agent build agent lifecycle",
                    should_defer=True,
                )
            )
            registry.register(
                build_tool(
                    name="demo-run-agent",
                    input_schema={"type": "object", "properties": {}},
                    call=lambda _i, _c: None,
                    prompt="run agent invoke agent lifecycle",
                    should_defer=True,
                )
            )
            registry.register(make_tool_search_tool(registry))

            bundle = build_bundle_context(
                bundle_path=bundle_path,
                skill_names=["demo-skill"],
                skill_dirs=[],
                tool_names=["demo-build-agent", "demo-run-agent"],
            )
            self._set_active_bundle(bundle)
            ctx = ToolContext(workspace_root=Path(tmp), bundle_context=bundle)
            tool_search = registry.get("ToolSearch")
            assert tool_search is not None
            result = tool_search.call(
                {"query": "lifecycle-chain agent"},
                ctx,
            )
            self.assertEqual(
                result.output["matches"][:2],
                ["demo-build-agent", "demo-run-agent"],
            )


class TestToolSearchIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(
            build_tool(
                name="Read",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="Read files from disk",
            )
        )
        op = SourceOperation(
            name="run_team_cli",
            description="Bring up the Team CLI against YAML team specs.",
        )
        generate_search_tags(op, comp_name="openjiuwen.agent_teams.cli")
        spec = operation_to_spec(
            op,
            source_dir="/tmp",
            script_path="/tmp/wrapper.py",
            comp_name="openjiuwen.agent_teams.cli",
        )
        self.assertTrue(spec.tags)
        self.registry.register(build_tool_from_spec(spec))
        self.registry.register(make_tool_search_tool(self.registry))
        self.ctx = ToolContext(workspace_root=".")

    def test_operation_to_spec_populates_tags(self) -> None:
        tool = self.registry.get("openjiuwen-agent-teams-cli-run-team-cli")
        assert tool is not None
        self.assertIsNotNone(tool.search_hint)
        self.assertIn("run team cli", tool.search_hint or "")

    def test_tool_search_finds_run_team_cli_by_phrase(self) -> None:
        tool_search = self.registry.get("ToolSearch")
        assert tool_search is not None
        result = tool_search.call({"query": "run team cli"}, self.ctx)
        self.assertIn(
            "openjiuwen-agent-teams-cli-run-team-cli",
            result.output["matches"],
        )
        details = result.output.get("match_details") or []
        hit = next(
            (item for item in details if item.get("name") == "openjiuwen-agent-teams-cli-run-team-cli"),
            None,
        )
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertIn("Team CLI", hit.get("description", ""))

    def test_match_details_include_parameter_descriptions(self) -> None:
        team_cli = self.registry.get("openjiuwen-agent-teams-cli-run-team-cli")
        assert team_cli is not None
        # Enrich persisted spec with param descriptions (as operation_to_spec would).
        enriched = build_tool_from_spec(
            AgentToolSpec(
                name="openjiuwen-agent-teams-cli-run-team-cli",
                description="Bring up the Team CLI against YAML team specs.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "yaml_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": ("Paths to YAML team spec files used to initialize the team conversation"),
                        },
                    },
                },
                call_type="bash",
                call_impl="echo {}",
                tags=tuple(team_cli.search_hint.split()) if team_cli.search_hint else (),
                source="pos-converter",
            )
        )
        self.registry.unregister("openjiuwen-agent-teams-cli-run-team-cli")
        self.registry.register(enriched)
        tool_search = self.registry.get("ToolSearch")
        assert tool_search is not None
        result = tool_search.call({"query": "initialize conversation yaml"}, self.ctx)
        self.assertIn(
            "openjiuwen-agent-teams-cli-run-team-cli",
            result.output["matches"],
        )
        details = result.output.get("match_details") or []
        hit = next(
            (item for item in details if item.get("name") == "openjiuwen-agent-teams-cli-run-team-cli"),
            None,
        )
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertIn("yaml_paths", hit.get("parameter_descriptions", {}))

    def test_select_prefix_still_works(self) -> None:
        tool_search = self.registry.get("ToolSearch")
        assert tool_search is not None
        result = tool_search.call(
            {
                "query": "select:openjiuwen-agent-teams-cli-run-team-cli",
            },
            self.ctx,
        )
        self.assertEqual(
            result.output["matches"],
            ["openjiuwen-agent-teams-cli-run-team-cli"],
        )
        details = result.output.get("match_details") or []
        self.assertEqual(len(details), 1)
        self.assertIn("description", details[0])


if __name__ == "__main__":
    unittest.main()
