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

"""Tests for dynamic cross-domain orchestration route generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from extensions.sop_converter.cross_domain_orchestration import (
    ORCHESTRATION_ROUTES_MAX,
    _should_suppress_multi_step_route,
    _title_from_operation,
    discover_orchestration_routes,
    format_orchestration_routes_block,
    generate_orchestration_routes_markdown,
    OrchestrationRoute,
    OrchestrationStep,
    write_orchestration_routes,
)
from extensions.sop_converter.core.intent_tags import collect_intent_phrases
from extensions.sop_converter.skill_grouper import SkillSpec
from extensions.sop_converter.source_parser import ParamSpec, SourceComponent, SourceOperation
from extensions.sop_converter.task_guide import _format_required_params_note
from extensions.sop_converter.tool_dependencies import build_tool_dependency_index
from extensions.sop_converter.sop_prompts import SOP_OVERVIEW_ROUTING, append_sop_overview_routing


class TestCrossDomainOrchestration(unittest.TestCase):
    def test_title_derived_from_docstring_not_hardcoded(self) -> None:
        op = SourceOperation(
            name="run_team_cli",
            description="Bring up the Team CLI against YAML specs.",
        )
        self.assertEqual(_title_from_operation(op), "Bring up the Team CLI against YAML specs")
        phrases = collect_intent_phrases(op)
        self.assertTrue(any("team cli" in p.lower() for p in phrases))

    def test_param_note_from_schema_not_hardcoded(self) -> None:
        comp = SourceComponent(name="cli", file_path="cli/app.py", description="")
        op = SourceOperation(
            name="run_team_cli",
            description="Bring up the Team CLI.",
            parameters=[ParamSpec(name="yaml_paths", type_hint="list", required=True)],
        )
        note = _format_required_params_note(comp, op)
        self.assertIn("yaml_paths", note)
        self.assertNotIn("启动团队", note)

    def test_cli_route_title_and_agent_from_metadata(self) -> None:
        cli_op = SourceOperation(
            name="run_team_cli",
            description="Bring up the Team CLI against YAML specs.",
            file_stem="cli",
            has_docstring=True,
            is_async=True,
            parameters=[ParamSpec(name="yaml_paths", type_hint="list", required=True)],
        )
        session_op = SourceOperation(
            name="create_agent_team_session",
            description="Create agent team session.",
            return_type="SessionHandle",
        )
        run_op = SourceOperation(
            name="run_agent_team",
            description="Run agent team.",
            parameters=[ParamSpec(name="agent_team", type_hint="TeamAgentSpec")],
        )
        components = [
            SourceComponent(
                name="openjiuwen.agent_teams.cli",
                file_path="openjiuwen/agent_teams/cli/app.py",
                description="CLI",
                operations=[cli_op],
            ),
            SourceComponent(
                name="openjiuwen.core.session",
                file_path="openjiuwen/core/session.py",
                description="Session",
                operations=[session_op],
            ),
            SourceComponent(
                name="openjiuwen.core.runner",
                file_path="openjiuwen/core/runner.py",
                description="Runner",
                operations=[run_op],
            ),
        ]
        skills = [
            SkillSpec(
                name="agent_teams",
                description="Teams",
                allowed_tools=["openjiuwen-agent-teams-cli-run-team-cli"],
            ),
            SkillSpec(
                name="core_engine",
                description="Core",
                allowed_tools=[
                    "openjiuwen-core-session-create-agent-team-session",
                    "openjiuwen-core-runner-teamrunnermixin-run-agent-team",
                ],
            ),
        ]
        deps = build_tool_dependency_index(components)
        routes = discover_orchestration_routes(skills, components=components, tool_deps_index=deps)
        self.assertTrue(routes)
        cli_routes = [r for r in routes if len(r.steps) == 1]
        self.assertEqual(len(cli_routes), 1)
        self.assertIn("Bring up the Team CLI", cli_routes[0].title)
        self.assertEqual(cli_routes[0].steps[0].agent, "agent_teams-agent")
        self.assertIn("yaml_paths", cli_routes[0].steps[0].param_hint)

    def test_suppress_cross_domain_programmatic_when_cli_exists(self) -> None:
        components = [
            SourceComponent(
                name="alpha.cli",
                file_path="alpha/cli/app.py",
                description="CLI",
                operations=[
                    SourceOperation(
                        name="run_widget_cli",
                        description="Start the widget UI.",
                        file_stem="cli",
                        has_docstring=True,
                        is_async=True,
                    ),
                ],
            ),
            SourceComponent(
                name="alpha.core",
                file_path="alpha/core.py",
                description="Alpha core",
                operations=[
                    SourceOperation(
                        name="make_config",
                        description="Build widget config.",
                        has_docstring=True,
                        return_type="WidgetConfig",
                    ),
                ],
            ),
            SourceComponent(
                name="beta.core",
                file_path="beta/core.py",
                description="Beta core",
                operations=[
                    SourceOperation(
                        name="run_widget",
                        description="Run widget engine.",
                        has_docstring=True,
                        parameters=[ParamSpec(name="config", type_hint="WidgetConfig")],
                    ),
                ],
            ),
        ]
        skills = [
            SkillSpec(
                name="alpha",
                description="Alpha",
                allowed_tools=[
                    "alpha-cli-run-widget-cli",
                    "alpha-core-make-config",
                ],
            ),
            SkillSpec(
                name="beta",
                description="Beta",
                allowed_tools=["beta-core-run-widget"],
            ),
        ]
        deps = build_tool_dependency_index(components)
        routes = discover_orchestration_routes(skills, components=components, tool_deps_index=deps)
        tool_refs = [tool for route in routes for tool in route.tool_refs]
        self.assertIn("alpha-cli-run-widget-cli", tool_refs)
        self.assertNotIn("beta-core-run-widget", tool_refs)

    def test_should_suppress_is_generic(self) -> None:
        route = OrchestrationRoute(
            title="cross",
            tool_refs=["beta-core-build-widget", "beta-core-run-widget"],
            steps=[
                OrchestrationStep(agent="alpha-agent", flow="a"),
                OrchestrationStep(agent="beta-agent", flow="b"),
            ],
        )
        suppressed = _should_suppress_multi_step_route(
            route,
            cli_agents={"alpha-agent"},
            components=[
                SourceComponent(
                    name="beta.core",
                    file_path="beta/core.py",
                    description="",
                    operations=[
                        SourceOperation(
                            name="run_widget",
                            description="Run widget.",
                            has_docstring=True,
                        ),
                    ],
                ),
            ],
        )
        self.assertTrue(suppressed)

    def test_write_and_format_orchestration_block_generic_header(self) -> None:
        cli_op = SourceOperation(
            name="run_demo_cli",
            description="Start the demo CLI.",
            file_stem="cli",
            has_docstring=True,
            is_async=True,
        )
        skills = [
            SkillSpec(
                name="demo",
                description="Demo",
                allowed_tools=["example-demo-cli-run-demo-cli"],
            ),
        ]
        components = [
            SourceComponent(
                name="example.demo.cli",
                file_path="example/demo/cli/app.py",
                description="CLI",
                operations=[cli_op],
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            path = write_orchestration_routes(
                bundle,
                skills,
                components=components,
            )
            self.assertTrue(path.is_file())
            block = format_orchestration_routes_block(bundle)
            self.assertIn("ORCHESTRATION_ROUTES.md", block)
            self.assertIn("Read", block)
            self.assertNotIn("Start the demo CLI", block)
            inlined = format_orchestration_routes_block(bundle, inline_content=True)
            self.assertIn("Start the demo CLI", inlined)
            self.assertIn("[单步 CLI]", inlined)
            self.assertIn("单步交互式 CLI", inlined)
            self.assertNotIn("create-agent-team-session", inlined)
            self.assertNotIn("run-team-cli", inlined)

    def test_orchestration_routes_max_is_40(self) -> None:
        self.assertEqual(ORCHESTRATION_ROUTES_MAX, 40)

    def test_cli_route_heading_prefixed_in_markdown(self) -> None:
        cli_op = SourceOperation(
            name="run_demo_cli",
            description="Start the demo CLI.",
            file_stem="cli",
            has_docstring=True,
            is_async=True,
        )
        skills = [
            SkillSpec(
                name="demo",
                description="Demo",
                allowed_tools=["example-demo-cli-run-demo-cli"],
            ),
        ]
        components = [
            SourceComponent(
                name="example.demo.cli",
                file_path="example/demo/cli/app.py",
                description="CLI",
                operations=[cli_op],
            ),
        ]
        md = generate_orchestration_routes_markdown(skills, components=components)
        self.assertIn("### [单步 CLI] Start the demo CLI", md)

    def test_overview_routing_no_hardcoded_examples(self) -> None:
        self.assertNotIn("openjiuwen_merged-agent", SOP_OVERVIEW_ROUTING)
        self.assertNotIn("create-agent-team-session", SOP_OVERVIEW_ROUTING)
        self.assertIn("跨域编排", SOP_OVERVIEW_ROUTING)

    def test_append_overview_includes_orchestration_block(self) -> None:
        cli_op = SourceOperation(
            name="run_demo_cli",
            description="Start the demo CLI.",
            file_stem="cli",
            has_docstring=True,
            is_async=True,
        )
        skills = [
            SkillSpec(
                name="demo",
                description="Demo",
                allowed_tools=["example-demo-cli-run-demo-cli"],
            ),
        ]
        components = [
            SourceComponent(
                name="example.demo.cli",
                file_path="example/demo/cli/app.py",
                description="CLI",
                operations=[cli_op],
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            write_orchestration_routes(bundle, skills, components=components)
            body = append_sop_overview_routing("", bundle_path=bundle)
            self.assertIn("ORCHESTRATION_ROUTES.md", body)
            self.assertIn("Read", body)
            self.assertNotIn("Start the demo CLI", body)


if __name__ == "__main__":
    unittest.main()
