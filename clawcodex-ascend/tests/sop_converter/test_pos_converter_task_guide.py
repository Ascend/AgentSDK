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

"""Tests for SOP task guide generation and intent tags."""

from __future__ import annotations

import unittest

from extensions.sop_converter.core.intent_tags import (
    collect_intent_phrases,
    enrich_with_intent_tags,
    get_intent_tags,
)
from extensions.sop_converter.core.search_tags import generate_search_tags
from extensions.sop_converter.skill_grouper import SkillSpec
from extensions.sop_converter.core.source_parser import (
    ParamSpec,
    SourceComponent,
    SourceOperation,
)
from extensions.sop_converter.task_guide import (
    _ensure_registry_chain_entries,
    _entry_point_score,
    _extract_noun,
    _resolve_operation,
    _select_task_guide_entries,
    _task_guide_rank_key,
    build_operation_index,
    format_flat_skill_markdown,
    generate_task_guide_markdown,
    is_entry_point,
)
from extensions.sop_converter.sop_prompts import SOP_OVERVIEW_ROUTING


def _entry_op() -> SourceOperation:
    return SourceOperation(
        name="open_widget_ui",
        description="Bring up the Widget UI against YAML specs.",
        file_stem="cli",
        has_docstring=True,
        is_async=True,
        parameters=[],
    )


def _factory_op() -> SourceOperation:
    return SourceOperation(
        name="make_widget_session",
        description="Create a programmatic widget session object.",
        file_stem="core",
        has_docstring=True,
    )


class TestIntentTags(unittest.TestCase):
    def test_phrases_come_from_docstring_not_name_patterns(self) -> None:
        op = _entry_op()
        tags = collect_intent_phrases(op)
        joined = " ".join(tags).lower()
        self.assertIn("bring up the widget ui", joined)
        self.assertIn("bring up", joined)
        # intent_tags does not invent run_*_cli style phrases from the name alone
        op_no_doc = SourceOperation(name="run_widget_cli", description="")
        self.assertEqual(collect_intent_phrases(op_no_doc), ())

    def test_generate_search_tags_name_and_docstring(self) -> None:
        op = _entry_op()
        tags = generate_search_tags(op, comp_name="example.widgets.cli")
        joined = " ".join(tags).lower()
        self.assertIn("open widget ui", joined)
        self.assertIn("bring up", joined)

    def test_enrich_adds_docstring_phrases(self) -> None:
        op = _entry_op()
        merged = enrich_with_intent_tags(op, base_tags=("open_widget_ui",))
        self.assertIn("open_widget_ui", merged)
        self.assertIn("bring up the widget ui", " ".join(merged).lower())

    def test_get_intent_tags_wrapper(self) -> None:
        tags = get_intent_tags(
            "open_widget_ui",
            description="Bring up the Widget UI.",
        )
        self.assertIn("bring up the widget ui", " ".join(tags).lower())

    def test_cjk_in_docstring_becomes_phrase(self) -> None:
        op = SourceOperation(
            name="start_app",
            description="启动应用交互界面。Start the interactive UI.",
        )
        tags = collect_intent_phrases(op)
        self.assertTrue(any("启动" in t for t in tags))


class TestTaskGuide(unittest.TestCase):
    def setUp(self) -> None:
        self.components = [
            SourceComponent(
                name="cli",
                file_path="example_sdk/widgets/cli.py",
                description="Widget CLI",
                operations=[_entry_op(), _factory_op()],
            ),
            SourceComponent(
                name="core",
                file_path="example_sdk/widgets/core.py",
                description="Widget core",
                operations=[_factory_op()],
            ),
        ]
        self.skill = SkillSpec(
            name="widgets_merged",
            description="Widget orchestration domain",
            allowed_tools=[
                "example-widgets-cli-open-widget-ui",
                "example-widgets-core-make-widget-session",
            ],
        )

    def test_entry_op_is_entry_point(self) -> None:
        comp, op = self.components[0], self.components[0].operations[0]
        self.assertTrue(is_entry_point(comp, op))

    def test_entry_op_scores_higher_than_factory_in_core_module(self) -> None:
        cli_comp, entry = self.components[0], self.components[0].operations[0]
        core_comp, factory = self.components[1], self.components[1].operations[0]
        self.assertGreater(
            _entry_point_score(cli_comp, entry),
            _entry_point_score(core_comp, factory),
        )

    def test_generates_task_guide_with_entry_op(self) -> None:
        guide = generate_task_guide_markdown(self.skill, self.components)
        self.assertIn("## 任务指南", guide)
        self.assertIn("example-widgets-cli-open-widget-ui", guide)
        self.assertIn("Bring up the Widget UI", guide)

    def test_flat_skill_markdown_includes_task_guide(self) -> None:
        md = format_flat_skill_markdown(self.skill, components=self.components)
        self.assertIn("## 任务指南", md)
        self.assertIn("## Included Tools", md)
        self.assertIn("example-widgets-cli-open-widget-ui", md)

    def test_cli_entry_ranks_above_build_helpers(self) -> None:
        cli_op = SourceOperation(
            name="run_team_cli",
            description="Bring up the Team CLI against YAML specs.",
            file_stem="cli",
            has_docstring=True,
            is_async=True,
            parameters=[ParamSpec(name="yaml_paths", type_hint="list", required=True)],
        )
        build_op = SourceOperation(
            name="build_agent_factory",
            description="Build a default AgentFactory from runtime config + tools.",
            file_stem="runtime",
            has_docstring=True,
        )
        cli_key = _task_guide_rank_key(
            SourceComponent(name="cli", file_path="sdk/cli/app.py", description=""),
            cli_op,
            "example-cli-run-team-cli",
        )
        build_key = _task_guide_rank_key(
            SourceComponent(name="runtime", file_path="sdk/runtime.py", description=""),
            build_op,
            "example-runtime-build-agent-factory",
        )
        self.assertGreater(cli_key, build_key)

    def test_interactive_cli_gets_terminal_footnote(self) -> None:
        components = [
            SourceComponent(
                name="cli",
                file_path="example_sdk/teams/cli/app.py",
                description="Team CLI",
                operations=[
                    SourceOperation(
                        name="run_team_cli",
                        description="Bring up the Team CLI against YAML specs.",
                        file_stem="cli",
                        has_docstring=True,
                        is_async=True,
                        parameters=[ParamSpec(name="yaml_paths", type_hint="list", required=True)],
                    ),
                ],
            ),
        ]
        skill = SkillSpec(
            name="teams_merged",
            description="Team orchestration",
            allowed_tools=["example-teams-cli-run-team-cli"],
        )
        guide = generate_task_guide_markdown(skill, components)
        self.assertIn("example-teams-cli-run-team-cli", guide)
        self.assertIn("yaml_paths", guide)
        self.assertIn("交互式终端停损", guide)
        self.assertIn("tests/fixtures", guide)

    def test_class_method_summary_includes_init_params(self) -> None:
        components = [
            SourceComponent(
                name="memory",
                file_path="example_sdk/memory/manager.py",
                description="Shared memory",
                operations=[
                    SourceOperation(
                        name="ensure_dir",
                        description="Ensure team-memory directory exists.",
                        class_name="SharedMemoryManager",
                        file_stem="memory",
                        has_docstring=True,
                    ),
                ],
                class_init_params={
                    "SharedMemoryManager": [
                        ParamSpec(name="team_memory_dir", type_hint="str", required=True),
                    ],
                },
            ),
        ]
        skill = SkillSpec(
            name="memory",
            description="Memory APIs",
            allowed_tools=["example-memory-sharedmemorymanager-ensure-dir"],
        )
        guide = generate_task_guide_markdown(skill, components)
        self.assertIn("example-memory-sharedmemorymanager-ensure-dir", guide)
        self.assertIn("team_memory_dir", guide)

    def test_build_helpers_capped_in_task_guide(self) -> None:
        build_ops = [
            SourceOperation(
                name=f"build_{idx}_section",
                description=f"Build section {idx} for prompts.",
                file_stem="prompts",
                has_docstring=True,
            )
            for idx in range(8)
        ]
        cli_op = SourceOperation(
            name="run_app_cli",
            description="Run the application CLI.",
            file_stem="cli",
            has_docstring=True,
            is_async=True,
        )
        components = [
            SourceComponent(
                name="prompts",
                file_path="example_sdk/prompts/sections.py",
                description="Prompt sections",
                operations=build_ops,
            ),
            SourceComponent(
                name="cli",
                file_path="example_sdk/cli/app.py",
                description="CLI",
                operations=[cli_op],
            ),
        ]
        allowed = [f"example-prompts-build-{idx}-section" for idx in range(8)]
        allowed.append("example-cli-run-app-cli")
        skill = SkillSpec(
            name="merged",
            description="Large merged skill",
            allowed_tools=allowed,
        )
        ranked = []
        index = build_operation_index(components)
        for tool_ref in skill.allowed_tools:
            resolved = _resolve_operation(tool_ref, index)
            if resolved is None:
                continue
            comp, op = resolved
            if not is_entry_point(comp, op):
                continue
            ranked.append((_task_guide_rank_key(comp, op, tool_ref), tool_ref, comp, op))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = _select_task_guide_entries(ranked, max_entries=12)
        tool_names = [ref for ref, _comp, op in selected]
        self.assertIn("example-cli-run-app-cli", tool_names)
        build_selected = sum(1 for _ref, _comp, op in selected if op.name.startswith("build_"))
        self.assertLessEqual(build_selected, 4)


class TestSopOverviewRouting(unittest.TestCase):
    def test_forbids_general_purpose_delegation(self) -> None:
        self.assertIn("general-purpose", SOP_OVERVIEW_ROUTING)
        self.assertIn("禁止", SOP_OVERVIEW_ROUTING)
        self.assertIn("SDK 模块总览", SOP_OVERVIEW_ROUTING)
        self.assertIn("SDK_OVERVIEW.md", SOP_OVERVIEW_ROUTING)
        self.assertIn("跨域编排", SOP_OVERVIEW_ROUTING)

    def test_includes_interactive_terminal_stop_loss(self) -> None:
        self.assertIn("交互式终端停损", SOP_OVERVIEW_ROUTING)
        self.assertIn("交互式终端 / TUI / REPL", SOP_OVERVIEW_ROUTING)
        self.assertIn("tests/", SOP_OVERVIEW_ROUTING)
        self.assertIn("fixtures", SOP_OVERVIEW_ROUTING)


class TestToolSearchDocstringQuery(unittest.TestCase):
    def test_docstring_query_ranks_entry_tool(self) -> None:
        from clawcodex_ext.agent.tool_authoring.factory import build_tool_from_spec
        from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
        from clawcodex_ext.tool_system.tools.tool_search_matching import rank_tool_matches

        op = _entry_op()
        tags = generate_search_tags(op, comp_name="example.widgets.cli")
        entry_tool = build_tool_from_spec(
            AgentToolSpec(
                name="example-widgets-cli-open-widget-ui",
                description=op.description,
                input_schema={"type": "object", "properties": {}},
                call_type="bash",
                call_impl="echo {}",
                tags=tags,
                source="pos-converter",
            )
        )
        noise_tool = build_tool_from_spec(
            AgentToolSpec(
                name="example-core-utils-get-chat-history",
                description="Get chat history for logging.",
                input_schema={"type": "object", "properties": {}},
                call_type="bash",
                call_impl="echo {}",
                tags=("chat", "history", "logging"),
                source="pos-converter",
            )
        )
        matches = rank_tool_matches(
            "bring up widget ui",
            [noise_tool, entry_tool],
            max_results=5,
        )
        self.assertEqual(matches[0], "example-widgets-cli-open-widget-ui")


class TestRegistryChainInjection(unittest.TestCase):
    """_ensure_registry_chain_entries — SDK-adaptive registry tool surfacing."""

    @staticmethod
    def _make_op(name: str, description: str = "", **kwargs: object) -> SourceOperation:
        kw: dict[str, object] = {"name": name, "description": description, "has_docstring": True}
        kw.update(kwargs)  # type: ignore[call-overload]
        return SourceOperation(**kw)  # type: ignore[arg-type]

    def _make_comp(self, comp_name: str, ops: list[SourceOperation]) -> SourceComponent:
        return SourceComponent(
            name=comp_name,
            file_path=f"example/{comp_name}.py",
            description=f"{comp_name} component",
            operations=ops,
        )

    def test_registry_injected_when_runner_present(self) -> None:
        """add_agent is injected when run_agent is already selected."""
        runner_op = self._make_op("run_agent", "Run a registered agent.")
        reg_op = self._make_op("add_agent", "Register an agent with the Runner.")
        components = [
            self._make_comp("runner", [runner_op]),
            self._make_comp("resources", [reg_op]),
        ]
        index = build_operation_index(components)
        # Simulate: only run_agent is selected by normal scoring
        selected: list[tuple[str, SourceComponent, SourceOperation]] = [
            ("runner-run-agent", components[0], runner_op),
        ]
        skill = SkillSpec(
            name="core",
            description="Core APIs",
            allowed_tools=["runner-run-agent", "resources-add-agent"],
        )
        result = _ensure_registry_chain_entries(selected, skill, index)
        tool_names = [ref for ref, _comp, _op in result]
        self.assertIn("runner-run-agent", tool_names)
        self.assertIn("resources-add-agent", tool_names)

    def test_registry_not_injected_without_matching_noun(self) -> None:
        """add_workflow is NOT injected when only run_agent is selected."""
        runner_op = self._make_op("run_agent", "Run a registered agent.")
        reg_op = self._make_op("add_workflow", "Register a workflow.")
        components = [
            self._make_comp("runner", [runner_op]),
            self._make_comp("resources", [reg_op]),
        ]
        index = build_operation_index(components)
        selected: list[tuple[str, SourceComponent, SourceOperation]] = [
            ("runner-run-agent", components[0], runner_op),
        ]
        skill = SkillSpec(
            name="core",
            description="Core APIs",
            allowed_tools=["runner-run-agent", "resources-add-workflow"],
        )
        result = _ensure_registry_chain_entries(selected, skill, index)
        tool_names = [ref for ref, _comp, _op in result]
        self.assertIn("runner-run-agent", tool_names)
        self.assertNotIn("resources-add-workflow", tool_names)

    def test_plural_matches_singular(self) -> None:
        """add_agents (plural) matches run_agent (singular)."""
        runner_op = self._make_op("run_agent", "Run a registered agent.")
        reg_op = self._make_op("add_agents", "Register agents with the Runner.")
        components = [
            self._make_comp("runner", [runner_op]),
            self._make_comp("resources", [reg_op]),
        ]
        index = build_operation_index(components)
        selected: list[tuple[str, SourceComponent, SourceOperation]] = [
            ("runner-run-agent", components[0], runner_op),
        ]
        skill = SkillSpec(
            name="core",
            description="Core APIs",
            allowed_tools=["runner-run-agent", "resources-add-agents"],
        )
        result = _ensure_registry_chain_entries(selected, skill, index)
        tool_names = [ref for ref, _comp, _op in result]
        self.assertIn("resources-add-agents", tool_names)

    def test_no_duplicate_when_already_selected(self) -> None:
        """Registry tool is not duplicated if already in the selected list."""
        runner_op = self._make_op("run_agent", "Run a registered agent.")
        reg_op = self._make_op("add_agent", "Register an agent with the Runner.")
        components = [
            self._make_comp("runner", [runner_op]),
            self._make_comp("resources", [reg_op]),
        ]
        index = build_operation_index(components)
        selected: list[tuple[str, SourceComponent, SourceOperation]] = [
            ("runner-run-agent", components[0], runner_op),
            ("resources-add-agent", components[1], reg_op),  # already there
        ]
        skill = SkillSpec(
            name="core",
            description="Core APIs",
            allowed_tools=["runner-run-agent", "resources-add-agent"],
        )
        result = _ensure_registry_chain_entries(selected, skill, index)
        # Count occurrences
        add_agent_count = sum(1 for ref, _comp, _op in result if ref == "resources-add-agent")
        self.assertEqual(add_agent_count, 1, "add-agent should not be duplicated")

    def test_extract_noun_edge_cases(self) -> None:
        """_extract_noun handles edge cases correctly."""
        from extensions.sop_converter.task_guide import (
            _ORCHESTRATION_NAME_PREFIXES,
            _REGISTRY_NAME_PREFIXES,
        )

        # Standard cases
        self.assertEqual(_extract_noun("run_agent", _ORCHESTRATION_NAME_PREFIXES), "agent")
        self.assertEqual(_extract_noun("add_agent", _REGISTRY_NAME_PREFIXES), "agent")
        self.assertEqual(
            _extract_noun("register_workflow", _REGISTRY_NAME_PREFIXES),
            "workflow",
        )

        # No match
        self.assertIsNone(_extract_noun("configure", _ORCHESTRATION_NAME_PREFIXES))
        self.assertIsNone(_extract_noun("run_", _ORCHESTRATION_NAME_PREFIXES))

        # build_ is not in _ORCHESTRATION_NAME_PREFIXES
        self.assertIsNone(_extract_noun("build_agent", _ORCHESTRATION_NAME_PREFIXES))

    def test_end_to_end_task_guide_includes_registry(self) -> None:
        """Full generate_task_guide_markdown injects registry when runner present."""
        runner_op = self._make_op(
            "run_agent",
            "Execute a single agent by ID.",
            file_stem="runner",
        )
        reg_op = self._make_op(
            "add_agent",
            "Register a built agent with the resource manager.",
            file_stem="manager",
        )
        components = [
            self._make_comp("runner", [runner_op]),
            self._make_comp("resources", [reg_op]),
        ]
        skill = SkillSpec(
            name="core_merged",
            description="Core domain",
            allowed_tools=["runner-run-agent", "resources-add-agent"],
        )
        guide = generate_task_guide_markdown(skill, components)
        self.assertIn("## 任务指南", guide)
        self.assertIn("runner-run-agent", guide)
        self.assertIn("resources-add-agent", guide)


class TestMacroRouteTaskGuideRows(unittest.TestCase):
    """Bundle MacroRoute rows appear in Task Guide when allowlisted."""

    def test_macro_rows_from_bundle_macros(self) -> None:
        import tempfile
        from pathlib import Path

        from extensions.sop_converter.task_guide import _macro_route_task_guide_rows

        with tempfile.TemporaryDirectory() as tmp:
            macros_dir = Path(tmp) / ".clawcodex" / "macros"
            macros_dir.mkdir(parents=True)
            (macros_dir / "text-processing-pipeline.yaml").write_text(
                """
version: 1
name: text-processing-pipeline
enabled: true
workflow:
  inputs: {}
  steps:
    - id: s1
      kind: tool
      callable_ref: skills-skill-handlers-execute-pipeline
      args: {}
  outputs: {}
routing:
  phrases:
    - 处理文本数据
    - 手写宏
    - 文本处理宏
  target_tool: text-processing-pipeline
  covered_tools:
    - skills-skill-handlers-execute-pipeline
""",
                encoding="utf-8",
            )
            skill = SkillSpec(
                name="skill_handlers",
                description="handlers",
                allowed_tools=[
                    "text-processing-pipeline",
                    "skills-skill-handlers-execute-pipeline",
                ],
            )
            # Minimal component so generate_task_guide_markdown does not early-exit.
            op = SourceOperation(
                name="execute_pipeline",
                description="执行完整 pipeline 并返回结构化摘要",
                file_stem="execution",
                has_docstring=True,
            )
            components = [
                SourceComponent(
                    name="skill_handlers",
                    file_path="x.py",
                    description="handlers",
                    operations=[op],
                )
            ]
            rows = _macro_route_task_guide_rows(skill, tmp)
            self.assertEqual(len(rows), 1)
            intent, tool, search, note = rows[0]
            self.assertEqual(tool, "text-processing-pipeline")
            self.assertIn("手写宏", intent)
            self.assertIn("select:text-processing-pipeline", search)
            self.assertIn("禁止用 Bash", note)
            self.assertNotIn("勿裸调", note)

            guide = generate_task_guide_markdown(skill, components, bundle=tmp)
            self.assertIn("text-processing-pipeline", guide)
            self.assertIn("禁止**先用 Bash", guide)
            self.assertIn("select:text-processing-pipeline", guide)

    def test_macro_rows_skipped_when_not_allowlisted(self) -> None:
        import tempfile
        from pathlib import Path

        from extensions.sop_converter.task_guide import _macro_route_task_guide_rows

        with tempfile.TemporaryDirectory() as tmp:
            macros_dir = Path(tmp) / ".clawcodex" / "macros"
            macros_dir.mkdir(parents=True)
            (macros_dir / "text-processing-pipeline.yaml").write_text(
                """
version: 1
name: text-processing-pipeline
enabled: true
workflow:
  inputs: {}
  steps:
    - id: s1
      kind: tool
      callable_ref: skills-skill-handlers-execute-pipeline
      args: {}
  outputs: {}
routing:
  phrases: [处理文本数据]
  target_tool: text-processing-pipeline
""",
                encoding="utf-8",
            )
            skill = SkillSpec(
                name="other",
                description="d",
                allowed_tools=["skills-skill-handlers-list-operations"],
            )
            self.assertEqual(_macro_route_task_guide_rows(skill, tmp), [])


if __name__ == "__main__":
    unittest.main()
