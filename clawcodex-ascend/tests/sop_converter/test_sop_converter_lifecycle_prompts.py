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

"""Unit tests for the L3 lifecycle prompt block + task guide rows.

Covers:

* ``_lifecycle_prompt_block`` returns empty for ``None`` / missing yaml /
    empty graph
* ``_lifecycle_prompt_block`` renders the dependency table when yaml exists
* ``domain_agent_sop_body`` includes the lifecycle block when bundle has deps
* ``generate_task_guide_markdown`` appends lifecycle rows
* ``append_task_guide_to_skill_body`` threads the ``bundle`` argument
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from extensions.sop_converter.dependency import (
    HiddenStep,
    ToolDependency,
    ToolDependencyGraph,
    write_tool_dependencies,
)
from extensions.sop_converter.skill_grouper import SkillSpec
from extensions.sop_converter.sop_prompts import (
    _lifecycle_prompt_block,
    domain_agent_sop_body,
)
from extensions.sop_converter.source_parser import SourceComponent, SourceOperation
from extensions.sop_converter.task_guide import (
    _lifecycle_task_guide_rows,
    append_task_guide_to_skill_body,
    format_flat_skill_markdown,
    generate_task_guide_markdown,
)


def _comp(name: str = "Comp", ops: list[SourceOperation] | None = None) -> SourceComponent:
    return SourceComponent(name=name, file_path="x.py", description="d", operations=ops or [])


def _op(name: str) -> SourceOperation:
    return SourceOperation(name=name, description="d", class_name="C", file_stem="x")


class TestLifecyclePromptBlock(unittest.TestCase):
    """``_lifecycle_prompt_block`` filtering and rendering."""

    def test_none_bundle_returns_empty(self) -> None:
        self.assertEqual(_lifecycle_prompt_block(None), "")

    def test_missing_yaml_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_lifecycle_prompt_block(tmp), "")

    def test_empty_graph_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_tool_dependencies(ToolDependencyGraph(), Path(tmp) / ".clawcodex" / "tool-dependencies.yaml")
            self.assertEqual(_lifecycle_prompt_block(tmp), "")

    def test_graph_renders_table_and_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph = ToolDependencyGraph(
                dependencies=[
                    ToolDependency(
                        from_tool="comp.build-agent",
                        to_tool="comp.run-agent",
                        shared_params=["agent_id"],
                        hidden_steps=[HiddenStep(action="persist_agent_catalog", description="保存映射")],
                        lifecycle="create → invoke",
                    )
                ]
            )
            write_tool_dependencies(graph, Path(tmp) / ".clawcodex" / "tool-dependencies.yaml")
            block = _lifecycle_prompt_block(tmp)
            self.assertIn("## 工具生命周期提示", block)
            self.assertIn("comp.build-agent", block)
            self.assertIn("comp.run-agent", block)
            self.assertIn("agent_id", block)
            self.assertIn("not found", block)


class TestDomainAgentSopBodyLifecycle(unittest.TestCase):
    """``domain_agent_sop_body`` integrates the lifecycle block."""

    def test_no_bundle_excludes_block(self) -> None:
        body = domain_agent_sop_body(
            agent_type="Test",
            description="d",
            skill_name="test-skill",
        )
        self.assertNotIn("工具生命周期提示", body)

    def test_bundle_with_yaml_includes_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph = ToolDependencyGraph(
                dependencies=[
                    ToolDependency(
                        from_tool="comp.build-agent",
                        to_tool="comp.run-agent",
                        shared_params=["agent_id"],
                        lifecycle="create → invoke",
                    )
                ]
            )
            write_tool_dependencies(graph, Path(tmp) / ".clawcodex" / "tool-dependencies.yaml")
            body = domain_agent_sop_body(
                agent_type="Test",
                description="d",
                skill_name="test-skill",
                bundle=tmp,
            )
            self.assertIn("## 工具生命周期提示", body)
            self.assertIn("comp.build-agent", body)


class TestLifecycleTaskGuideRows(unittest.TestCase):
    """``_lifecycle_task_guide_rows`` shape."""

    def test_none_graph_returns_empty(self) -> None:
        self.assertEqual(_lifecycle_task_guide_rows(None), [])

    def test_empty_graph_returns_empty(self) -> None:
        self.assertEqual(_lifecycle_task_guide_rows(ToolDependencyGraph()), [])

    def test_dependency_emits_forward_and_backward_rows(self) -> None:
        graph = ToolDependencyGraph(
            dependencies=[
                ToolDependency(
                    from_tool="a",
                    to_tool="b",
                    shared_params=["id"],
                    hidden_steps=[HiddenStep("s1", "d1")],
                    lifecycle="create → invoke",
                )
            ]
        )
        rows = _lifecycle_task_guide_rows(graph)
        self.assertEqual(len(rows), 2)
        intent1, tool1, search1, note1 = rows[0]
        self.assertEqual(tool1, "a")
        self.assertEqual(search1, "select:a")
        self.assertIn("前置步骤", note1)
        intent2, tool2, search2, note2 = rows[1]
        self.assertEqual(tool2, "b")
        self.assertEqual(search2, "select:a")
        self.assertIn("先调用", note2)


class TestGenerateTaskGuideLifecycle(unittest.TestCase):
    """``generate_task_guide_markdown`` includes dependency rows."""

    def test_lifecycle_rows_inserted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph = ToolDependencyGraph(
                dependencies=[
                    ToolDependency(
                        from_tool="comp.build-agent",
                        to_tool="comp.run-agent",
                        shared_params=["agent_id"],
                        lifecycle="create → invoke",
                    )
                ]
            )
            write_tool_dependencies(graph, Path(tmp) / ".clawcodex" / "tool-dependencies.yaml")
            skill = SkillSpec(
                name="test",
                description="d",
                allowed_tools=["comp.build-agent", "comp.run-agent"],
            )
            # Make one op appear as entry point
            ops = [
                SourceOperation(
                    name="build_agent",
                    description="Create an agent.",
                    class_name="Builder",
                    file_stem="builder",
                ),
                SourceOperation(
                    name="run_agent",
                    description="Run an agent.",
                    class_name="Runner",
                    file_stem="runner",
                    parameters=[],
                ),
            ]
            components = [_comp("comp", ops)]
            md = generate_task_guide_markdown(skill, components, bundle=tmp)
            self.assertIn("## 任务指南", md)
            self.assertIn("调用 `comp.run-agent` 之前", md)
            self.assertIn("select:comp.build-agent", md)

    def test_verified_macro_removes_conflicting_atomic_lifecycle_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            atomic = "openjiuwen-core-controller-legacy-basecontroller-send-to-agent"
            graph = ToolDependencyGraph(
                dependencies=[
                    ToolDependency(
                        from_tool="create-agent",
                        to_tool=atomic,
                        shared_params=["agent_id"],
                        lifecycle="create → invoke",
                    )
                ]
            )
            write_tool_dependencies(graph, Path(tmp) / ".clawcodex" / "tool-dependencies.yaml")
            skill = SkillSpec(
                name="test",
                description="d",
                allowed_tools=["invoke-existing-agent", "create-agent", atomic],
            )
            components = [
                _comp(
                    "comp",
                    [
                        SourceOperation(
                            name="create_agent",
                            description="Create an agent.",
                            class_name="Builder",
                            file_stem="builder",
                        )
                    ],
                )
            ]
            md = generate_task_guide_markdown(skill, components, bundle=tmp)
            self.assertIn("select:invoke-existing-agent", md)
            self.assertNotIn(f"调用 `{atomic}` 之前", md)
            self.assertNotIn(f"`{atomic}` 返回 not found", md)


class TestAppendTaskGuideBundlesLifecycle(unittest.TestCase):
    """``append_task_guide_to_skill_body`` threads ``bundle``."""

    def test_append_threads_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph = ToolDependencyGraph(
                dependencies=[
                    ToolDependency(
                        from_tool="a",
                        to_tool="b",
                        shared_params=["id"],
                        lifecycle="create → invoke",
                    )
                ]
            )
            write_tool_dependencies(graph, Path(tmp) / ".clawcodex" / "tool-dependencies.yaml")
            skill = SkillSpec(
                name="test",
                description="d",
                allowed_tools=["a", "b"],
            )
            body = "# Skill\n\ndescription"
            out = append_task_guide_to_skill_body(body, skill, [_comp("comp")], bundle=tmp)
            self.assertIn("## 任务指南", out)


class TestLifecycleFrontmatter(unittest.TestCase):
    def test_flat_skill_frontmatter_includes_lifecycle_deps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph = ToolDependencyGraph(
                dependencies=[
                    ToolDependency(
                        from_tool="a",
                        to_tool="b",
                        shared_params=["id"],
                        lifecycle="create -> invoke",
                    )
                ]
            )
            write_tool_dependencies(graph, Path(tmp) / ".clawcodex" / "tool-dependencies.yaml")
            skill = SkillSpec(
                name="test",
                description="d",
                allowed_tools=["a", "b"],
            )
            md = format_flat_skill_markdown(skill, bundle=tmp)
            self.assertIn("lifecycle-deps: .clawcodex/tool-dependencies.yaml", md)


if __name__ == "__main__":
    unittest.main()
