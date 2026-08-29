# ruff: noqa
# pylint: skip-file
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

"""Tests for the SOP Converter source parser, skill grouper, and agent markdown writer."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import ClassVar

import pytest

from extensions.sop_converter.source_parser import (
    SourceCodeParser,
    SourceComponent,
    SourceOperation,
    ParamSpec,
    infer_type_hint_from_description,
)
from extensions.sop_converter.skill_grouper import (
    GroupStrategy,
    SkillGrouper,
    group_source_components,
    SkillSpec,
    MappingRule,
    MatchType,
    MatchTarget,
)
from extensions.sop_converter.agent_md_writer import (
    AgentMarkdownWriter,
    AgentComponentInfo,
    WorkflowStage,
)
from extensions.sop_converter.default_agent import (
    resolve_default_agent,
    resolve_agent_by_type,
    _parse_frontmatter,
)
from extensions.sop_converter.agent_builder import AgentBuilder
from extensions.sop_converter.core.templates import (
    AGENT_MD_TEMPLATE,
    SKILL_MD_TEMPLATE_JINJA,
    OVERVIEW_AGENT_TEMPLATE,
)


# =========================================================================
# SourceCodeParser tests
# =========================================================================


class TestParamSpec:
    def test_default_required(self) -> None:
        p = ParamSpec(name="x")
        assert p.name == "x"
        assert p.required is True
        assert p.type_hint is None
        assert p.default is None

    def test_optional_param(self) -> None:
        p = ParamSpec(name="y", type_hint="str", default="hello", required=False)
        assert p.name == "y"
        assert p.type_hint == "str"
        assert p.default == "hello"
        assert p.required is False


class TestSourceOperation:
    def test_minimal(self) -> None:
        op = SourceOperation(name="do_stuff", description="Does stuff")
        assert op.name == "do_stuff"
        assert op.parameters == []
        assert op.return_type is None

    def test_full(self) -> None:
        params = [ParamSpec(name="x", type_hint="int")]
        op = SourceOperation(
            name="add",
            description="Add numbers",
            parameters=params,
            return_type="int",
            source_code="def add(x): pass",
        )
        assert op.name == "add"
        assert len(op.parameters) == 1
        assert op.return_type == "int"


class TestSourceComponent:
    def test_minimal(self) -> None:
        comp = SourceComponent(
            name="MathOps",
            file_path="math/ops.py",
            description="Math operations",
        )
        assert comp.name == "MathOps"
        assert comp.operations == []
        assert comp.dependencies == []
        assert comp.input_schema == {}

    def test_with_ops(self) -> None:
        ops = [SourceOperation(name="add", description="Add")]
        comp = SourceComponent(
            name="MathOps",
            file_path="math.py",
            description="Math",
            operations=ops,
            dependencies=["math_utils"],
        )
        assert len(comp.operations) == 1
        assert "math_utils" in comp.dependencies


class TestSourceCodeParser:
    """Tests for SourceCodeParser with sample Python source files."""

    def test_parse_single_class(self) -> None:
        """Parse a single Python file with a class and methods."""
        source = '''
class VideoProcessor:
    """Process video files with various operations."""

    def transcode(self, input_path: str, output_format: str = "mp4") -> bool:
        """Transcode a video file to the specified format.

        Args:
            input_path: Path to the input video file.
            output_format: Target output format (default: mp4).

        Returns:
            True if successful, False otherwise.
        """
        return True

    def get_metadata(self, file_path: str) -> dict:
        """Get video file metadata."""
        return {}
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            py_file = tmp / "video_processor.py"
            py_file.write_text(source)

            parser = SourceCodeParser(tmp)
            components = parser.parse()

        assert len(components) >= 1
        # Find our component
        comp = next(c for c in components if c.name == tmp.name)
        assert len(comp.operations) >= 1

        # Check transcode method
        transcode = next((op for op in comp.operations if op.name == "transcode"), None)
        assert transcode is not None, f"transcode not found in {[op.name for op in comp.operations]}"
        assert "transcode" in transcode.description.lower()
        assert transcode.return_type == "bool"

        # Check parameters
        assert len(transcode.parameters) >= 1
        param_names = {p.name for p in transcode.parameters}
        assert "input_path" in param_names
        assert "output_format" in param_names

        # Check type hints
        input_param = next(p for p in transcode.parameters if p.name == "input_path")
        assert "str" in (input_param.type_hint or "")

    def test_parse_top_level_functions(self) -> None:
        """Parse a Python file with module-level functions."""
        source = '''
"""Utility functions for data processing."""

import json
import os


def load_config(path: str) -> dict:
    """Load configuration from a JSON file.

    Args:
        path: Path to the config file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path) as f:
        return json.load(f)


def save_result(data: dict, output_path: str) -> None:
    """Save results to a JSON file.

    Args:
        data: The data to save.
        output_path: Path to save the file.
    """
    with open(output_path, "w") as f:
        json.dump(data, f)
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            py_file = tmp / "utils.py"
            py_file.write_text(source)

            parser = SourceCodeParser(tmp)
            components = parser.parse()

        assert len(components) >= 1
        comp = next(c for c in components if c.name == tmp.name)
        op_names = {op.name for op in comp.operations}
        assert "load_config" in op_names, f"load_config not in {op_names}"
        assert "save_result" in op_names, f"save_result not in {op_names}"

    def test_exclude_patterns(self) -> None:
        """Test that exclude_patterns filters out unwanted files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create a normal file
            (tmp / "normal.py").write_text("def foo(): pass\n")
            # Create an excluded file
            (tmp / "test_normal.py").write_text("def test_foo(): pass\n")

            parser = SourceCodeParser(tmp, exclude_patterns=["test_*"], extern_only=False)
            components = parser.parse()

            # Should find the normal file while excluding the test file.
            comp = next(c for c in components if c.name == tmp.name)
            op_names = {op.name for op in comp.operations}
            assert "foo" in op_names
            assert "test_foo" not in op_names

    def test_default_exclude_test_and_example_dirs(self) -> None:
        """Default exclude patterns skip *test* and *example* directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Normal source directory — should be parsed.
            src = tmp / "src"
            src.mkdir()
            (src / "mod.py").write_text('''
def api_func(x: int) -> str:
    """Public API function.

    Args:
        x: Input value.

    Returns:
        String result.
    """
    return str(x)
''')

            # Test directory — should be skipped by default *test* pattern.
            test_dir = tmp / "unit_tests"
            test_dir.mkdir()
            (test_dir / "test_lib.py").write_text('''
def test_api_func() -> None:
    """Test api_func."""
    pass
''')

            # Examples directory — should be skipped by default *example* pattern.
            example_dir = tmp / "examples"
            example_dir.mkdir()
            (example_dir / "demo.py").write_text('''
def demo_api() -> str:
    """Demo usage of the API.

    Returns:
        Demo string.
    """
    return "demo"
''')

            parser = SourceCodeParser(tmp)
            components = parser.parse()

        comp_names = {c.name for c in components}
        assert "src" in comp_names, f"Expected 'src', got: {comp_names}"
        assert "unit_tests" not in comp_names, (
            f"'unit_tests' directory should be excluded by *test* pattern, got: {comp_names}"
        )
        assert "examples" not in comp_names, (
            f"'examples' directory should be excluded by *example* pattern, got: {comp_names}"
        )

    def test_empty_directory(self) -> None:
        """Parse an empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = SourceCodeParser(tmpdir)
            components = parser.parse()
            assert len(components) == 0

    def test_parse_file_single(self) -> None:
        """Test parse_file() for a single file."""
        source = '''
def greet(name: str) -> str:
    """Greet someone.

    Args:
        name: The person's name.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            py_file = tmp / "greeter.py"
            py_file.write_text(source)

            parser = SourceCodeParser(tmp)
            operations = parser.parse_file(py_file)

        assert len(operations) == 1
        assert operations[0].name == "greet"
        assert operations[0].return_type == "str"


class TestDocstringParsing:
    """Test docstring parsing in various formats."""

    def test_google_style(self) -> None:
        source = '''
def func(a: int, b: str) -> bool:
    """Do something.

    Args:
        a: An integer value.
        b: A string value.

    Returns:
        True on success.
    """
    return True
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source)
            parser = SourceCodeParser(tmp)
            comps = parser.parse()
            ops = [op for comp in comps for op in comp.operations]
            op = next((o for o in ops if o.name == "func"), None)
            assert op is not None
            assert "Do something" in op.description

    def test_numpy_style(self) -> None:
        source = '''
def func(x: float) -> float:
    """Compute the square of a number.

    Parameters
    ----------
    x : float
        The input value.

    Returns
    -------
    float
        The square of x.
    """
    return x * x
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source)
            parser = SourceCodeParser(tmp)
            comps = parser.parse()
            ops = [op for comp in comps for op in comp.operations]
            op = next((o for o in ops if o.name == "func"), None)
            assert op is not None
            assert "square" in op.description.lower()

    def test_rest_style(self) -> None:
        source = '''
def func(name: str) -> str:
    """Say hello.

    :param name: The person to greet.
    :type name: str
    :returns: A greeting string.
    """
    return f"Hi {name}"
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source)
            parser = SourceCodeParser(tmp)
            comps = parser.parse()
            ops = [op for comp in comps for op in comp.operations]
            op = next((o for o in ops if o.name == "func"), None)
            assert op is not None
            assert "hello" in op.description.lower()

    def test_chinese_style_params(self) -> None:
        source = '''
def send_email(to_email, content, smtp_config=None, sender_config=None):
    """发送邮件

    参数:
        to_email: 收件人邮箱
        content: 邮件内容配置字典，包含 subject, body, html(可选)
        smtp_config: SMTP服务器配置字典，包含 smtp_server, smtp_port(可选)
        sender_config: 发件人配置字典，包含 from_email, password(可选)
    """
    subject = content.get("subject", "")
    return subject
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source, encoding="utf-8")
            parser = SourceCodeParser(tmp)
            comps = parser.parse()
            ops = [op for comp in comps for op in comp.operations]
            op = next((o for o in ops if o.name == "send_email"), None)
            assert op is not None
            by_name = {p.name: p for p in op.parameters}
            assert by_name["to_email"].type_hint is None
            assert by_name["content"].type_hint == "dict"
            assert by_name["smtp_config"].type_hint == "dict"
            assert by_name["sender_config"].type_hint == "dict"
            assert "subject" in by_name["content"].description


class TestInferTypeHintFromDescription:
    def test_dict_keywords(self) -> None:
        assert infer_type_hint_from_description("邮件内容配置字典，包含 subject, body") == "dict"
        assert infer_type_hint_from_description("JSON mapping of keys") == "dict"

    def test_non_dict_descriptions(self) -> None:
        assert infer_type_hint_from_description("收件人邮箱") is None
        assert infer_type_hint_from_description("SMTP port number") is None
        assert infer_type_hint_from_description("") is None


class TestAgentMarkdownWriter:
    def test_write_agent(self) -> None:
        """Write a single agent markdown file and verify frontmatter."""
        writer = AgentMarkdownWriter()
        agent_def = {
            "name": "test-helper",
            "description": "A test agent",
            "model": "claude-4",
            "tools": ["tool_a", "tool_b"],
            "skills": ["skill_x"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            path = writer.write_agent(agent_def, output_dir)

            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "name: test-helper" in content
            assert "description: 'A test agent'" in content
            assert "model: claude-4" in content
            assert "tool_a" in content
            assert "skill_x" in content

    def test_write_skills(self) -> None:
        """Write skill markdown files with parameters."""
        writer = AgentMarkdownWriter()
        skills = [
            {
                "name": "transcode-video",
                "description": "Transcode video to target format",
                "allowed_tools": ["transcode"],
                "parameters": [
                    {
                        "name": "input_path",
                        "type_hint": "str",
                        "required": True,
                        "description": "Input file",
                    }
                ],
                "source_code": "def transcode(path): pass",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paths = writer.write_skills(skills, output_dir)

            assert len(paths) >= 1
            skill_path = paths[0]
            assert skill_path.exists()
            content = skill_path.read_text(encoding="utf-8")
            assert "transcode-video" in content
            assert "input_path" in content

    def test_write_overview_agent(self) -> None:
        """Write overview agent for multi-component project."""
        writer = AgentMarkdownWriter()
        agents = [
            AgentComponentInfo(
                name="video-ops-agent",
                description="Video processing operations",
                capabilities=["transcode", "slice"],
                input_types=["mp4"],
                output_types=["hls"],
                invoke_pattern="@video-ops-agent transcode input.mp4",
            ),
            AgentComponentInfo(
                name="data-process-agent",
                description="Data processing operations",
                capabilities=["filter", "aggregate"],
                input_types=["csv"],
                output_types=["json"],
                invoke_pattern="@data-process-agent process data.csv",
            ),
        ]
        stages = [
            WorkflowStage(
                name="Video Processing",
                order=1,
                description="Process video files",
                responsible_agent="video-ops-agent",
                output_type="HLS segments",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            path = writer.write_overview_agent(
                name="clawcodex-overview",
                description="Overview agent for test",
                component_agents=agents,
                workflow_stages=stages,
                output_dir=output_dir,
            )

            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "clawcodex-overview" in content
            assert "video-ops-agent" in content
            assert "data-process-agent" in content
            assert "Video Processing" in content

    def test_write_workflow(self) -> None:
        """Write WORKFLOW.md for orchestrator."""
        writer = AgentMarkdownWriter()
        agents = [
            AgentComponentInfo(name="agent-a", description="Agent A", capabilities=["a"]),
        ]
        stages = [
            WorkflowStage(name="Stage 1", order=1, responsible_agent="agent-a", output_type="result"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = writer.write_workflow("test", "Test workflow", agents, stages, Path(tmpdir))
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "WORKFLOW.md" in content or "test" in content


# =========================================================================
# Default Agent tests
# =========================================================================


class TestResolveDefaultAgent:
    def test_no_overview_file(self) -> None:
        """No overview file → return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = resolve_default_agent(tmpdir)
            assert result is None

    def test_with_overview_file(self) -> None:
        """Overview file exists → return parsed dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / ".claude" / "agents"
            agents_dir.mkdir(parents=True)
            overview = agents_dir / "clawcodex-overview.md"
            overview.write_text("""\
---
name: clawcodex-overview
description: Overview agent
model: claude-4
tools:
  - "*"
skills:
  - skill-a
---

# Overview Agent

This is the overview agent.
""")
            result = resolve_default_agent(tmpdir)
            assert result is not None
            assert result["name"] == "clawcodex-overview"
            assert result["description"] == "Overview agent"
            assert result["model"] == "claude-4"

    def test_resolve_agent_by_type(self) -> None:
        """Find an agent by its frontmatter name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / ".claude" / "agents"
            agents_dir.mkdir(parents=True)
            (agents_dir / "my-agent.md").write_text("""\
---
name: my-agent
description: My custom agent
---

# My Agent

Custom agent body.
""")
            result = resolve_agent_by_type(tmpdir, "my-agent")
            assert result is not None
            assert result["name"] == "my-agent"
            assert "custom agent" in result.get("description", "").lower()


class TestParseFrontmatter:
    def test_simple_frontmatter(self) -> None:
        content = """\
---
name: test
description: Test
tools:
  - tool_a
  - tool_b
---

Body text
"""
        fm, body = _parse_frontmatter(content)
        assert fm["name"] == "test"
        assert fm["description"] == "Test"
        assert "tool_a" in fm.get("tools", [])
        assert "Body text" in body

    def test_no_frontmatter(self) -> None:
        content = "Just some text\nNo frontmatter here."
        fm, body = _parse_frontmatter(content)
        assert fm == {}
        assert "Just some text" in body


# =========================================================================
# AgentBuilder tests
# =========================================================================


class TestAgentBuilder:
    def test_build_agent_definition_format(self) -> None:
        """Default format='agent_definition' still works."""
        skills = [SkillSpec(name="test-skill", description="Test", allowed_tools=["tool_a"])]
        builder = AgentBuilder(
            skills=skills,
            agent_name="test-agent",
            agent_description="A test agent",
        )
        result = builder.build()
        assert result.agent.agent_type == "test-agent"
        assert "tool_a" in (result.agent.tools or [])

    def test_build_with_markdown_format(self) -> None:
        """format='markdown' writes agent markdown files."""
        skills = [SkillSpec(name="test-skill", description="Test", allowed_tools=["tool_a"])]
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = AgentBuilder(
                skills=skills,
                agent_name="test-agent",
                agent_description="A test agent",
                output_dir=tmpdir,
            )
            result = builder.build(format="markdown")
            assert result.markdown_files is not None

    def test_invalid_format_raises(self) -> None:
        """Invalid format string raises ValueError."""
        builder = AgentBuilder(
            skills=[],
            agent_name="test",
            agent_description="test",
        )
        with pytest.raises(ValueError):
            builder.build(format="invalid")

    def test_overview_built_from_skills_not_components(self) -> None:
        """Overview agent reflects grouped skills, not raw components."""
        ops_a = [SourceOperation(name="op_a", description="Op A")]
        ops_b = [SourceOperation(name="op_b", description="Op B")]
        comp_a = SourceComponent(name="CompA", file_path="a.py", description="Comp A", operations=ops_a)
        comp_b = SourceComponent(name="CompB", file_path="b.py", description="Comp B", operations=ops_b)

        # Simulate IO_RELATION: 2 components merged into 1 skill
        skills = [SkillSpec(name="merged_skill", description="Merged", allowed_tools=["op_a", "op_b"])]
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = AgentBuilder(
                skills=skills,
                agent_name="test-agent",
                agent_description="A test agent",
                source_components=[comp_a, comp_b],
                output_dir=tmpdir,
            )
            result = builder.build(format="markdown")
            # 1 agent + 1 skill file; no overview since only 1 skill
            assert len(result.markdown_files) == 2
            agent_file = next(f for f in result.markdown_files if "test-agent" in f.name)
            agent_content = agent_file.read_text(encoding="utf-8")
            assert "merged_skill" in agent_content

    def test_multi_skill_overview_generation(self) -> None:
        """Multiple grouped skills → overview agent generated."""
        skills = [
            SkillSpec(name="skill_a", description="Skill A", allowed_tools=["tool_a1"]),
            SkillSpec(name="skill_b", description="Skill B", allowed_tools=["tool_b1"]),
        ]
        components = [
            SourceComponent(
                name="CompA",
                file_path="a.py",
                description="A",
                operations=[SourceOperation(name="tool_a1", description="Tool A1")],
            ),
            SourceComponent(
                name="CompB",
                file_path="b.py",
                description="B",
                operations=[SourceOperation(name="tool_b1", description="Tool B1")],
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = AgentBuilder(
                skills=skills,
                agent_name="multi-agent",
                agent_description="Multi-agent",
                source_components=components,
                output_dir=tmpdir,
            )
            result = builder.build(format="both")
            # 2 agent files + 1 overview + skill files
            assert result.markdown_files is not None
            assert len(result.markdown_files) >= 3  # agent + skills + overview
            # Find overview agent
            overview_files = [f for f in result.markdown_files if "clawcodex-overview" in f.name]
            assert len(overview_files) >= 1, f"Expected overview agent, got: {[f.name for f in result.markdown_files]}"
            overview_content = overview_files[0].read_text(encoding="utf-8")
            assert "skill_a-agent" in overview_content
            assert "skill_b-agent" in overview_content


# =========================================================================
# Auto-generate rules naming tests
# =========================================================================


class TestAutoGenerateRulesNaming:
    def _make_components(self, paths: list[str]) -> list[SourceComponent]:
        """Create SourceComponents from a list of file_path strings."""
        comps = []
        for fp in paths:
            name = Path(fp).stem if fp.endswith(".py") else Path(fp).name
            comps.append(
                SourceComponent(
                    name=name,
                    file_path=fp,
                    description=f"Component in {fp}",
                    operations=[SourceOperation(name="do_work", description="Work")],
                )
            )
        return comps

    def test_merged_group_named_by_common_ancestor(self) -> None:
        """Merged group skill_name uses common ancestor, not one leaf's unique tag."""
        from extensions.sop_converter.skill_grouper import SkillGrouper

        comps = self._make_components(
            [
                "proj/examples/permissions/perm_ops.py",
                "proj/examples/rl_calculator/rl_ops.py",
                "proj/examples/session/session_ops.py",
                "proj/core/memory/mem_ops.py",
                "proj/core/runner/runner_ops.py",
            ]
        )

        rules = SkillGrouper._auto_generate_rules(comps, max_groups=4)
        skill_names = {r.skill_name for r in rules}

        # The examples/* paths should be merged into one group named
        # after their common ancestor "examples" + "_merged" suffix.
        # Single-path groups (memory, runner, session) stay un-suffixed.
        assert "examples_merged" in skill_names, f"Expected 'examples_merged' in skill_names, got: {skill_names}"

    def test_merged_group_no_common_ancestor_fallback(self) -> None:
        """When sub_keys share no common segment, fallback to distinguishing pattern."""
        from extensions.sop_converter.skill_grouper import SkillGrouper

        comps = self._make_components(
            [
                "proj/a_domain/ops.py",
                "proj/b_domain/ops.py",
                "proj/c_domain/ops.py",
                "proj/d_domain/ops.py",
            ]
        )

        rules = SkillGrouper._auto_generate_rules(comps, max_groups=2)

        # These 4 groups get merged into 2. The first merged group may
        # have sub_keys like "proj/a_domain|proj/b_domain" — their common
        # ancestor is "proj" which is also shared by all other groups.
        # Fallback should use a distinguishing segment.
        skill_names = {r.skill_name for r in rules}
        assert len(skill_names) <= 2

    def test_common_ancestor_segment_basic(self) -> None:
        """_common_ancestor_segment finds the deepest shared segment."""
        from extensions.sop_converter.skill_grouper import _common_ancestor_segment

        sub_keys = ["proj/examples/permissions", "proj/examples/rl_calc", "proj/examples/session"]
        other = {"core", "memory", "runner", "harness", "proj"}
        result = _common_ancestor_segment(sub_keys, other)
        assert result == "examples", f"Expected 'examples', got: {result}"

    def test_common_ancestor_segment_unique_to_group(self) -> None:
        """_common_ancestor_segment prefers segments not shared with other groups."""
        from extensions.sop_converter.skill_grouper import _common_ancestor_segment

        sub_keys = ["proj/examples/permissions", "proj/examples/rl_calc"]
        # "examples" is NOT in other → should be picked even though it's shallow
        other = {"core", "memory", "runner"}
        result = _common_ancestor_segment(sub_keys, other)
        assert result == "examples"

    def test_common_ancestor_segment_all_shared_with_others(self) -> None:
        """When all common segments are in other_segments, fallback to deepest common."""
        from extensions.sop_converter.skill_grouper import _common_ancestor_segment

        sub_keys = ["proj/examples/perm", "proj/examples/rl"]
        # "proj" and "examples" are both in other_segments
        other = {"proj", "examples", "core"}
        result = _common_ancestor_segment(sub_keys, other)
        # Should return the deepest common segment ("examples") as fallback
        assert result == "examples"

    def test_common_ancestor_no_common_segments(self) -> None:
        """When sub_keys share no segments, returns None."""
        from extensions.sop_converter.skill_grouper import _common_ancestor_segment

        sub_keys = ["alpha/ops", "beta/ops"]
        other = set()
        result = _common_ancestor_segment(sub_keys, other)
        # Only "ops" is common, but it's a very generic leaf name
        assert result == "ops" or result is None

    def test_single_path_group_not_affected(self) -> None:
        """Single-path groups still use _best_distinguishing_pattern (unchanged)."""
        from extensions.sop_converter.skill_grouper import SkillGrouper

        comps = self._make_components(
            [
                "proj/core/memory/mem_ops.py",
            ]
        )

        rules = SkillGrouper._auto_generate_rules(comps, max_groups=10)
        # Single path group — should use distinguishing pattern
        assert len(rules) == 1
        # With only one group, "proj" is the most distinguishing segment
        # (it's unique since no other groups exist)
        assert rules[0].skill_name in {"proj", "memory", "mem_ops"}


# =========================================================================
# Template string tests
# =========================================================================


class TestTemplates:
    def test_agent_md_template_has_required_fields(self) -> None:
        assert "name:" in AGENT_MD_TEMPLATE
        assert "description:" in AGENT_MD_TEMPLATE
        assert "tools:" in AGENT_MD_TEMPLATE

    def test_skill_md_template_has_required_fields(self) -> None:
        assert "allowed-tools:" in SKILL_MD_TEMPLATE_JINJA
        assert "user-invocable:" in SKILL_MD_TEMPLATE_JINJA

    def test_overview_agent_template_has_required_fields(self) -> None:
        assert "总览 Agent" in OVERVIEW_AGENT_TEMPLATE
        assert "component_agents" in OVERVIEW_AGENT_TEMPLATE
        assert "workflow_stages" in OVERVIEW_AGENT_TEMPLATE


# =========================================================================
# LLM_SEMANTIC strategy tests (mock-based, no real LLM call)
# =========================================================================


class TestLLMSemanticStrategy:
    def test_fallback_without_provider(self) -> None:
        """Without an LLM provider, LLM_SEMANTIC falls back to KEYWORD_MATCH."""
        ops = [
            SourceOperation(
                name="encode",
                description="Encode video",
                parameters=[ParamSpec(name="data", type_hint="str")],
            ),
            SourceOperation(
                name="decode",
                description="Decode video",
                parameters=[ParamSpec(name="data", type_hint="bytes")],
            ),
        ]
        comp = SourceComponent(name="VideoOps", file_path="v.py", description="Video operations", operations=ops)

        result = group_source_components(
            [comp],
            strategy=GroupStrategy.LLM_SEMANTIC,
            llm_provider=None,
        )
        assert len(result.skills) > 0
        # Fallback is KEYWORD_MATCH, not IO_RELATION
        assert not result.skills[0].name.startswith("io_group_")

    def test_parse_valid_json_response(self) -> None:
        """_parse_llm_patterns correctly parses a patterns-based JSON response."""
        ops = [
            SourceOperation(
                name="encode",
                description="Encode video",
                parameters=[ParamSpec(name="data", type_hint="str")],
            ),
            SourceOperation(
                name="decode",
                description="Decode video",
                parameters=[ParamSpec(name="data", type_hint="bytes")],
            ),
            SourceOperation(
                name="load",
                description="Load config",
                parameters=[ParamSpec(name="path", type_hint="Path")],
            ),
        ]
        comp = SourceComponent(
            name="VideoOps",
            file_path="video/video_ops.py",
            description="Video operations",
            operations=ops,
        )

        grouper = SkillGrouper(
            methods=[],
            strategy=GroupStrategy.LLM_SEMANTIC,
            source_components=[comp],
            llm_provider=None,
        )

        raw_json = (
            '{"skills": ['
            '{"name": "video_processing", "description": "Video codec", '
            '"patterns": ["video"]}, '
            '{"name": "config_management", "description": "Config loading", '
            '"patterns": ["config"]}]}'
        )
        dir_paths = ["video/video_ops.py", "config/loader.py"]
        rules = grouper._parse_llm_patterns(raw_json, dir_paths)
        assert rules is not None
        assert len(rules) == 2
        skill_names = {r.skill_name for r in rules}
        assert "video_processing" in skill_names
        assert "config_management" in skill_names
        # Patterns become FILE_PATH / SUBSTRING rules.
        for r in rules:
            assert r.match_target == MatchTarget.FILE_PATH
            assert r.match_type == MatchType.SUBSTRING

    def test_parse_json_with_preamble(self) -> None:
        """_extract_json_from_raw tolerates LLM preamble text."""
        raw = 'Here is the grouping result:\n{"skills": [{"name": "a", "description": "b", "tools": []}]}\nHope this helps!'
        result = SkillGrouper._extract_json_from_raw(raw)
        assert result is not None
        import json

        parsed = json.loads(result)
        assert "skills" in parsed

    def test_extract_json_ignores_braces_inside_strings(self) -> None:
        """Braces inside JSON string values must not close the object early."""
        import json

        # A '}' inside a string value would previously zero the depth counter
        # and truncate the candidate to invalid JSON; the parser must keep
        # scanning until the real closing brace.
        raw = (
            'Here is the grouping:\n'
            '{"skills": [{"name": "a", "description": "Use } for closing", '
            '"patterns": ["some}path", "core"]}]}\nDone.'
        )
        result = SkillGrouper._extract_json_from_raw(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["skills"][0]["patterns"] == ["some}path", "core"]

    def test_extract_json_ignores_braces_inside_escaped_strings(self) -> None:
        """Escaped quotes inside string values keep the string boundary intact."""
        import json

        raw = '{"skills": [{"name": "a", "description": "say \\"hi {there}\\"", "patterns": ["core"]}]}'
        result = SkillGrouper._extract_json_from_raw(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["skills"][0]["name"] == "a"

    def test_orphaned_tools_assigned_to_largest(self) -> None:
        """Unmatched dirs go through auto prefix inference in _keyword_match_group."""
        ops = [
            SourceOperation(name="op_a", description="A", parameters=[ParamSpec(name="x", type_hint="str")]),
            SourceOperation(name="op_b", description="B", parameters=[ParamSpec(name="x", type_hint="int")]),
            SourceOperation(name="op_c", description="C", parameters=[ParamSpec(name="x", type_hint="bool")]),
        ]
        comp_a = SourceComponent(
            name="CompA", file_path="core/module_a.py", description="Module A", operations=[ops[0]]
        )
        comp_b = SourceComponent(
            name="CompB", file_path="core/module_b.py", description="Module B", operations=[ops[1]]
        )
        comp_c = SourceComponent(
            name="CompC", file_path="extra/module_c.py", description="Module C", operations=[ops[2]]
        )

        grouper = SkillGrouper(
            methods=[],
            strategy=GroupStrategy.LLM_SEMANTIC,
            source_components=[comp_a, comp_b, comp_c],
            max_io_groups=10,
            llm_provider=None,
        )

        # LLM only covers "core" — "extra" is unmatched, handled by auto prefix.
        raw_json = '{"skills": [{"name": "core_group", "description": "Core modules", "patterns": ["core"]}]}'
        dir_paths = ["core/module_a.py", "core/module_b.py", "extra/module_c.py"]
        rules = grouper._parse_llm_patterns(raw_json, dir_paths)
        assert rules is not None
        assert len(rules) == 1
        assert rules[0].skill_name == "core_group"

        # Set rules and run keyword_match — unmatched "extra" should go
        # to utility (single-segment prefix without underscore).
        grouper._rules = rules
        skills = grouper._keyword_match_group()
        all_tools = {t for s in skills for t in s.allowed_tools}
        assert "CompA.op_a" in all_tools
        assert "CompB.op_b" in all_tools
        assert "CompC.op_c" in all_tools

    def test_invalid_tools_filtered(self) -> None:
        """LLM patterns that match no dir are dropped (they could never match)."""
        ops = [
            SourceOperation(
                name="real_op",
                description="Real",
                parameters=[ParamSpec(name="x", type_hint="str")],
            ),
        ]
        comp = SourceComponent(name="Comp", file_path="core/real.py", description="Test", operations=ops)

        grouper = SkillGrouper(
            methods=[],
            strategy=GroupStrategy.LLM_SEMANTIC,
            source_components=[comp],
            llm_provider=None,
        )

        # LLM returns a pattern for a dir that exists + one that doesn't.
        # The non-matching pattern is dropped (with a warning) — it could
        # never match any component, so it must not pollute the rule set.
        raw_json = '{"skills": [{"name": "s1", "description": "d1", "patterns": ["core", "nonexistent"]}]}'
        dir_paths = ["core/real.py"]
        rules = grouper._parse_llm_patterns(raw_json, dir_paths)
        assert rules is not None
        assert len(rules) == 1  # only the pattern that hits a real path survives
        assert rules[0].method_pattern == "core"
        skill_names = {r.skill_name for r in rules}
        assert skill_names == {"s1"}

    def test_phantom_patterns_logged_warning(self, caplog) -> None:
        """Skipped LLM patterns surface a warning so degraded grouping is visible."""
        import logging

        comp = SourceComponent(
            name="Comp",
            file_path="core/real.py",
            description="Test",
            operations=[
                SourceOperation(
                    name="real_op",
                    description="Real",
                    parameters=[ParamSpec(name="x", type_hint="str")],
                )
            ],
        )
        grouper = SkillGrouper(
            methods=[],
            strategy=GroupStrategy.LLM_SEMANTIC,
            source_components=[comp],
            llm_provider=None,
        )
        raw_json = '{"skills": [{"name": "s1", "description": "d1", "patterns": ["nonexistent"]}]}'
        with caplog.at_level(logging.WARNING, logger="extensions.sop_converter.runtime.skill_grouper"):
            rules = grouper._parse_llm_patterns(raw_json, ["core/real.py"])
        assert rules is None  # all patterns dropped → no rules
        assert any("matches no source path" in record.message for record in caplog.records)

    def test_max_groups_enforcement(self) -> None:
        """_parse_llm_patterns handles many skills; LLM prompt tells LLM to respect max."""
        ops = []
        for i in range(6):
            ops.append(
                SourceOperation(
                    name=f"op_{i}",
                    description=f"Op {i}",
                    parameters=[ParamSpec(name="x", type_hint="str")],
                )
            )
        comps = []
        for i in range(6):
            comps.append(
                SourceComponent(
                    name=f"Comp{i}",
                    file_path=f"dir_{i}/mod.py",
                    description=f"Module {i}",
                    operations=[ops[i]],
                )
            )

        grouper = SkillGrouper(
            methods=[],
            strategy=GroupStrategy.LLM_SEMANTIC,
            source_components=comps,
            max_io_groups=3,
            llm_provider=None,
        )

        # LLM produces 6 skills with 1 pattern each.
        raw_json = (
            '{"skills": ['
            + ", ".join(f'{{"name": "g{i}", "description": "Group {i}", "patterns": ["dir_{i}"]}}' for i in range(6))
            + "]}"
        )
        dir_paths = [f"dir_{i}/mod.py" for i in range(6)]
        rules = grouper._parse_llm_patterns(raw_json, dir_paths)
        assert rules is not None
        assert len(rules) == 6  # all 6 patterns become rules

    def test_no_source_components_falls_back_to_static(self) -> None:
        """With no source_components, _group_with_llm falls back to _static_group."""
        grouper = SkillGrouper(
            methods=[],
            strategy=GroupStrategy.LLM_SEMANTIC,
            source_components=[],
            llm_provider=None,
        )
        result = grouper._group_with_llm("")
        assert result == grouper._static_group()


# =========================================================================
# --extern-only filtering tests
# =========================================================================


class TestExternOnlyFiltering:
    """Tests for --extern-only flag that filters to docstring-documented API."""

    def test_extern_only_docstring_filter(self) -> None:
        """extern_only=True: only methods with docstrings are included."""
        source = '''
class Processor:
    def documented(self, x: int) -> str:
        """Process the input value.

        Args:
            x: The value to process.

        Returns:
            Processed string.
        """
        return str(x)

    def undocumented(self, y: str) -> None:
        pass
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source)

            parser = SourceCodeParser(tmp, extern_only=True)
            components = parser.parse()

        ops = [op for comp in components for op in comp.operations]
        op_names = {op.name for op in ops}
        assert "documented" in op_names, f"Expected 'documented', got: {op_names}"
        assert "undocumented" not in op_names, f"'undocumented' should be filtered out, got: {op_names}"

    def test_extern_only_all_exports(self) -> None:
        """extern_only=True + __all__: only classes/functions in __all__ pass."""
        source = '''
__all__ = ["Processor"]

class Processor:
    """A document processor."""

    def process(self, data: str) -> str:
        """Process data.

        Args:
            data: Input data.

        Returns:
            Processed data.
        """
        return data

class InternalHelper:
    """Internal helper class."""

    def help(self) -> None:
        """Provide help."""
        pass
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source)

            parser = SourceCodeParser(tmp, extern_only=True)
            components = parser.parse()

        ops = [op for comp in components for op in comp.operations]
        op_names = {op.name for op in ops}
        # Processor is in __all__ and has docstring
        assert "process" in op_names
        # InternalHelper is NOT in __all__
        assert "help" not in op_names, f"'help' from InternalHelper should be filtered out, got: {op_names}"

    def test_extern_only_all_and_docstring(self) -> None:
        """__all__ + extern_only: export with no docstring is still filtered out."""
        source = '''
__all__ = ["Processor", "utility_func"]

class Processor:
    """Processor class."""

    def process(self, data: str) -> str:
        """Process data.

        Args:
            data: Input data.

        Returns:
            Result.
        """
        return data

def utility_func(x: int) -> int:
    """Do something useful.

    Args:
        x: Input value.

    Returns:
        Result.
    """
    return x * 2

def internal_func(y: int) -> int:
    pass
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source)

            parser = SourceCodeParser(tmp, extern_only=True)
            components = parser.parse()

        ops = [op for comp in components for op in comp.operations]
        op_names = {op.name for op in ops}
        # Documented methods/functions in __all__ pass
        assert "process" in op_names
        assert "utility_func" in op_names
        # internal_func has no docstring → filtered out even if in __all__
        assert "internal_func" not in op_names, f"'internal_func' should be filtered (no docstring), got: {op_names}"

    def test_extern_only_no_all(self) -> None:
        """No __all__ defined: filter by docstring only."""
        source = '''
class Worker:
    """Worker class."""

    def do_work(self) -> str:
        """Do some work.

        Returns:
            Result.
        """
        return "done"

    def _internal(self) -> None:
        """Internal helper."""
        pass

    def no_docs(self, x: int) -> int:
        pass

def public_func() -> None:
    """A public function."""
    pass

def hidden_func() -> None:
    pass
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source)

            parser = SourceCodeParser(tmp, extern_only=True)
            components = parser.parse()

        ops = [op for comp in components for op in comp.operations]
        op_names = {op.name for op in ops}
        # has docstring → passes
        assert "do_work" in op_names
        assert "public_func" in op_names
        # no docstring → filtered out
        assert "no_docs" not in op_names
        assert "hidden_func" not in op_names
        # _-prefixed already excluded by existing logic
        assert "_internal" not in op_names

    def test_extern_only_empty_all(self) -> None:
        """Empty __all__ with extern_only=True: no operations pass."""
        source = '''
__all__ = []

class Processor:
    """Processor class."""

    def process(self, data: str) -> str:
        """Process data.

        Args:
            data: Input data.

        Returns:
            Result.
        """
        return data
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source)

            parser = SourceCodeParser(tmp, extern_only=True)
            components = parser.parse()

        ops = [op for comp in components for op in comp.operations]
        assert len(ops) == 0, f"Empty __all__ should yield no ops, got: {[op.name for op in ops]}"

    def test_all_methods_flag(self) -> None:
        """extern_only=False (--all): all public methods included, regardless of docstring."""
        source = '''
class Processor:
    def with_docs(self, x: int) -> str:
        """Documented method.

        Args:
            x: Input.

        Returns:
            Output.
        """
        return str(x)

    def no_docs(self, y: str) -> None:
        pass
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source)

            parser = SourceCodeParser(tmp, extern_only=False)  # --all
            components = parser.parse()

        ops = [op for comp in components for op in comp.operations]
        op_names = {op.name for op in ops}
        # Both should be included when extern_only=False
        assert "with_docs" in op_names
        assert "no_docs" in op_names

    def test_extern_only_top_level_func(self) -> None:
        """Top-level functions: documented ones pass, undocumented filtered out."""
        source = '''
def documented_func(x: int) -> int:
    """Square the input.

    Args:
        x: Input value.

    Returns:
        Squared value.
    """
    return x * x

def undocumented_func(y: str) -> None:
    pass
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source)

            parser = SourceCodeParser(tmp, extern_only=True)
            components = parser.parse()

        ops = [op for comp in components for op in comp.operations]
        op_names = {op.name for op in ops}
        assert "documented_func" in op_names
        assert "undocumented_func" not in op_names

    def test_extern_only_excludes_init(self) -> None:
        """extern_only=True: __init__ is always excluded, even with docstring."""
        source = '''
class Service:
    """A public service class."""

    def __init__(self, endpoint: str, timeout: int = 30) -> None:
        """Initialise the service.

        Args:
            endpoint: Service endpoint URL.
            timeout: Connection timeout in seconds.
        """
        self.endpoint = endpoint
        self.timeout = timeout

    def call(self, payload: str) -> str:
        """Call the service.

        Args:
            payload: Request payload.

        Returns:
            Response string.
        """
        return f"response: {payload}"
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source)

            parser = SourceCodeParser(tmp, extern_only=True)
            components = parser.parse()

        ops = [op for comp in components for op in comp.operations]
        op_names = {op.name for op in ops}
        # call() has docstring → passes
        assert "call" in op_names
        # __init__ has docstring but should always be excluded in extern_only mode
        assert "__init__" not in op_names, f"__init__ should be excluded in --extern-only mode, got: {op_names}"

    def test_extern_only_excludes_init_with_all(self) -> None:
        """extern_only=True + __all__: __init__ excluded even if class is in __all__."""
        source = '''
__all__ = ["Service"]

class Service:
    """A public service class."""

    def __init__(self, endpoint: str, timeout: int = 30) -> None:
        """Initialise the service.

        Args:
            endpoint: Service endpoint URL.
            timeout: Connection timeout in seconds.
        """
        self.endpoint = endpoint
        self.timeout = timeout

    def call(self, payload: str) -> str:
        """Call the service.

        Args:
            payload: Request payload.

        Returns:
            Response string.
        """
        return f"response: {payload}"
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source)

            parser = SourceCodeParser(tmp, extern_only=True)
            components = parser.parse()

        ops = [op for comp in components for op in comp.operations]
        op_names = {op.name for op in ops}
        # call() has docstring and class is in __all__ → passes
        assert "call" in op_names
        # __init__ should be excluded even when its class is in __all__
        assert "__init__" not in op_names, f"__init__ should be excluded in --extern-only mode, got: {op_names}"

    def test_extern_only_excludes_test_methods(self) -> None:
        """extern_only=True: test_* methods are never external interfaces."""
        source = '''
class TestService:
    """Tests for Service."""

    def test_connect(self) -> None:
        """Test that service connects successfully."""
        pass

    def test_timeout(self) -> None:
        """Test timeout behaviour."""
        pass

    def helper(self) -> str:
        """A helper method, not a test."""
        return "helper"
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source)

            parser = SourceCodeParser(tmp, extern_only=True)
            components = parser.parse()

        ops = [op for comp in components for op in comp.operations]
        op_names = {op.name for op in ops}
        # helper() has docstring → passes
        assert "helper" in op_names
        # test_* methods are never external API
        assert "test_connect" not in op_names, f"test_connect should be excluded in --extern-only mode, got: {op_names}"
        assert "test_timeout" not in op_names, f"test_timeout should be excluded in --extern-only mode, got: {op_names}"

    def test_extern_only_excludes_dunder_methods(self) -> None:
        """extern_only=True: all dunder methods (__call__, __enter__, etc.) excluded."""
        source = '''
class Handler:
    """An event handler class."""

    def __call__(self, event: str) -> str:
        """Handle the event.

        Args:
            event: The event payload.

        Returns:
            Processed result.
        """
        return f"handled: {event}"

    def __enter__(self) -> "Handler":
        """Enter context.

        Returns:
            Self.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context."""
        pass

    def process(self, data: str) -> str:
        """Process data.

        Args:
            data: Input data.

        Returns:
            Result.
        """
        return data
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source)

            parser = SourceCodeParser(tmp, extern_only=True)
            components = parser.parse()

        ops = [op for comp in components for op in comp.operations]
        op_names = {op.name for op in ops}
        # process() is the only real external API
        assert "process" in op_names, f"Expected 'process', got: {op_names}"
        # dunder methods are never external interfaces
        for dunder in ("__call__", "__enter__", "__exit__"):
            assert dunder not in op_names, f"'{dunder}' should be excluded in --extern-only mode, got: {op_names}"

    def test_extern_only_excludes_main(self) -> None:
        """extern_only=True: main() CLI entry points are never external API."""
        source = '''
def main() -> None:
    """CLI entry point for the pipeline.

    Parses command-line arguments and runs the pipeline.
    """
    pass

def run_pipeline(config_path: str) -> dict:
    """Run the pipeline with given config.

    Args:
        config_path: Path to the config file.

    Returns:
        Pipeline result dict.
    """
    return {}
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "mod.py").write_text(source)

            parser = SourceCodeParser(tmp, extern_only=True)
            components = parser.parse()

        ops = [op for comp in components for op in comp.operations]
        op_names = {op.name for op in ops}
        # run_pipeline() has docstring → passes
        assert "run_pipeline" in op_names
        # main() is CLI entry point, not external API
        assert "main" not in op_names, f"'main' should be excluded in --extern-only mode, got: {op_names}"


# =========================================================================
# E2E: pos convert → loadable agent/skill files
# =========================================================================


class TestPosConvertE2E:
    """End-to-end: run pos convert on a real source tree, verify output."""

    SAMPLE_PROJECT: ClassVar[dict[str, str]] = {
        "video_ops/__init__.py": '"""Video processing SDK."""\n',
        "video_ops/transcoder.py": '''
class Transcoder:
    """Transcode video between formats."""

    def transcode(self, input_path: str, output_format: str = "mp4") -> bool:
        """Convert a video file to the target format.

        Args:
            input_path: Path to the input video file.
            output_format: Target output format (default: "mp4").

        Returns:
            True if successful, False otherwise.
        """
        return True

    def probe(self, file_path: str) -> dict:
        """Get video metadata and format info.

        Args:
            file_path: Path to the video file.

        Returns:
            Dict with codec, resolution, bitrate keys.
        """
        return {"codec": "h264", "resolution": "1920x1080"}
''',
        "video_ops/thumbs.py": '''
class Thumbnailer:
    """Generate video thumbnails."""

    def extract_thumbnail(self, input_path: str, time_sec: float = 0.0) -> bytes:
        """Extract a thumbnail frame at a given timestamp.

        Args:
            input_path: Path to the video file.
            time_sec: Timestamp in seconds (default: 0.0).

        Returns:
            PNG image bytes.
        """
        return b""
''',
        "audio_ops/__init__.py": '"""Audio processing SDK."""\n',
        "audio_ops/mixer.py": '''
class AudioMixer:
    """Mix and process audio streams."""

    def mix(self, tracks: list[str], output_path: str) -> str:
        """Mix multiple audio tracks into one.

        Args:
            tracks: List of input track paths.
            output_path: Output file path.

        Returns:
            Path to the mixed output file.
        """
        return output_path
''',
    }

    def _write_project(self, root: Path) -> None:
        for relpath, content in self.SAMPLE_PROJECT.items():
            full = root / relpath
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")

    def test_convert_writes_agent_and_skill_files(self) -> None:
        """pos convert writes loadable agent .md and skill files to --out and --skills."""
        import tempfile

        from clawcodex_ext.cli.sop_cmd.commands import run_sop_command

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            proj = tmp / "media_sdk"
            self._write_project(proj)
            out_dir = tmp / "output"
            skills_dir = tmp / "skills"

            # Run pos convert with --all (include all methods)
            exit_code = run_sop_command(
                [
                    "convert",
                    str(proj),
                    "--strategy",
                    "keyword",
                    "--out",
                    str(out_dir),
                    "--skills",
                    str(skills_dir),
                    "--all",
                ]
            )
            assert exit_code == 0, f"pos convert failed with exit code {exit_code}"

            # ── Verify agent files ──
            agents_dir = out_dir / ".claude" / "agents"
            assert agents_dir.is_dir(), f"Expected agents dir at {agents_dir}"
            agent_files = sorted(agents_dir.glob("*.md"))
            assert len(agent_files) >= 1, f"No agent files in {agents_dir}"

            # Each agent .md should have valid frontmatter
            for af in agent_files:
                content = af.read_text(encoding="utf-8")
                assert content.startswith("---"), f"Agent {af.name} missing frontmatter"
                parts = content.split("---", 2)
                assert len(parts) >= 3, f"Agent {af.name} malformed frontmatter"
                fm_text = parts[1]
                assert "name:" in fm_text, f"Agent {af.name} missing 'name' in frontmatter"
                assert "description:" in fm_text, f"Agent {af.name} missing 'description' in frontmatter"
                assert "tools:" in fm_text, f"Agent {af.name} missing 'tools' in frontmatter"

            # ── Verify skill files ──
            assert skills_dir.is_dir(), f"Expected skills dir at {skills_dir}"
            skill_files = sorted(skills_dir.glob("*-skill.md"))
            assert len(skill_files) >= 1, f"No skill files in {skills_dir}"

            # Each skill file should have required frontmatter fields
            for sf in skill_files:
                content = sf.read_text(encoding="utf-8")
                assert "name:" in content, f"Skill {sf.name} missing name"
                assert "description:" in content, f"Skill {sf.name} missing description"
                assert "allowed-tools:" in content, f"Skill {sf.name} missing allowed-tools"

    def test_convert_extern_only_default(self) -> None:
        """Default behavior (no --all): only documented external interfaces pass."""
        import tempfile

        from clawcodex_ext.cli.sop_cmd.commands import run_sop_command

        # Create source with both documented and undocumented methods
        source = {
            "lib/__init__.py": "",
            "lib/api.py": '''
class Service:
    """Public service."""

    def call(self, payload: str) -> str:
        """Call the service endpoint.

        Args:
            payload: Request payload.

        Returns:
            Response string.
        """
        return f"response: {payload}"

    def _internal(self) -> None:
        """Internal helper."""
        pass
''',
            "lib/hidden.py": '''
class HiddenUtil:
    """Utility without documented methods."""

    def no_docs(self, x: int) -> int:
        return x * 2
''',
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            lib = tmp / "lib_src"
            for relpath, content in source.items():
                full = lib / relpath
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text(content, encoding="utf-8")

            out_dir = tmp / "out"
            exit_code = run_sop_command(
                [
                    "convert",
                    str(lib),
                    "--strategy",
                    "keyword",
                    "--out",
                    str(out_dir),
                    # No --all → default extern-only filtering
                ]
            )
            assert exit_code == 0

            # Verify agents were generated
            agents_dir = out_dir / ".claude" / "agents"
            agent_files = sorted(agents_dir.glob("*.md"))
            # Should have at least the service agent
            assert len(agent_files) >= 1

            # Read agent file and check it references the documented method
            agent_content = agent_files[0].read_text(encoding="utf-8")
            assert "call" in agent_content or "Service" in agent_content, (
                f"Agent should reference documented API, got:\n{agent_content[:500]}"
            )
