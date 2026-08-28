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
