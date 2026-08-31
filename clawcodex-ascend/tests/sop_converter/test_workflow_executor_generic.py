#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# AgentSDK is licensed under Mulan PSL v2.

"""Regression tests for SDK-neutral workflow executor wrappers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from extensions.sop_converter.core.source_parser import ParamSpec, SourceOperation
from extensions.sop_converter.tool_registry_bridge import (
    _generate_pipeline_execute_stage_stub,
    _generate_wrapper_script,
    _is_workflow_execute_operation,
)


def _workflow_op() -> SourceOperation:
    return SourceOperation(
        name="run_step",
        description="Run one workflow step.",
        parameters=[
            ParamSpec(name="step_id", type_hint="int", required=True),
            ParamSpec(name="project_dir", type_hint="Path", required=True),
        ],
        file_stem="runner",
    )


def test_workflow_executor_detection_uses_callable_shape() -> None:
    assert _is_workflow_execute_operation(_workflow_op())


def test_workflow_executor_stub_uses_source_module_without_vendor_imports() -> None:
    stub = _generate_pipeline_execute_stage_stub(
        _workflow_op(),
        module_name="demo.runner",
    )

    assert "demo.runner" in stub
    assert "researchclaw" not in stub
    assert "RCConfig" not in stub
    assert "AdapterBundle" not in stub


def test_generic_workflow_wrapper_executes_with_path_and_enum_coercion(tmp_path: Path) -> None:
    package = tmp_path / "demo"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "runner.py").write_text(
        """
from enum import Enum
from pathlib import Path


class Step(Enum):
    PREPARE_DATA = "prepare-data"


def run_step(step: Step, project_dir: Path):
    return {
        "step": step.name,
        "project_dir": str(project_dir),
        "is_path": isinstance(project_dir, Path),
    }
""",
        encoding="utf-8",
    )
    op = SourceOperation(
        name="run_step",
        description="Run one workflow step.",
        parameters=[
            ParamSpec(name="step", type_hint="Step", required=True),
            ParamSpec(name="project_dir", type_hint="Path", required=True),
        ],
        file_stem="runner",
    )

    script = _generate_wrapper_script(
        [op],
        class_name=None,
        module_name="demo.runner",
        file_stem="runner",
        source_dir=str(tmp_path),
        scripts_dir=tmp_path / "scripts",
    )
    spec = importlib.util.spec_from_file_location("generated_runner", script)
    assert spec is not None and spec.loader is not None
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)

    result = wrapper.run_step("prepare-data", str(tmp_path))

    assert result == {
        "step": "PREPARE_DATA",
        "project_dir": str(tmp_path),
        "is_path": True,
    }
