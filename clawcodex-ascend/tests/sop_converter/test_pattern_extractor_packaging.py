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

"""Regression tests for the production PatternExtractor package boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_examples_namespace_reexports_production_types() -> None:
    compatibility = pytest.importorskip(
        "examples.sdk_extractor",
        reason="AgentSDK migration intentionally excludes example-only code",
    )
    from extensions.sop_converter.workflow_mode.extractors.pattern import (
        PatternExtractor,
        PipelineConfig,
    )

    assert compatibility.PatternExtractor is PatternExtractor
    assert compatibility.PipelineConfig is PipelineConfig


def test_production_modules_do_not_import_examples() -> None:
    production_files = (REPO_ROOT / "extensions" / "sop_converter" / "workflow_mode" / "extractors" / "pattern.py",)

    for path in production_files:
        source = path.read_text(encoding="utf-8")
        assert "from examples" not in source
        assert "import examples" not in source

    deprecated_arc = REPO_ROOT / "extensions" / "sop_converter" / "workflow_mode" / "extractors" / "adapters" / "arc.py"
    assert not deprecated_arc.exists()


def test_capability_import_outside_repo_does_not_load_arc_support(tmp_path: Path) -> None:
    script = """
import sys
from pathlib import Path
from extensions.sop_converter.workflow_mode.capability import (
    StageCapabilityMapper,
    ensure_stage_skills,
)

skills = [object()]
assert ensure_stage_skills(object(), [], skills, Path.cwd()) is skills
assert StageCapabilityMapper.__name__ == "StageCapabilityMapper"
assert "extensions.sop_converter.workflow_mode.capability.arc_mapper" not in sys.modules
assert "extensions.sop_converter.workflow_mode.extractors.adapters.arc" not in sys.modules
"""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
