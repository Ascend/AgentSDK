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

"""Tests for the workflow extractor."""

from __future__ import annotations

from pathlib import Path

from extensions.sop_converter.workflow_mode.extractors.adapters.generic import (
    GenericPipelineExtractor,
)
from extensions.sop_converter.workflow_mode.extractors.registry import ExtractorRegistry
from extensions.sop_converter.workflow_mode.pipeline import discriminate_and_extract
from extensions.sop_converter.workflow_mode.scan_context import SourceScanContext

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class TestGenericPipelineExtractor:
    def test_fwa_stages_and_transitions(self):
        path = FIXTURES / "fixture_fwa_project"
        scan = SourceScanContext.build(path)
        ext = GenericPipelineExtractor(scan=scan, mode="fwa")
        graph = ext.extract(path)
        assert len(graph.stages) == 3
        labels = {s.label for s in graph.stages}
        assert labels == {"PREPROCESS", "ANALYZE", "GENERATE"}
        assert len(graph.transitions) == 2
        assert graph.gates
        assert 2 in graph.gates

    def test_sdk_empty_graph(self):
        path = FIXTURES / "fixture_sdk_project"
        disc, graph = discriminate_and_extract(path)
        assert disc.mode == "sdk"
        assert graph is None

    def test_fwa_force_on_sdk_fallback(self):
        path = FIXTURES / "fixture_sdk_project"
        disc, graph = discriminate_and_extract(path, force_mode="fwa")
        assert disc.mode == "fwa"
        assert graph is None

    def test_registry_default(self):
        path = FIXTURES / "fixture_fwa_project"
        ext = ExtractorRegistry.get_extractor(path)
        assert isinstance(ext, GenericPipelineExtractor)

    def test_primary_stage_enum_filter(self):
        path = FIXTURES / "fixture_fwa_project"
        scan = SourceScanContext.build(path)
        assert scan.primary_stage_enum == "Stage"
        ext = GenericPipelineExtractor(scan=scan, mode="fwa")
        stages = ext.extract_stages(path)
        assert all(s.source_class == "Stage" for s in stages)

    def test_nested_bool_returns_are_not_decision_outcomes(self, tmp_path: Path):
        (tmp_path / "stages.py").write_text(
            "from enum import IntEnum\n"
            "class Stage(IntEnum):\n"
            "    START = 1\n"
            "    REFINE = 2\n"
            "    DONE = 3\n"
            "STAGE_SEQUENCE = tuple(Stage)\n"
            "NEXT_STAGE = {\n"
            "    stage: STAGE_SEQUENCE[i + 1] if i + 1 < len(STAGE_SEQUENCE) else None\n"
            "    for i, stage in enumerate(STAGE_SEQUENCE)\n"
            "}\n"
            "def _execute_refine():\n"
            "    def _is_better(candidate, current):\n"
            "        if candidate is None:\n"
            "            return False\n"
            "        return True\n"
            "    return Stage.DONE\n"
            "def should_pause():\n"
            "    return False\n",
            encoding="utf-8",
        )
        scan = SourceScanContext.build(tmp_path)
        ext = GenericPipelineExtractor(scan=scan, mode="fwa")
        decisions = ext.extract_decisions(tmp_path)
        assert 3 not in decisions
        refine = decisions.get(2)
        assert refine is not None
        assert "False" not in refine.outcomes
        assert "True" not in refine.outcomes
        assert "DONE" in refine.outcomes

    def test_entry_function_from_name_fallback(self, tmp_path: Path):
        (tmp_path / "pipeline.py").write_text(
            "from enum import IntEnum\n"
            "class Stage(IntEnum):\n"
            "    START = 1\n"
            "    DONE = 2\n"
            "def _execute_start():\n"
            "    return Stage.DONE\n"
            "def _execute_done():\n"
            "    return None\n",
            encoding="utf-8",
        )
        scan = SourceScanContext.build(tmp_path)
        ext = GenericPipelineExtractor(scan=scan, mode="fwa")
        stages = ext.extract_stages(tmp_path)
        by_label = {s.label: s for s in stages}
        assert by_label["START"].entry_function == "_execute_start"
        assert by_label["DONE"].entry_function == "_execute_done"
        assert by_label["START"].file_path == "pipeline.py"
