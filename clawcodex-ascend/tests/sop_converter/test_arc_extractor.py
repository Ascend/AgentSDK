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

"""Tests for ArcExtractor (now via PatternExtractor + ARC_COMPAT_CONFIG)."""

from __future__ import annotations

from pathlib import Path

import pytest

from extensions.sop_converter.workflow_mode.discriminator import _detect_adapter_name
from extensions.sop_converter.workflow_mode.extractors.adapters.generic import (
    GenericPipelineExtractor,
)
from extensions.sop_converter.workflow_mode.extractors.registry import ExtractorRegistry
from extensions.sop_converter.workflow_mode.scan_context import SourceScanContext

from extensions.sop_converter.workflow_mode.extractors.pattern import (
    ARC_COMPAT_CONFIG,
    PatternExtractor,
    _resolve_pipeline_dir,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ARC_FIXTURE = FIXTURES / "fixture_arc_project"
ARC_REPO = Path(r"D:\projects\AutoResearchClaw")


class TestArcExtractorFixture:
    def test_resolve_pipeline_dir(self):
        assert _resolve_pipeline_dir(ARC_FIXTURE, ARC_COMPAT_CONFIG) == ARC_FIXTURE.resolve()

    def test_arc_fixture_graph(self):
        ext = PatternExtractor(config=ARC_COMPAT_CONFIG, mode="fwa")
        graph = ext.extract(ARC_FIXTURE)
        assert len(graph.stages) == 3
        assert len(graph.transitions) == 2
        assert graph.gates[2].stage_id == 2
        assert graph.gates[2].rollback_to == 1
        assert graph.contracts[1].output_files == ["normalized.json"]
        assert graph.stages[0].entry_function == "_execute_preprocess"
        assert graph.stages[0].file_path == "executor.py"

    def test_generic_dict_comp_transitions(self):
        scan = SourceScanContext.build(ARC_FIXTURE)
        ext = GenericPipelineExtractor(scan=scan, mode="fwa")
        transitions = ext.extract_transitions(ARC_FIXTURE)
        assert len(transitions) == 2
        assert (1, 2) in {(t.from_stage, t.to_stage) for t in transitions}
        # GATE_ROLLBACK (2→1) must not become a DAG edge.
        assert (2, 1) not in {(t.from_stage, t.to_stage) for t in transitions}

    def test_generic_gate_rollback_target(self):
        scan = SourceScanContext.build(ARC_FIXTURE)
        ext = GenericPipelineExtractor(scan=scan, mode="fwa")
        gates = ext.extract_gates(ARC_FIXTURE)
        assert gates[2].rollback_to == 1

    def test_generic_contracts_dict(self):
        scan = SourceScanContext.build(ARC_FIXTURE)
        ext = GenericPipelineExtractor(scan=scan, mode="fwa")
        contracts = ext.extract_contracts(ARC_FIXTURE)
        assert contracts[1].output_files == ["normalized.json"]
        assert contracts[3].input_files == ["analysis.json"]

    def test_generic_fixture_entry_function(self):
        scan = SourceScanContext.build(ARC_FIXTURE)
        ext = GenericPipelineExtractor(scan=scan, mode="fwa")
        stages = ext.extract_stages(ARC_FIXTURE)
        by_label = {s.label: s for s in stages}
        assert by_label["PREPROCESS"].entry_function == "_execute_preprocess"
        assert by_label["PREPROCESS"].file_path == "stage_impls/_stages.py"
        assert by_label["ANALYZE"].entry_function == "_execute_analyze"


@pytest.mark.skipif(not ARC_REPO.is_dir(), reason="AutoResearchClaw not checked out locally")
class TestArcExtractorRealRepo:
    def test_detect_adapter_arc(self):
        # _detect_adapter_name now always returns "generic" (no built-in project-specific adapter)
        assert _detect_adapter_name(ARC_REPO) == "generic"

    def test_full_pipeline_extraction(self):
        ext = PatternExtractor(config=ARC_COMPAT_CONFIG, mode="fwa")
        graph = ext.extract(ARC_REPO)
        assert len(graph.stages) == 23
        assert len(graph.transitions) == 22
        assert len(graph.contracts) == 23
        assert graph.gates
        assert graph.gates[5].rollback_to == 4
        assert graph.stages[0].entry_function == "_execute_topic_init"
        assert graph.extraction_quality == "full"

    def test_generic_real_repo_forward_dag(self):
        scan = SourceScanContext.build(ARC_REPO)
        ext = GenericPipelineExtractor(scan=scan, mode="fwa")
        graph = ext.extract(ARC_REPO)
        assert len(graph.stages) == 23
        assert len(graph.transitions) == 22
        seq = [s.id for s in sorted(graph.stages, key=lambda s: s.id)]
        for t in graph.transitions:
            assert seq.index(t.from_stage) < seq.index(t.to_stage), t
        assert graph.gates[5].rollback_to == 4
        assert graph.gates[9].rollback_to == 8
        assert graph.gates[20].rollback_to == 16
        from extensions.sop_converter.workflow_mode.capability.models import StageAgentMap
        from extensions.sop_converter.workflow_mode.schema import (
            graph_to_engine_yaml_dict,
            validate_workflow_dict,
        )

        data = graph_to_engine_yaml_dict(
            graph,
            StageAgentMap(by_stage_id={}, skill_to_agent={}),
            workflow_name="arc",
        )
        result = validate_workflow_dict(data)
        assert result.ok, result.errors
        # Unmatched should_/refine helpers must not dump onto CITATION_VERIFY.
        assert 23 not in graph.decisions
        assert 15 in graph.decisions
        assert set(graph.decisions[15].outcomes) >= {"pivot", "refine"}
        assert graph.stages[0].entry_function == "_execute_topic_init"
        assert graph.stages[0].file_path and "stage_impls" in graph.stages[0].file_path.replace("\\", "/")

    def test_registry_auto_select_arc(self):
        # With the built-in arc adapter removed, get_extractor defaults to GenericPipelineExtractor
        ext = ExtractorRegistry.get_extractor(ARC_REPO)
        assert isinstance(ext, GenericPipelineExtractor)
