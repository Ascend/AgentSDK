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
