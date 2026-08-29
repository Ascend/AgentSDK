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

"""Tests for the stage capability mapper."""

from __future__ import annotations

from pathlib import Path

from extensions.sop_converter.skill_grouper import group_source_components, GroupStrategy
from extensions.sop_converter.core.source_parser import SourceCodeParser
from extensions.sop_converter.workflow_mode.capability import (
    ExecutionMode,
    StageCapabilityMapper,
)
from extensions.sop_converter.workflow_mode.capability.analyzer import recommend_execution_mode
from extensions.sop_converter.workflow_mode.extractors.adapters.generic import (
    GenericPipelineExtractor,
)
from extensions.sop_converter.workflow_mode.scan_context import SourceScanContext

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class TestExecutionModeMatrix:
    def test_low_complexity_low_fragility(self):
        assert recommend_execution_mode(0.2, 0.1) == ExecutionMode.AGENT_NATIVE

    def test_high_fragility_wrapper(self):
        assert recommend_execution_mode(0.8, 0.9) == ExecutionMode.WRAPPER

    def test_mid_hybrid(self):
        assert recommend_execution_mode(0.5, 0.4) == ExecutionMode.HYBRID


class TestStageCapabilityMapper:
    def test_fwa_fixture_profiles(self):
        path = FIXTURES / "fixture_fwa_project"
        scan = SourceScanContext.build(path)
        graph = GenericPipelineExtractor(scan=scan, mode="fwa").extract(path)
        components = SourceCodeParser(str(path)).parse()
        skills = group_source_components(components, strategy=GroupStrategy.COMPONENT_GROUP).skills

        agent_map = StageCapabilityMapper().map(graph, components, skills, scan=scan)

        assert len(agent_map.by_stage_id) == len(graph.stages)
        for stage in graph.stages:
            profile = agent_map.by_stage_id[stage.id]
            assert profile.stage_id == stage.id
            assert stage.capability_profile is profile
            assert profile.mapped_agent is not None
