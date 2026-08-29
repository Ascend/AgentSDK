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

"""Tests for ARC stage to skill mapping."""

from __future__ import annotations

from pathlib import Path

from extensions.sop_converter.skill_grouper import SkillSpec, GroupStrategy, group_source_components
from extensions.sop_converter.core.source_parser import SourceCodeParser
from extensions.sop_converter.workflow_mode.bridge.dispatch import resolve_stage_module_path
from extensions.sop_converter.workflow_mode.capability import (
    StageCapabilityMapper,
    ensure_arc_stage_skills,
)
from extensions.sop_converter.workflow_mode.capability.arc_mapper import (
    arc_stage_impl_rel_path,
    resolve_arc_stage_impl_path,
)
from extensions.sop_converter.workflow_mode.extractors.pattern import (
    ARC_COMPAT_CONFIG,
    PatternExtractor,
    _resolve_pipeline_dir,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ARC_FIXTURE = FIXTURES / "fixture_arc_project"
ARC_REPO = Path(r"D:\projects\AutoResearchClaw")


class TestArcStageImplResolution:
    def test_resolve_impl_prefers_stage_impls(self):
        pipeline = _resolve_pipeline_dir(ARC_FIXTURE, ARC_COMPAT_CONFIG)
        assert pipeline is not None
        graph = PatternExtractor(config=ARC_COMPAT_CONFIG, mode="fwa").extract(ARC_FIXTURE)
        stage = graph.stages[0]
        impl = resolve_arc_stage_impl_path(ARC_FIXTURE, pipeline, stage)
        assert impl is not None
        assert impl.name == "_stages.py"
        assert impl.parent.name == "stage_impls"

    def test_bridge_module_path_uses_stage_impls(self):
        graph = PatternExtractor(config=ARC_COMPAT_CONFIG, mode="fwa").extract(ARC_FIXTURE)
        stage = graph.stages[0]
        rel = resolve_stage_module_path(stage, ARC_FIXTURE)
        assert rel is not None
        assert "stage_impls" in rel
        assert rel.endswith("_stages.py")


class TestArcStageSkillSynthesis:
    def test_synthesizes_per_stage_skills(self):
        graph = PatternExtractor(config=ARC_COMPAT_CONFIG, mode="fwa").extract(ARC_FIXTURE)
        coarse = [
            SkillSpec(name="researchclaw_merged", description="coarse", allowed_tools=["execute_stage"]),
        ]
        skills = ensure_arc_stage_skills(graph, [], coarse, ARC_FIXTURE)
        names = {s.name for s in skills}
        assert "researchclaw_merged" in names
        for stage in graph.stages:
            assert stage.name in names

    def test_mapper_assigns_unique_agents_per_stage(self):
        graph = PatternExtractor(config=ARC_COMPAT_CONFIG, mode="fwa").extract(ARC_FIXTURE)
        coarse = [SkillSpec(name="merged", description="coarse", allowed_tools=[])]
        skills = ensure_arc_stage_skills(graph, [], coarse, ARC_FIXTURE)
        agent_map = StageCapabilityMapper().map(graph, [], skills)
        assert agent_map.has_mapped_stages
        agents = {agent_map.by_stage_id[s.id].mapped_agent for s in graph.stages}
        assert len(agents) == len(graph.stages)
        for stage in graph.stages:
            profile = agent_map.by_stage_id[stage.id]
            assert profile.mapping_confidence > 0
            assert profile.mapped_skill == stage.name
            assert profile.mapped_agent == f"{stage.name}-agent"


class TestArcMapperRealRepo:
    def test_real_repo_unique_stage_agents(self):
        if not ARC_REPO.is_dir():
            return
        graph = PatternExtractor(config=ARC_COMPAT_CONFIG, mode="fwa").extract(ARC_REPO)
        components = SourceCodeParser(str(ARC_REPO)).parse()
        coarse = group_source_components(components, strategy=GroupStrategy.KEYWORD_MATCH).skills
        skills = ensure_arc_stage_skills(graph, components, coarse, ARC_REPO)
        agent_map = StageCapabilityMapper().map(graph, components, skills)
        assert agent_map.has_mapped_stages
        agents = [agent_map.by_stage_id[s.id].mapped_agent for s in graph.stages]
        assert len(set(agents)) == len(graph.stages)
        pipeline = _resolve_pipeline_dir(ARC_REPO, ARC_COMPAT_CONFIG)
        assert pipeline is not None
        rel = arc_stage_impl_rel_path(ARC_REPO, pipeline, graph.stages[0])
        assert rel is not None
        assert "stage_impls" in rel
