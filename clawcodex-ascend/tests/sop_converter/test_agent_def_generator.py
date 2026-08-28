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

"""Tests for the agent definition generator."""

from __future__ import annotations

from pathlib import Path

from extensions.sop_converter.skill_grouper import GroupStrategy, SkillSpec, group_source_components
from extensions.sop_converter.source_parser import SourceCodeParser
from extensions.sop_converter.workflow_mode.capability import (
    StageCapabilityMapper,
    ensure_arc_stage_skills,
)
from extensions.sop_converter.workflow_mode.extractors.adapters.generic import (
    GenericPipelineExtractor,
)
from extensions.sop_converter.workflow_mode.generator import (
    AgentDefinitionGenerator,
    coarse_agent_skills,
)
from extensions.sop_converter.workflow_mode.generator.overview_gen import control_flow_markdown
from extensions.sop_converter.workflow_mode.scan_context import SourceScanContext

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class TestAgentDefinitionGenerator:
    def test_generate_stage_agents(self, tmp_path: Path):
        path = FIXTURES / "fixture_fwa_project"
        scan = SourceScanContext.build(path)
        graph = GenericPipelineExtractor(scan=scan, mode="fwa").extract(path)
        components = SourceCodeParser(str(path)).parse()
        skills = group_source_components(components, strategy=GroupStrategy.COMPONENT_GROUP).skills
        agent_map = StageCapabilityMapper().map(graph, components, skills, scan=scan)

        gen = AgentDefinitionGenerator()
        paths = gen.generate_stage_agents(graph, agent_map, tmp_path, project_name="fwa-test")
        assert len(paths) >= len(graph.stages)
        for p in paths:
            assert p.exists()
            assert "Agent:" in p.read_text(encoding="utf-8")

    def test_control_flow_markdown(self):
        path = FIXTURES / "fixture_fwa_project"
        scan = SourceScanContext.build(path)
        graph = GenericPipelineExtractor(scan=scan, mode="fwa").extract(path)
        md = control_flow_markdown(graph)
        assert "GATE" in md

    def test_enrich_workflow_stages(self):
        path = FIXTURES / "fixture_fwa_project"
        scan = SourceScanContext.build(path)
        graph = GenericPipelineExtractor(scan=scan, mode="fwa").extract(path)
        components = SourceCodeParser(str(path)).parse()
        skills = group_source_components(components, strategy=GroupStrategy.COMPONENT_GROUP).skills
        agent_map = StageCapabilityMapper().map(graph, components, skills, scan=scan)
        skill_agent_map = {s.name: f"{s.name}-agent" for s in skills}

        stages = AgentDefinitionGenerator().enrich_workflow_stages(
            graph,
            agent_map,
            skill_agent_map=skill_agent_map,
        )
        assert len(stages) == len(graph.stages)
        assert any("[GATE:" in s.description for s in stages)

    def test_coarse_agent_skills_excludes_stage_names(self):
        path = FIXTURES / "fixture_fwa_project"
        scan = SourceScanContext.build(path)
        graph = GenericPipelineExtractor(scan=scan, mode="fwa").extract(path)
        components = SourceCodeParser(str(path)).parse()
        skills = group_source_components(components, strategy=GroupStrategy.COMPONENT_GROUP).skills
        coarse = coarse_agent_skills(skills, graph)
        coarse_names = {s.name for s in coarse}
        stage_names = {s.name for s in graph.stages}
        assert stage_names.isdisjoint(coarse_names)

    def test_finalize_stage_agent_names_prefixed_on_collision(self):
        path = FIXTURES / "fixture_fwa_project"
        scan = SourceScanContext.build(path)
        graph = GenericPipelineExtractor(scan=scan, mode="fwa").extract(path)
        components = SourceCodeParser(str(path)).parse()
        skills = group_source_components(components, strategy=GroupStrategy.COMPONENT_GROUP).skills
        agent_map = StageCapabilityMapper().map(graph, components, skills, scan=scan)

        from extensions.sop_converter.workflow_mode.generator import stage_agent_existing_names

        gen = AgentDefinitionGenerator()
        existing = stage_agent_existing_names(
            skills,
            graph,
            overview_agent_name="clawcodex-overview",
        )
        project = "my-sdk"
        first_stage = graph.stages[0]
        existing.add(f"{first_stage.name}-agent")
        gen.finalize_stage_agent_names(
            graph,
            agent_map,
            project_name=project,
            existing_agent_names=existing,
        )
        profile = agent_map.profile_for_stage(first_stage.id)
        assert profile is not None
        assert profile.mapped_agent == f"{project}-{first_stage.name}-agent"

    def test_stage_skill_name_uses_skill_suffix(self, tmp_path: Path):
        path = FIXTURES / "fixture_fwa_project"
        scan = SourceScanContext.build(path)
        graph = GenericPipelineExtractor(scan=scan, mode="fwa").extract(path)
        components = SourceCodeParser(str(path)).parse()
        skills = group_source_components(components, strategy=GroupStrategy.COMPONENT_GROUP).skills
        agent_map = StageCapabilityMapper().map(graph, components, skills, scan=scan)

        gen = AgentDefinitionGenerator()
        paths = gen.generate_stage_agents(
            graph,
            agent_map,
            tmp_path,
            project_name="fwa-test",
            write_skills=True,
        )
        skill_paths = [p for p in paths if p.name == "SKILL.md"]
        assert skill_paths, "expected stage SKILL.md files"
        for p in skill_paths:
            assert "-stage-skill" not in p.as_posix()
            assert p.parent.name.endswith("-skill")
            content = p.read_text(encoding="utf-8")
            assert "name: " in content
            assert "-stage-skill" not in content.split("---", 2)[1]

    def test_hybrid_agent_includes_output_contract(self, tmp_path: Path):
        from extensions.sop_converter.workflow_mode.extractors.pattern import (
            ARC_COMPAT_CONFIG,
            PatternExtractor,
        )

        path = FIXTURES / "fixture_arc_project"
        graph = PatternExtractor(config=ARC_COMPAT_CONFIG, mode="fwa").extract(path)
        coarse = [SkillSpec(name="merged", description="coarse", allowed_tools=[])]
        skills = ensure_arc_stage_skills(graph, [], coarse, path)
        agent_map = StageCapabilityMapper().map(graph, [], skills)
        gen = AgentDefinitionGenerator()
        gen.generate_stage_agents(
            graph,
            agent_map,
            tmp_path,
            project_name="arc-test",
            bridge_script="bridge/arc-test_bridge.py",
        )
        preprocess = (tmp_path / ".claude" / "agents" / "preprocess-agent.md").read_text(encoding="utf-8")
        assert "## 输出契约" in preprocess
        assert "`normalized.json`" in preprocess

    def test_stage_agent_survives_coarse_write(self, tmp_path: Path):
        """Coarse write_agent must not replace the stage agent markdown."""
        from extensions.sop_converter.agent_md_writer import AgentMarkdownWriter

        path = FIXTURES / "fixture_fwa_project"
        scan = SourceScanContext.build(path)
        graph = GenericPipelineExtractor(scan=scan, mode="fwa").extract(path)
        components = SourceCodeParser(str(path)).parse()
        skills = group_source_components(components, strategy=GroupStrategy.COMPONENT_GROUP).skills
        agent_map = StageCapabilityMapper().map(graph, components, skills, scan=scan)

        writer = AgentMarkdownWriter()
        for skill in coarse_agent_skills(skills, graph):
            writer.write_agent(
                {
                    "name": f"{skill.name}-agent",
                    "description": skill.description,
                    "tools": skill.allowed_tools,
                    "skills": [f"{skill.name}-skill"],
                },
                tmp_path,
            )

        stage = graph.stages[0]
        AgentDefinitionGenerator().generate_stage_agents(
            graph,
            agent_map,
            tmp_path,
            project_name="fwa-test",
            bridge_script="bridge/fwa-test_bridge.py",
            existing_agent_names={f"{s.name}-agent" for s in skills},
        )

        stage_agent = agent_map.by_stage_id[stage.id].mapped_agent
        assert stage_agent is not None
        body = (tmp_path / ".claude" / "agents" / f"{stage_agent}.md").read_text(encoding="utf-8")
        assert "## SOP 工作流" not in body
        assert "# Stage Agent:" in body or "# Hybrid Agent:" in body or "# Wrapper Agent:" in body
