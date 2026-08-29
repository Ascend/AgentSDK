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

"""Tests for the workflow YAML emitter."""

from __future__ import annotations

from pathlib import Path

from extensions.sop_converter.skill_grouper import GroupStrategy, group_source_components
from extensions.sop_converter.core.source_parser import SourceCodeParser
from extensions.sop_converter.workflow_mode.capability import StageCapabilityMapper
from extensions.sop_converter.workflow_mode.extractors.adapters.generic import (
    GenericPipelineExtractor,
)
from extensions.sop_converter.workflow_mode.schema import (
    graph_to_engine_yaml_dict,
    validate_workflow_dict,
)
from extensions.sop_converter.workflow_mode.scan_context import SourceScanContext
from extensions.orchestrator.workflow_engine import WorkflowSchema

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _fwa_graph_and_map():
    path = FIXTURES / "fixture_fwa_project"
    scan = SourceScanContext.build(path)
    graph = GenericPipelineExtractor(scan=scan, mode="fwa").extract(path)
    components = SourceCodeParser(str(path)).parse()
    skills = group_source_components(components, strategy=GroupStrategy.COMPONENT_GROUP).skills
    agent_map = StageCapabilityMapper().map(graph, components, skills, scan=scan)
    return graph, agent_map


class TestWorkflowEmitter:
    def test_emitter_dict_shape(self):
        graph, agent_map = _fwa_graph_and_map()
        data = graph_to_engine_yaml_dict(graph, agent_map, workflow_name="test-wf")
        assert data["name"] == "test-wf"
        assert isinstance(data["stages"], list)
        assert len(data["stages"]) >= len(graph.stages)

    def test_dag_validation(self):
        graph, agent_map = _fwa_graph_and_map()
        data = graph_to_engine_yaml_dict(graph, agent_map, workflow_name="test-wf")
        result = validate_workflow_dict(data)
        assert result.ok, result.errors

    def test_f110_round_trip(self):
        graph, agent_map = _fwa_graph_and_map()
        data = graph_to_engine_yaml_dict(graph, agent_map, workflow_name="test-wf")
        data.pop("_emitter_version", None)
        schema = WorkflowSchema.from_dict(data)
        order = schema.build_dag_order()
        assert len(order) == len(data["stages"])

    def test_synthetic_gate_node(self):
        graph, agent_map = _fwa_graph_and_map()
        data = graph_to_engine_yaml_dict(graph, agent_map, workflow_name="test-wf")
        kinds = [s.get("kind") for s in data["stages"]]
        assert "gate" in kinds
