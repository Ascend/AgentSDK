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

# pylint: disable=relative-beyond-top-level
# tech_v26.2.0 has not merged package marker files (e.g. extensions/__init__.py)
# yet, so pylint cannot tell that sop_converter is a Python package and flags
# valid relative imports as E0402. Drop this tag once the package markers land.


"""Map WorkflowGraph IR to overview WorkflowStage containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from extensions.sop_converter.agent_md_writer import AgentComponentInfo, WorkflowStage
from .extractors.models import ExtractedStage, WorkflowGraph

if TYPE_CHECKING:
    from .capability.models import StageAgentMap


def extracted_stage_to_workflow_stage(
    node: ExtractedStage,
    graph: WorkflowGraph,
    *,
    stage_by_id: dict[int, ExtractedStage],
    skill_agent_map: dict[str, str],
) -> WorkflowStage:
    dep_ids = [t.from_stage for t in graph.transitions if t.to_stage == node.id]
    dep_labels = [stage_by_id[d].label for d in dep_ids if d in stage_by_id]
    contract = graph.contracts.get(node.id)
    agent = skill_agent_map.get(node.name, f"TODO-{node.name}-agent")
    desc = node.description
    if node.inferred and "[inferred]" not in desc:
        desc = (desc + " [inferred]").strip()
    return WorkflowStage(
        name=node.label,
        order=node.id,
        description=desc,
        responsible_agent=agent,
        depends_on=dep_labels or None,
        output_type=", ".join(contract.output_files) if contract else "",
    )


def _is_empty_stage(
    node: ExtractedStage,
    graph: WorkflowGraph,
    agent_map: StageAgentMap | None,
    skill_agent_map: dict[str, str],
) -> bool:
    """A stage is "empty" (should be skipped) when it has no real mapping.

    Criteria (all must hold):
    1. No mapped_agent in StageAgentMap (or profile absent).
    2. mapping_confidence <= 0.
    3. Stage name not present in skill_agent_map (no skill matched by name).
    4. No contract with output_files (no I/O contract).
    """
    if agent_map is not None:
        profile = agent_map.profile_for_stage(node.id)
        if profile and profile.mapped_agent and profile.mapping_confidence > 0:
            return False
    if node.name in skill_agent_map:
        return False
    contract = graph.contracts.get(node.id)
    if contract and contract.output_files:
        return False
    return True


def build_workflow_stages(
    graph: WorkflowGraph,
    *,
    skill_agent_map: dict[str, str],
) -> list[WorkflowStage]:
    stage_by_id = {s.id: s for s in graph.stages}
    stages: list[WorkflowStage] = []
    for node in sorted(graph.stages, key=lambda s: s.id):
        if _is_empty_stage(node, graph, None, skill_agent_map):
            continue
        stages.append(
            extracted_stage_to_workflow_stage(
                node,
                graph,
                stage_by_id=stage_by_id,
                skill_agent_map=skill_agent_map,
            )
        )
    return stages


def build_workflow_stages_with_map(
    graph: WorkflowGraph,
    agent_map: StageAgentMap,
    *,
    skill_agent_map: dict[str, str],
) -> list[WorkflowStage]:
    """Build workflow stages preferring StageAgentMap mappings."""

    stage_by_id = {s.id: s for s in graph.stages}
    merged_map = dict(skill_agent_map)
    for skill, agent in agent_map.skill_to_agent.items():
        merged_map.setdefault(skill, agent)

    stages: list[WorkflowStage] = []
    for node in sorted(graph.stages, key=lambda s: s.id):
        if _is_empty_stage(node, graph, agent_map, merged_map):
            continue
        profile = agent_map.profile_for_stage(node.id)
        if profile and profile.mapped_agent:
            dep_ids = [t.from_stage for t in graph.transitions if t.to_stage == node.id]
            dep_labels = [stage_by_id[d].label for d in dep_ids if d in stage_by_id]
            contract = graph.contracts.get(node.id)
            desc = node.description
            if node.inferred and "[inferred]" not in desc:
                desc = (desc + " [inferred]").strip()
            stages.append(
                WorkflowStage(
                    name=node.label,
                    order=node.id,
                    description=desc,
                    responsible_agent=profile.mapped_agent,
                    depends_on=dep_labels or None,
                    output_type=", ".join(contract.output_files) if contract else "",
                )
            )
        else:
            stages.append(
                extracted_stage_to_workflow_stage(
                    node,
                    graph,
                    stage_by_id=stage_by_id,
                    skill_agent_map=merged_map,
                )
            )
    return stages


def sync_workflow_stages_agents(
    stages: list[WorkflowStage],
    graph: WorkflowGraph,
    agent_map: StageAgentMap,
) -> list[WorkflowStage]:
    """Refresh ``responsible_agent`` from finalized ``StageAgentMap`` profiles."""
    synced: list[WorkflowStage] = []
    for ws in stages:
        stage = next((s for s in graph.stages if s.id == ws.order), None)
        agent = ws.responsible_agent
        if stage is not None:
            profile = agent_map.profile_for_stage(stage.id)
            if profile and profile.mapped_agent:
                agent = profile.mapped_agent
        synced.append(
            WorkflowStage(
                name=ws.name,
                order=ws.order,
                description=ws.description,
                responsible_agent=agent,
                depends_on=ws.depends_on,
                output_type=ws.output_type,
            )
        )
    return synced


def sync_overview_component_agents(
    component_agents: list[AgentComponentInfo],
    workflow_graph: WorkflowGraph,
    agent_map: StageAgentMap,
) -> list[AgentComponentInfo]:
    """Align overview component entries with finalized stage agent names."""
    stage_names = {s.name for s in workflow_graph.stages}
    synced: list[AgentComponentInfo] = []
    for agent in component_agents:
        skill_name = agent.name[: -len("-agent")] if agent.name.endswith("-agent") else agent.name
        if skill_name not in stage_names:
            synced.append(agent)
            continue
        stage = next(s for s in workflow_graph.stages if s.name == skill_name)
        profile = agent_map.profile_for_stage(stage.id)
        if not profile or not profile.mapped_agent:
            synced.append(agent)
            continue
        mapped = profile.mapped_agent
        synced.append(
            AgentComponentInfo(
                name=mapped,
                description=agent.description,
                capabilities=agent.capabilities,
                input_types=agent.input_types,
                output_types=agent.output_types,
                invoke_pattern=f"@{mapped} {{task}}",
            )
        )
    return synced
