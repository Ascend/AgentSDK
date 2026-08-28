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


"""Emit workflow.yaml from WorkflowGraph IR."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from ..capability.models import StageAgentMap
from ..extractors.models import DecisionSpec, ExtractedStage, GateSpec, WorkflowGraph
from .dag_validator import validate_workflow_dict
from .validator_spec import contract_to_validators

logger = logging.getLogger(__name__)

EMITTER_VERSION = "1"


def _build_base_depends_on(graph: WorkflowGraph) -> dict[int, list[int]]:
    deps: dict[int, list[int]] = {s.id: [] for s in graph.stages}
    for t in graph.transitions:
        if t.to_stage in deps and t.from_stage not in deps[t.to_stage]:
            deps[t.to_stage].append(t.from_stage)
    return deps


def _allocate_synthetic_id(used: set[int], anchor: int, suffix: int) -> int:
    candidate = anchor * 100 + suffix
    while candidate in used:
        candidate += 1
    used.add(candidate)
    return candidate


def _stage_prompt(
    stage: ExtractedStage,
    graph: WorkflowGraph,
    agent_map: StageAgentMap,
) -> str:
    profile = agent_map.profile_for_stage(stage.id)
    agent = profile.mapped_agent if profile and profile.mapped_agent else f"TODO-{stage.name}-agent"
    contract = graph.contracts.get(stage.id)
    lines = [
        f"执行阶段 {stage.label}。负责 Agent: @{agent}。",
        "",
        stage.description or f"Run stage {stage.name}.",
    ]
    if contract:
        if contract.input_files:
            lines.append(f"输入契约: {', '.join(contract.input_files)}")
        if contract.output_files:
            lines.append(f"输出契约: {', '.join(contract.output_files)}")
    if profile:
        lines.append(f"执行模式: {profile.execution_mode.value}")
    return "\n".join(lines)


def _agent_stage_dict(
    stage: ExtractedStage,
    graph: WorkflowGraph,
    agent_map: StageAgentMap,
    depends_on: list[int],
) -> dict[str, Any]:
    profile = agent_map.profile_for_stage(stage.id)
    agent = profile.mapped_agent if profile and profile.mapped_agent else f"TODO-{stage.name}-agent"
    contract = graph.contracts.get(stage.id)
    validators = contract_to_validators(contract) if contract else []

    node: dict[str, Any] = {
        "id": stage.id,
        "name": stage.label,
        "kind": "agent",
        "phase": stage.name,
        "prompt": _stage_prompt(stage, graph, agent_map),
        "depends_on": sorted(depends_on),
        "timeout_seconds": 600,
        "on_error": "fail",
    }
    if validators:
        node["validators"] = validators

    agent_config: dict[str, Any] = {"agent": agent}
    if profile:
        agent_config["execution_mode"] = profile.execution_mode.value
        if profile.recommended_tools:
            agent_config["tools"] = profile.recommended_tools
    node["agent_config"] = agent_config
    return node


def _gate_stage_dict(
    gate: GateSpec,
    gate_id: int,
    anchor: ExtractedStage,
    depends_on: list[int],
) -> dict[str, Any]:
    threshold = 0.8 if gate.approval_mode == "threshold" else None
    node: dict[str, Any] = {
        "id": gate_id,
        "name": f"GATE: {anchor.label}",
        "kind": "gate",
        "phase": f"gate-{anchor.name}",
        "prompt": gate.description or f"Quality gate after {anchor.label}.",
        "depends_on": sorted(depends_on),
        "gate_mode": gate.approval_mode or "manual",
        "gate_rollback_to": gate.stage_id,
        "timeout_seconds": 300,
        "on_error": "rollback",
    }
    if threshold is not None:
        node["gate_threshold"] = threshold
    return node


def _decision_stage_dict(
    decision: DecisionSpec,
    decision_id: int,
    anchor: ExtractedStage,
    depends_on: list[int],
) -> dict[str, Any]:
    outcomes: dict[str, dict[str, Any]] = {}
    for name, spec in decision.outcomes.items():
        entry: dict[str, Any] = {}
        if spec.next_stage is not None:
            entry["next"] = spec.next_stage
        # ponytail: null next means "user fill in", validator skips null refs (L114 dag_validator)
        if spec.rollback_to is not None:
            entry["rollback_to"] = spec.rollback_to
        if spec.max_times is not None:
            entry["max_times"] = spec.max_times
        outcomes[name] = entry

    return {
        "id": decision_id,
        "name": f"DECISION: {anchor.label}",
        "kind": "decision",
        "phase": f"decide-{anchor.name}",
        "prompt": decision.source_func or f"Decision after {anchor.label}.",
        "depends_on": sorted(depends_on),
        "timeout_seconds": 300,
        "on_error": "fail",
        "decision_outcomes": outcomes,
    }


def graph_to_engine_yaml_dict(
    graph: WorkflowGraph,
    agent_map: StageAgentMap,
    *,
    workflow_name: str,
    description: str = "",
) -> dict[str, Any]:
    """WorkflowGraph to workflow.yaml shape."""
    stage_by_id = {s.id: s for s in graph.stages}
    used_ids = {s.id for s in graph.stages}
    base_deps = _build_base_depends_on(graph)

    # sid → [(kind, synthetic_id, spec), ...] inserted after agent stage
    synthetics_after: dict[int, list[tuple[str, int, Any]]] = {}

    for sid, gate in graph.gates.items():
        if sid not in stage_by_id:
            continue
        gate_id = _allocate_synthetic_id(used_ids, sid, 1)
        synthetics_after.setdefault(sid, []).append(("gate", gate_id, gate))

    for sid, decision in graph.decisions.items():
        if sid not in stage_by_id:
            continue
        if decision.inferred and not decision.outcomes:
            continue
        if not decision.outcomes:
            continue
        decision_id = _allocate_synthetic_id(used_ids, sid, 2)
        synthetics_after.setdefault(sid, []).append(("decision", decision_id, decision))

    def _resolve_dep(dep: int) -> int:
        chain = synthetics_after.get(dep, [])
        return chain[-1][1] if chain else dep

    rewritten_deps: dict[int, list[int]] = {
        stage.id: [_resolve_dep(d) for d in base_deps.get(stage.id, [])] for stage in graph.stages
    }

    # Filter out empty stages: no mapped agent AND no contract output AND no skill match.
    # These are placeholder stages (e.g. misidentified enum members) that should not
    # be emitted to workflow.yaml or overview markdown.
    skill_to_agent = agent_map.skill_to_agent
    visible_stage_ids: set[int] = set()
    for stage in graph.stages:
        profile = agent_map.profile_for_stage(stage.id)
        if profile and profile.mapped_agent and profile.mapping_confidence > 0:
            visible_stage_ids.add(stage.id)
            continue
        if stage.name in skill_to_agent:
            visible_stage_ids.add(stage.id)
            continue
        contract = graph.contracts.get(stage.id)
        if contract and contract.output_files:
            visible_stage_ids.add(stage.id)

    stages_out: list[dict[str, Any]] = []
    for stage in sorted(graph.stages, key=lambda s: s.id):
        if stage.id not in visible_stage_ids:
            continue
        stages_out.append(_agent_stage_dict(stage, graph, agent_map, rewritten_deps.get(stage.id, [])))
        prev_dep = [stage.id]
        for kind, syn_id, spec in synthetics_after.get(stage.id, []):
            if kind == "gate":
                stages_out.append(_gate_stage_dict(spec, syn_id, stage, prev_dep))
            else:
                stages_out.append(_decision_stage_dict(spec, syn_id, stage, prev_dep))
            prev_dep = [syn_id]

    return {
        "name": workflow_name,
        "version": "1.0",
        "description": description or f"Auto-generated workflow from {graph.source_dir}",
        "stages": stages_out,
        "config": {"workspace": "."},
        "_emitter_version": EMITTER_VERSION,
    }


def emit_engine_workflow_yaml(
    graph: WorkflowGraph,
    agent_map: StageAgentMap,
    output_dir: Path,
    *,
    workflow_name: str,
    description: str = "",
    strict: bool = False,
) -> Path:
    """Write {output_dir}/workflow.yaml and return its path."""
    data = graph_to_engine_yaml_dict(graph, agent_map, workflow_name=workflow_name, description=description)
    # Strip internal metadata before YAML write
    emitter_version = data.pop("_emitter_version", EMITTER_VERSION)
    validation = validate_workflow_dict(data)

    for w in validation.warnings:
        logger.warning("workflow.yaml validation: %s", w)
    if validation.errors:
        for e in validation.errors:
            logger.warning("workflow.yaml validation error: %s", e)
        if strict:
            raise ValueError("; ".join(validation.errors))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "workflow.yaml"

    header_lines = [
        "# Auto-generated by the sop_converter emitter",
        f"# emitter_version: {emitter_version}",
    ]
    if validation.warnings:
        header_lines.append("# VALIDATION_WARNINGS:")
        for msg in validation.warnings:
            header_lines.append(f"#   - {msg}")
    if validation.errors:
        header_lines.append("# VALIDATION_ERRORS:")
        for msg in validation.errors:
            header_lines.append(f"#   - {msg}")

    yaml_body = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)
    out_path.write_text("\n".join(header_lines) + "\n" + yaml_body, encoding="utf-8")
    return out_path
