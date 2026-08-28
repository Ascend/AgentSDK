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


"""Workflow extraction preview formatting."""

from __future__ import annotations

from ..models import THRESHOLD_FWA, THRESHOLD_SDK, DiscriminationResult
from .models import WorkflowGraph


def format_discrimination_summary(disc: DiscriminationResult) -> str:
    """Human-readable mode selection report."""
    lines: list[str] = []
    forced = " (forced via --mode)" if disc.forced else ""
    lines.append(f"Workflow discrimination{forced}:")
    lines.append(f"  Selected mode: {disc.mode}")
    lines.append(
        f"  Total score: {disc.total_score:.2f} (thresholds: hybrid/sdk ≥ {THRESHOLD_SDK}, fwa ≥ {THRESHOLD_FWA})"
    )
    matched_n = sum(1 for m in disc.matches if m.matched)
    total_n = len(disc.matches)
    lines.append(f"  Rule hit rate: {disc.confidence:.0%} ({matched_n}/{total_n} heuristics)")
    if not disc.forced:
        qual = "passed" if disc.fwa_qualified else "not passed"
        lines.append(f"  FWA combo gate: {qual} (needs stage_enum + state_transition or gate_definition)")
    lines.append(f"  Recommended extractor: {disc.recommended_extractor}")
    lines.append("  Matched features:")
    for m in disc.matches:
        if m.matched:
            lines.append(f"    + {m.name} (weight={m.weight}, score={m.score:.2f}): {m.evidence}")
    for m in disc.matches:
        if not m.matched:
            lines.append(f"    - {m.name} (weight={m.weight}): not detected")
    if disc.forced:
        lines.append("  Selection: explicit --mode flag")
    elif disc.mode == "fwa":
        lines.append(
            "  Selection: score ≥ fwa threshold and combo gate passed → emit workflow.yaml, stage agents, bridge"
        )
    elif disc.mode == "hybrid":
        lines.append(
            "  Selection: score ≥ sdk threshold but fwa gate not met → workflow-capable, lighter emit defaults"
        )
    else:
        lines.append("  Selection: score < sdk threshold → SDK-only grouping, no workflow bundle")
    return "\n".join(lines)


def format_workflow_preview(graph: WorkflowGraph, disc_result: DiscriminationResult) -> str:
    lines: list[str] = []
    lines.append(
        f"Discrimination: mode={disc_result.mode}, score={disc_result.total_score:.2f}, "
        f"confidence={disc_result.confidence:.2f}"
    )
    for m in disc_result.matches:
        if m.matched:
            lines.append(f"  - {m.name} (w={m.weight}): {m.evidence}")
    lines.append("")
    lines.append(
        f"Workflow Graph: quality={graph.extraction_quality}, "
        f"{len(graph.stages)} stages, {len(graph.transitions)} transitions"
    )
    lines.append("Stages:")
    for s in graph.stages:
        gate = " [GATE]" if s.id in graph.gates else ""
        decision = " [DECISION]" if s.id in graph.decisions else ""
        contract = " [CONTRACT]" if s.id in graph.contracts else ""
        inferred = " [?]" if s.inferred else ""
        lines.append(f"  {s.id}: {s.name} ({s.label}){gate}{decision}{contract}{inferred}")
    if graph.transitions:
        lines.append("Transitions:")
        for t in graph.transitions:
            cond = f" ({t.condition})" if t.condition else ""
            lines.append(f"  {t.from_stage} → {t.to_stage}{cond}")
    if graph.gates:
        lines.append("Gates:")
        for sid, g in graph.gates.items():
            lines.append(f"  stage {sid}: {g.approval_mode} — {g.description}")
    if graph.decisions:
        lines.append("Decisions:")
        for sid, d in graph.decisions.items():
            parts = []
            for k, o in d.outcomes.items():
                mt = f", max={o.max_times}" if o.max_times else ""
                parts.append(f"{k}→{o.next_stage}{mt}")
            inf = " [inferred]" if d.inferred else ""
            lines.append(f"  stage {sid}: outcomes=[{', '.join(parts)}]{inf}")
    if graph.contracts:
        lines.append("Contracts:")
        for sid, c in graph.contracts.items():
            lines.append(f"  stage {sid}: input={c.input_files}, output={c.output_files}")
    return "\n".join(lines)
