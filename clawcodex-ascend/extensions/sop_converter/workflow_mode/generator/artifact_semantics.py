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


"""Human-readable semantics for known pipeline artifact files."""

from __future__ import annotations

from ..extractors.models import StageContract

# Field-level descriptions for artifacts referenced across ARC / ResearchClaw stages.
ARTIFACT_SEMANTICS: dict[str, str] = {
    "goal.md": (
        "Markdown research goal. Sections: Topic, Scope, SMART Goal "
        "(Specific / Measurable / Achievable / Relevant / Time-bound), "
        "Constraints, Success Criteria. "
        "DoD: SMART goal statement with topic, scope, and constraints."
    ),
    "hardware_profile.json": (
        "JSON hardware snapshot (`detect_hardware()`). Fields: "
        "`has_gpu` (bool), `gpu_type` (`cuda` | `mps` | `cpu`), "
        "`gpu_name` (str), `vram_mb` (int | null), "
        "`tier` (`high` | `limited` | `cpu_only`), "
        "`warning` (str; empty when tier is high)."
    ),
    "problem_tree.md": (
        "Markdown problem decomposition derived from `goal.md`: sub-questions, "
        "priority ranking, risks. DoD: at least three prioritized sub-questions."
    ),
}


def describe_output_file(filename: str, *, stage_dod: str = "") -> str:
    """Return a one-line semantic description for an output artifact."""
    if filename in ARTIFACT_SEMANTICS:
        return ARTIFACT_SEMANTICS[filename]
    if stage_dod:
        return f"Stage DoD: {stage_dod}"
    return "Must exist and be non-empty."


def output_descriptions(
    output_files: list[str],
    *,
    stage_dod: str = "",
) -> dict[str, str]:
    return {f: describe_output_file(f, stage_dod=stage_dod) for f in output_files}


def format_io_contract_markdown(contract: StageContract | None) -> str:
    """Render Inputs/Outputs sections for skill markdown bodies."""
    if contract is None:
        return ""
    lines: list[str] = []
    if contract.input_files:
        lines.append("## Inputs")
        for f in contract.input_files:
            desc = ARTIFACT_SEMANTICS.get(f)
            if desc:
                lines.append(f"- `{f}` — {desc}")
            else:
                lines.append(f"- `{f}`")
        lines.append("")
    if contract.output_files:
        lines.append("## Outputs")
        for f in contract.output_files:
            desc = describe_output_file(f, stage_dod=contract.dod)
            lines.append(f"- `{f}` — {desc}")
        if contract.dod:
            lines.append("")
            lines.append(f"**DoD:** {contract.dod}")
    return "\n".join(lines)
