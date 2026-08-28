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


"""Interactive completion mode — TODO template generation.

When workflow extraction fails (empty graph) in hybrid/fwa mode,
``--interactive`` flag triggers generation of structured TODO templates
that guide the user to manually complete the missing workflow definitions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import DiscriminationResult, HeuristicMatch

logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────

_MATCH_NAMES: dict[str, str] = {
    "stage_enum": "阶段枚举",
    "state_transition": "状态转换",
    "io_contract": "IO 契约",
    "control_flow": "控制流决策",
    "stage_dir": "阶段实现目录",
    "gate_definition": "GATE 定义",
}


def _matched(name: str, matches: list[HeuristicMatch]) -> HeuristicMatch | None:
    for m in matches:
        if m.name == name and m.matched:
            return m
    return None


def _format_matched_evidence(matches: list[HeuristicMatch]) -> str:
    """Build a human-readable summary of detected features."""
    lines: list[str] = []
    for m in matches:
        label = _MATCH_NAMES.get(m.name, m.name)
        if m.matched:
            lines.append(f"  ✅ {label} (权重={m.weight}, 证据: {m.evidence})")
        else:
            lines.append(f"  ❌ {label} (权重={m.weight})")
    return "\n".join(lines)


# ── template sections ────────────────────────────────────────────────


def _stage_enum_section(
    matches: list[HeuristicMatch],
    source_dir: Path,
) -> str:
    """Generate TODO for stage definitions."""
    m = _matched("stage_enum", matches)
    if m:
        return (
            "## 阶段定义 (Stage Definitions)\n"
            f"\n"
            f"检测到阶段枚举: {m.evidence}\n"
            "请按以下格式列出所有阶段 ID 和名称:\n"
            "```yaml\n"
            "stages:\n"
            "  - id: 1\n"
            "    name: stage_one\n"
            "    label: 阶段一\n"
            "  - id: 2\n"
            "    name: stage_two\n"
            "    label: 阶段二\n"
            "```\n"
        )
    # Check if stage directories exist
    dir_m = _matched("stage_dir", matches)
    if dir_m:
        return (
            "## 阶段定义 (Stage Definitions)\n"
            f"\n"
            f"检测到阶段实现目录: {dir_m.evidence}\n"
            "未检测到阶段枚举，请根据目录结构手动定义阶段:\n"
            "```yaml\n"
            "stages:\n"
            "  - id: 1\n"
            "    name: stage_one\n"
            "    label: 阶段一\n"
            "  - id: 2\n"
            "    name: stage_two\n"
            "    label: 阶段二\n"
            "```\n"
        )
    return (
        "## 阶段定义 (Stage Definitions) — TODO\n"
        "\n"
        "未检测到阶段枚举或阶段目录。请手动定义各阶段:\n"
        "```yaml\n"
        "stages:\n"
        "  - id: 1\n"
        "    name: stage_one\n"
        "    label: 阶段一\n"
        "  - id: 2\n"
        "    name: stage_two\n"
        "    label: 阶段二\n"
        "```\n"
    )


def _transitions_section(matches: list[HeuristicMatch]) -> str:
    """Generate TODO for transition rules."""
    m = _matched("state_transition", matches)
    if m:
        return (
            "## 状态转换 (Transitions)\n"
            f"\n"
            f"检测到状态转换映射: {m.evidence}\n"
            "请按以下格式确认或补充转换规则:\n"
            "```yaml\n"
            "transitions:\n"
            "  - from: 1\n"
            "    to: 2\n"
            "    condition: on_success\n"
            "  - from: 2\n"
            "    to: 3\n"
            "    condition: on_success\n"
            "```\n"
        )
    return (
        "## 状态转换 (Transitions) — TODO\n"
        "\n"
        "未检测到状态转换映射。请定义各阶段之间的流转关系:\n"
        "```yaml\n"
        "transitions:\n"
        "  - from: 1\n"
        "    to: 2\n"
        "  - from: 2\n"
        "    to: 3\n"
        "```\n"
    )


def _gates_section(matches: list[HeuristicMatch]) -> str:
    """Generate TODO for gate definitions."""
    m = _matched("gate_definition", matches)
    if m:
        return (
            "## GATE 定义 (Gate Definitions)\n"
            f"\n"
            f"检测到 GATE 定义: {m.evidence}\n"
            "请按以下格式配置每个 GATE 的审批模式:\n"
            "```yaml\n"
            "gates:\n"
            "  stage_id: 2\n"
            "  approval_mode: manual  # manual | auto | threshold\n"
            "  description: 人工审核阶段\n"
            "```\n"
        )
    return (
        "## GATE 定义 (Gate Definitions) — TODO（可选）\n"
        "\n"
        "未检测到 GATE 定义。如果工作流需要人工审批节点，请补充:\n"
        "```yaml\n"
        "gates:\n"
        "  stage_id: 2\n"
        "  approval_mode: manual\n"
        "```\n"
    )


def _decisions_section(matches: list[HeuristicMatch]) -> str:
    """Generate TODO for decision functions."""
    m = _matched("control_flow", matches)
    if m:
        return (
            "## 决策函数 (Decision Functions)\n"
            f"\n"
            f"检测到控制流决策函数: {m.evidence}\n"
            "请按以下格式定义每个决策分支的后续流转:\n"
            "```yaml\n"
            "decisions:\n"
            "  stage_id: 3\n"
            "  outcomes:\n"
            "    approved:\n"
            "      next: 4\n"
            "    rejected:\n"
            "      next: 1\n"
            "      max_times: 3\n"
            "```\n"
        )
    return (
        "## 决策函数 (Decision Functions) — TODO（可选）\n"
        "\n"
        "未检测到决策函数。如果工作流需要分支判断，请补充:\n"
        "```yaml\n"
        "decisions:\n"
        "  stage_id: 3\n"
        "  outcomes:\n"
        "    approved:\n"
        "      next: 4\n"
        "    rejected:\n"
        "      next: 1\n"
        "```\n"
    )


def _contracts_section(matches: list[HeuristicMatch]) -> str:
    """Generate TODO for IO contracts."""
    m = _matched("io_contract", matches)
    if m:
        return (
            "## IO 契约 (Contracts)\n"
            f"\n"
            f"检测到 IO 契约 dataclass: {m.evidence}\n"
            "请按以下格式为每个阶段指定输入/输出:\n"
            "```yaml\n"
            "contracts:\n"
            "  stage_id: 1\n"
            "  input_files:\n"
            "    - data/input.csv\n"
            "  output_files:\n"
            "    - data/intermediate.json\n"
            "```\n"
        )
    return (
        "## IO 契约 (Contracts) — TODO（可选）\n"
        "\n"
        "未检测到 IO 契约。如果阶段之间有文件传递约束，请补充:\n"
        "```yaml\n"
        "contracts:\n"
        "  stage_id: 1\n"
        "  input_files:\n"
        "    - data/input.csv\n"
        "  output_files:\n"
        "    - data/output.json\n"
        "```\n"
    )


# ── public API ───────────────────────────────────────────────────────


def generate_completion_todo(
    disc: DiscriminationResult,
    source_dir: str | Path,
    out_path: str | Path | None = None,
) -> str:
    """Generate a structured TODO markdown template for manual completion.

    Parameters
    ----------
    disc:
        Discrimination result (carries scan context and
        heuristic matches that inform the template).
    source_dir:
        Path to the source directory being converted.
    out_path:
        When set, the TODO template is also written to
        ``{out_path}/TODO-workflow-completion.md``.

    Returns
    -------
    str:
        The full TODO template content.
    """
    path = Path(source_dir)
    matches = disc.matches
    source_name = path.name

    sections: list[str] = [
        f"# TODO: 工作流补全模板 — {source_name}",
        "",
        "生成时间: 自动",
        "",
        "## 概述",
        "",
        f"自动提取未能从 `{source_name}` 中提取完整的工作流结构。",
        "以下是根据启发式判别结果生成的 TODO 模板，请逐项补充完整。",
        "",
        "### 判别结果摘要",
        "",
        f"模式: {disc.mode}",
        f"总分: {disc.total_score:.2f}",
        f"推荐提取器: {disc.recommended_extractor}",
        "",
        "### 特征检测明细",
        "",
        _format_matched_evidence(matches),
        "",
        "---",
        "",
        _stage_enum_section(matches, path),
        "",
        _transitions_section(matches),
        "",
        _gates_section(matches),
        "",
        _decisions_section(matches),
        "",
        _contracts_section(matches),
        "",
        "---",
        "",
        "## 完成指引",
        "",
        "1. 逐项填写上述 TODO 区域，补充阶段定义、转换规则等。",
        "2. 重命名此文件为 ``workflow.yaml`` 并放入 ``.clawcodex/`` 目录，",
        "   或使用 ``sop convert`` 配合 ``--mode fwa`` 重新尝试自动提取。",
        "3. 也可使用 ``--mode fwa --extractor <name>`` 指定专用提取器。",
        "",
    ]

    content = "\n".join(sections)

    if out_path:
        out = Path(out_path)
        out.mkdir(parents=True, exist_ok=True)
        todo_path = out / "TODO-workflow-completion.md"
        todo_path.write_text(content, encoding="utf-8")
        logger.info("Wrote TODO completion template to %s", todo_path)

    return content


def generate_completion_yaml_stub(
    disc: DiscriminationResult,
    source_dir: str | Path,
    out_path: str | Path | None = None,
) -> str:
    """Generate a minimal workflow.yaml stub with TODO placeholders.

    Useful when the user wants a YAML skeleton instead of a markdown
    template — fills in whatever was detected and leaves TODO markers
    for the rest.
    """
    path = Path(source_dir)
    matches = disc.matches
    source_name = path.name
    # Quote the name for the YAML stub so directory names containing
    # YAML special characters (``:``, ``#``, quotes, etc.) stay valid.
    yaml_name = source_name.replace("\\", "\\\\").replace('"', '\\"')

    # Build best-effort stage list
    stage_lines: list[str] = []
    m = _matched("stage_enum", matches)
    if m:
        stage_lines.append("    # TODO: fill in actual stage names and labels")
        stage_lines.append("    # Detected stage enum, derive concrete entries:")
        stage_lines.append("    # - id: 1")
        stage_lines.append("    #   name: stage_one")
        stage_lines.append("    #   label: 阶段一")
    else:
        stage_lines.append("    # TODO: define stages")
        stage_lines.append("    # - id: 1")
        stage_lines.append("    #   name: stage_one")
        stage_lines.append("    #   label: 阶段一")

    lines = [
        f"# TODO: 工作流定义 — {source_name}",
        "# 由 clawcodex sop convert --interactive 自动生成骨架",
        "# 请逐项补充完整后重试",
        "",
        "version: '1.0'",
        f'name: "{yaml_name}"',
        "",
        "stages:",
        *stage_lines,
        "",
        "# TODO: uncomment and fill in transitions",
        "# transitions:",
        "#   - from: 1",
        "#     to: 2",
        "",
        "# TODO: uncomment and fill in gates (if any)",
        "# gates:",
        "#   stage_id: 2",
        "#   approval_mode: manual",
        "",
        "# TODO: uncomment and fill in decisions (if any)",
        "# decisions:",
        "#   stage_id: 3",
        "#   outcomes:",
        "#     approved:",
        "#       next: 4",
        "",
        "config:",
        "  workspace: '.'",
        "",
    ]
    content = "\n".join(lines)

    if out_path:
        out = Path(out_path)
        out.mkdir(parents=True, exist_ok=True)
        yaml_path = out / "workflow.yaml.stub"
        yaml_path.write_text(content, encoding="utf-8")
        logger.info("Wrote YAML stub to %s", yaml_path)

    return content
