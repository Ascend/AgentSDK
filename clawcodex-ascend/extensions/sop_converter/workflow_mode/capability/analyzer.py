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


"""Static analysis of stage implementation files."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from ..ast_helpers import parse_ast
from .models import Capability, CapabilityKind, ExecutionMode, StageCapabilityProfile
from .patterns import ABSOLUTE_PATH_RE, CALL_PATTERNS, FRAGILITY_PATTERNS, IMPORT_PATTERNS

logger = logging.getLogger(__name__)


_CONTROL_FLOW_NODES = (ast.For, ast.While, ast.If, ast.Try, ast.With)


def _max_control_flow_depth(tree: ast.Module) -> int:
    """Return the maximum nesting depth of For/While/If/Try/With nodes.

    A top-level ``for`` counts as depth 1; a ``for`` containing an ``if``
    puts the ``if`` at depth 2. Tracking real depth (rather than counting
    nodes) keeps the score low for files with many sibling branches.
    """

    max_depth = 0

    def visit(node: ast.AST, depth: int) -> None:
        nonlocal max_depth
        if isinstance(node, _CONTROL_FLOW_NODES):
            depth += 1
            max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            visit(child, depth)

    visit(tree, 0)
    return max_depth


def _score_complexity(tree: ast.Module) -> float:
    lines = 0
    imports = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body_lines = getattr(node, "end_lineno", node.lineno) - node.lineno
            lines += max(body_lines, 1)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports += 1

    line_score = min(lines / 200.0, 1.0)
    depth_score = min(_max_control_flow_depth(tree) / 15.0, 1.0)
    import_score = min(imports / 20.0, 1.0)
    return min(0.5 * line_score + 0.3 * depth_score + 0.2 * import_score, 1.0)


def _score_fragility(source: str, tree: ast.Module) -> float:
    score = 0.0
    for pattern, weight in FRAGILITY_PATTERNS:
        if pattern.search(source):
            score += weight
    if ABSOLUTE_PATH_RE.search(source):
        score += 0.2

    has_try = any(isinstance(n, ast.Try) for n in ast.walk(tree))
    if not has_try and ("subprocess" in source or "os.system" in source):
        score += 0.1

    return min(score, 1.0)


def _detect_capabilities(source: str, tree: ast.Module) -> list[Capability]:
    found: dict[CapabilityKind, Capability] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _match_import(alias.name, found)
        elif isinstance(node, ast.ImportFrom) and node.module:
            _match_import(node.module, found)

    for pattern, kind, conf in CALL_PATTERNS:
        if pattern.search(source) and kind not in found:
            found[kind] = Capability(kind=kind, evidence=pattern.pattern[:40], confidence=conf)

    return list(found.values())


def _match_import(module: str, found: dict[CapabilityKind, Capability]) -> None:
    for pattern, kind, conf in IMPORT_PATTERNS:
        if pattern.search(module) and kind not in found:
            found[kind] = Capability(kind=kind, evidence=module, confidence=conf)


def _find_entry_function(tree: ast.Module, stage_name: str) -> str | None:
    candidates: list[str] = []
    preferred = f"run_{stage_name.replace('-', '_')}"
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in ("main", "run", preferred):
                return node.name
            if node.name == stage_name.replace("-", "_"):
                return node.name
            if not node.name.startswith("_"):
                candidates.append(node.name)
    return candidates[0] if candidates else None


def recommend_execution_mode(complexity: float, fragility: float) -> ExecutionMode:
    """Matrix from feature plan / C-G design §2.3."""
    if complexity < 0.4:
        if fragility < 0.3:
            return ExecutionMode.AGENT_NATIVE
        if fragility <= 0.6:
            return ExecutionMode.AGENT_NATIVE
        return ExecutionMode.WRAPPER
    if complexity <= 0.7:
        if fragility < 0.3:
            return ExecutionMode.AGENT_NATIVE
        if fragility <= 0.6:
            return ExecutionMode.HYBRID
        return ExecutionMode.WRAPPER
    if fragility < 0.3:
        return ExecutionMode.HYBRID
    return ExecutionMode.WRAPPER


def analyze_stage_file(
    stage_id: int,
    file_path: Path | None,
    *,
    stage_name: str = "",
) -> StageCapabilityProfile:
    """Analyze a single stage implementation file."""
    profile = StageCapabilityProfile(stage_id=stage_id)

    if file_path is None or not file_path.is_file():
        profile.notes.append("no implementation file")
        return profile

    tree = parse_ast(file_path)
    if tree is None:
        profile.notes.append(f"failed to parse {file_path}")
        return profile

    try:
        source = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        profile.notes.append(str(exc))
        return profile

    profile.capabilities = _detect_capabilities(source, tree)
    profile.complexity = _score_complexity(tree)
    profile.fragility = _score_fragility(source, tree)
    profile.execution_mode = recommend_execution_mode(profile.complexity, profile.fragility)
    profile.entry_function = _find_entry_function(tree, stage_name)

    return profile


def analyze_stage_sources(
    stage_id: int,
    paths: list[Path],
    *,
    stage_name: str = "",
) -> StageCapabilityProfile:
    """Merge analysis across multiple related source paths."""
    if not paths:
        return StageCapabilityProfile(stage_id=stage_id, notes=["no source paths"])

    merged = analyze_stage_file(stage_id, paths[0], stage_name=stage_name)
    cap_kinds = {c.kind for c in merged.capabilities}

    for path in paths[1:]:
        sub = analyze_stage_file(stage_id, path, stage_name=stage_name)
        merged.complexity = max(merged.complexity, sub.complexity)
        merged.fragility = max(merged.fragility, sub.fragility)
        merged.notes.extend(sub.notes)
        for cap in sub.capabilities:
            if cap.kind not in cap_kinds:
                merged.capabilities.append(cap)
                cap_kinds.add(cap.kind)
        if sub.entry_function and not merged.entry_function:
            merged.entry_function = sub.entry_function

    merged.execution_mode = recommend_execution_mode(merged.complexity, merged.fragility)
    return merged
