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


"""Heuristic rules for workflow discrimination."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable

from .ast_helpers import (
    class_annotation_names,
    find_dataclass_defs,
    find_dict_assignments,
    find_enum_classes,
    find_func_by_prefix,
    find_gate_assigns,
    get_enum_members,
    get_name,
    is_stage_like_enum,
    parse_enum_dict_mapping,
)
from .models import HeuristicMatch
from .scan_context import SourceScanContext

_STAGE_DIR_NAMES = ("stage_impls", "stages", "pipeline")


@dataclass
class HeuristicRule:
    name: str
    weight: float
    check_fn: Callable[..., HeuristicMatch]

    def check(self, ctx: SourceScanContext, **kwargs) -> HeuristicMatch:
        return self.check_fn(ctx, **kwargs)


def _empty_match(name: str, weight: float) -> HeuristicMatch:
    return HeuristicMatch(name=name, weight=weight, matched=False, evidence="", score=0.0)


def _check_stage_enum(ctx: SourceScanContext, **_kwargs) -> HeuristicMatch:
    weight = 0.25
    best_evidence = ""
    best_score = 0.0
    for tree in ctx.trees.values():
        for cls in find_enum_classes(tree):
            members = get_enum_members(cls)
            if not members:
                continue
            is_stage, factor = is_stage_like_enum(cls, members)
            if not is_stage:
                continue
            score = weight * factor
            if score > best_score:
                best_score = score
                best_evidence = f"class {cls.name}(IntEnum)"
    if best_score > 0:
        return HeuristicMatch(
            name="stage_enum",
            weight=weight,
            matched=True,
            evidence=best_evidence,
            score=best_score,
        )
    return _empty_match("stage_enum", weight)


def _check_state_transition(ctx: SourceScanContext, *, enum_names: set[str] | None = None) -> HeuristicMatch:
    weight = 0.20
    names = enum_names or ctx.enum_member_names
    if not names:
        return _empty_match("state_transition", weight)

    enum_classes = ctx.enum_class_names
    member_to_value = ctx.member_to_value

    for tree in ctx.trees.values():
        for var_name, dict_node in find_dict_assignments(tree):
            pairs = parse_enum_dict_mapping(dict_node, enum_classes, member_to_value)
            if pairs:
                return HeuristicMatch(
                    name="state_transition",
                    weight=weight,
                    matched=True,
                    evidence=f"{var_name} ({len(pairs)} transitions)",
                    score=weight,
                )
            # Fallback: string key overlap for Name-based dicts
            keys: list[str] = []
            for key in dict_node.keys:
                if key is None:
                    continue
                key_name = get_name(key)
                if key_name:
                    keys.append(key_name)
            if keys:
                overlap = sum(1 for k in keys if k in names) / len(keys)
                if overlap > 0.5:
                    return HeuristicMatch(
                        name="state_transition",
                        weight=weight,
                        matched=True,
                        evidence=f"{var_name} (name overlap {overlap:.0%})",
                        score=weight,
                    )
    return _empty_match("state_transition", weight)


def _check_io_contract(ctx: SourceScanContext, **_kwargs) -> HeuristicMatch:
    weight = 0.20
    for tree in ctx.trees.values():
        for cls in find_dataclass_defs(tree):
            ann = class_annotation_names(cls)
            if "input_files" in ann or "output_files" in ann:
                return HeuristicMatch(
                    name="io_contract",
                    weight=weight,
                    matched=True,
                    evidence=f"@dataclass {cls.name}",
                    score=weight,
                )
    return _empty_match("io_contract", weight)


def _check_control_flow(ctx: SourceScanContext, **_kwargs) -> HeuristicMatch:
    weight = 0.15
    for tree in ctx.trees.values():
        for func in find_func_by_prefix(tree):
            return HeuristicMatch(
                name="control_flow",
                weight=weight,
                matched=True,
                evidence=f"def {func.name}(...)",
                score=weight,
            )
    return _empty_match("control_flow", weight)


def _check_stage_dirs(ctx: SourceScanContext, **_kwargs) -> HeuristicMatch:
    weight = 0.10
    root = ctx.source_dir
    for child in root.iterdir():
        if child.is_dir() and child.name in _STAGE_DIR_NAMES:
            py_count = sum(1 for _ in child.glob("*.py"))
            if py_count > 0:
                return HeuristicMatch(
                    name="stage_dir",
                    weight=weight,
                    matched=True,
                    evidence=f"directory {child.name}/ ({py_count} .py files)",
                    score=weight,
                )
    return _empty_match("stage_dir", weight)


def _check_gate_definition(ctx: SourceScanContext, **_kwargs) -> HeuristicMatch:
    weight = 0.10
    for tree in ctx.trees.values():
        for var_name, value in find_gate_assigns(tree):
            if isinstance(value, (ast.Call, ast.Set)):
                call_name = get_name(value) if isinstance(value, ast.Call) else None
                if call_name in ("frozenset", "set") or isinstance(value, ast.Set):
                    return HeuristicMatch(
                        name="gate_definition",
                        weight=weight,
                        matched=True,
                        evidence=var_name,
                        score=weight,
                    )
    return _empty_match("gate_definition", weight)


ALL_RULES: list[HeuristicRule] = [
    HeuristicRule("stage_enum", 0.25, _check_stage_enum),
    HeuristicRule("state_transition", 0.20, _check_state_transition),
    HeuristicRule("io_contract", 0.20, _check_io_contract),
    HeuristicRule("control_flow", 0.15, _check_control_flow),
    HeuristicRule("stage_dir", 0.10, _check_stage_dirs),
    HeuristicRule("gate_definition", 0.10, _check_gate_definition),
]
