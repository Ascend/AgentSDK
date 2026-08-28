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


"""Generic Python pipeline workflow extractor."""

# pylint: disable=too-many-nested-blocks
from __future__ import annotations

import ast
import logging
from pathlib import Path

from ...ast_helpers import (
    class_annotation_names,
    extract_docstring_first_para,
    find_dataclass_defs,
    find_dict_mapping_assignments,
    find_enum_classes,
    find_func_by_prefix,
    find_gate_assigns,
    get_enum_members,
    get_enum_members_ordered,
    is_stage_like_enum,
    parse_ast,
    parse_contracts_dict,
    parse_enum_dict_mapping_from_expr,
    parse_frozenset_members,
    parse_stage_sequence_from_expr,
    parse_string_to_stage_dict,
    resolve_enum_member,
    to_kebab,
    walk_py_files,
)
from ..base import WorkflowExtractorBase
from ..models import (
    DecisionSpec,
    ExtractedStage,
    GateSpec,
    OutcomeSpec,
    StageContract,
    Transition,
)

logger = logging.getLogger(__name__)

_STAGE_DIR_NAMES = ("stage_impls", "stages", "pipeline")

# Retry caps for decision outcomes: rollback tables are limited to 2 retries
# per branch, while outcomes inferred from return statements default to 3.
_ROLLBACK_MAX_TIMES = 2
_INFERRED_DECISION_MAX_TIMES = 3


class GenericPipelineExtractor(WorkflowExtractorBase):
    def _ensure_scan(self, source_dir: Path):
        if self._scan is None:  # pylint: disable=access-member-before-definition
            from ...scan_context import SourceScanContext

            self._scan = SourceScanContext.build(source_dir)  # pylint: disable=attribute-defined-outside-init

    def extract_stages(self, source_dir: Path) -> list[ExtractedStage]:
        self._ensure_scan(source_dir)
        ctx = self._scan
        if ctx is None:
            return []

        stages: list[ExtractedStage] = []
        primary = ctx.primary_stage_enum

        for tree in ctx.trees.values():
            for cls in find_enum_classes(tree):
                if primary and cls.name != primary:
                    continue
                if not primary:
                    members = get_enum_members(cls)
                    is_stage, _ = is_stage_like_enum(cls, members)
                    if not is_stage:
                        continue
                members = get_enum_members(cls)
                for name, value in sorted(members.items(), key=lambda x: x[1]):
                    stages.append(
                        ExtractedStage(
                            id=value,
                            name=to_kebab(name),
                            label=name,
                            source_class=cls.name,
                            source_value=value,
                            description=extract_docstring_first_para(cls),
                        )
                    )
            if stages:
                break

        if stages:
            return stages

        stages = _stages_from_directory(source_dir)
        if stages:
            return stages

        if self._allow_coarse:
            return _stages_from_files_coarse(source_dir)
        return []

    def extract_transitions(self, source_dir: Path) -> list[Transition]:
        self._ensure_scan(source_dir)
        ctx = self._scan
        if ctx is None or not ctx.member_to_value:
            return []

        transitions: list[Transition] = []
        seen: set[tuple[int, int]] = set()
        for tree in ctx.trees.values():
            enum_members_ordered: list[tuple[str, int]] = []
            for cls in find_enum_classes(tree):
                if ctx.primary_stage_enum and cls.name != ctx.primary_stage_enum:
                    continue
                enum_members_ordered = get_enum_members_ordered(cls)
                if enum_members_ordered:
                    break
            module_assigns = dict(find_dict_mapping_assignments(tree))
            stage_sequence: list[int] = []
            if "STAGE_SEQUENCE" in module_assigns:
                stage_sequence = parse_stage_sequence_from_expr(
                    module_assigns["STAGE_SEQUENCE"],
                    enum_class_names=ctx.enum_class_names,
                    enum_members_ordered=enum_members_ordered,
                    module_assigns=module_assigns,
                )
            if not stage_sequence and enum_members_ordered:
                stage_sequence = [v for _, v in enum_members_ordered]

            for var_name, dict_expr in find_dict_mapping_assignments(tree):
                pairs = parse_enum_dict_mapping_from_expr(
                    dict_expr,
                    ctx.enum_class_names,
                    ctx.member_to_value,
                    stage_sequence=stage_sequence,
                )
                for from_id, to_id in pairs:
                    key = (from_id, to_id)
                    if key not in seen:
                        seen.add(key)
                        transitions.append(
                            Transition(
                                from_stage=from_id,
                                to_stage=to_id,
                                condition=var_name,
                                is_default=True,
                            )
                        )
        return transitions

    def extract_gates(self, source_dir: Path) -> dict[int, GateSpec]:
        self._ensure_scan(source_dir)
        ctx = self._scan
        if ctx is None:
            return {}

        gates: dict[int, GateSpec] = {}
        for tree in ctx.trees.values():
            for var_name, value in find_gate_assigns(tree):
                stage_ids = parse_frozenset_members(
                    value,
                    ctx.enum_class_names,
                    ctx.member_to_value,
                )
                for sid in stage_ids:
                    gates[sid] = GateSpec(
                        stage_id=sid,
                        approval_mode="manual",
                        description=f"Gate from {var_name}",
                        source_name=var_name,
                    )
        return gates

    def extract_decisions(self, source_dir: Path) -> dict[int, DecisionSpec]:
        self._ensure_scan(source_dir)
        ctx = self._scan
        if ctx is None:
            return {}

        decisions: dict[int, DecisionSpec] = {}
        for tree in ctx.trees.values():
            for var_name, dict_expr in find_dict_mapping_assignments(tree):
                if isinstance(dict_expr, ast.Dict) and "DECISION_ROLLBACK" in var_name.upper():
                    rollback = parse_string_to_stage_dict(
                        dict_expr,
                        ctx.enum_class_names,
                        ctx.member_to_value,
                    )
                    decision_stage = ctx.member_to_value.get("RESEARCH_DECISION")
                    if decision_stage is None:
                        for member, value in ctx.member_to_value.items():
                            if "DECISION" in member:
                                decision_stage = value
                                break
                    if decision_stage is not None and rollback:
                        outcomes = {
                            name: OutcomeSpec(next_stage=sid, max_times=_ROLLBACK_MAX_TIMES)
                            for name, sid in rollback.items()
                        }
                        decisions[decision_stage] = DecisionSpec(
                            stage_id=decision_stage,
                            outcomes=outcomes,
                            source_func=var_name,
                            inferred=False,
                        )
            for func in find_func_by_prefix(tree):
                spec = _parse_decision_func(func, ctx.member_to_value, ctx.enum_class_names)
                stage_id = _guess_decision_stage(func.name, ctx.member_to_value)
                if stage_id is None and ctx.member_to_value:
                    stage_id = max(ctx.member_to_value.values())
                if stage_id is not None:
                    if stage_id in decisions:
                        # ponytail: merge — multiple decision funcs may map to same stage
                        existing = decisions[stage_id]
                        existing.outcomes.update(spec["outcomes"])
                        if not spec["inferred"]:
                            existing.inferred = False
                        if existing.source_func:
                            existing.source_func = f"{existing.source_func}, {func.name}"
                        else:
                            existing.source_func = func.name
                    else:
                        decisions[stage_id] = DecisionSpec(
                            stage_id=stage_id,
                            outcomes=spec["outcomes"],
                            source_func=func.name,
                            inferred=spec["inferred"],
                        )
        return decisions

    def extract_contracts(self, source_dir: Path) -> dict[int, StageContract]:
        self._ensure_scan(source_dir)
        ctx = self._scan
        if ctx is None:
            return {}

        contracts: dict[int, StageContract] = {}
        for tree in ctx.trees.values():
            for cls in find_dataclass_defs(tree):
                ann = class_annotation_names(cls)
                if "input_files" not in ann and "output_files" not in ann:
                    continue
                stage_id = _match_stage_id(cls.name, ctx.member_to_value)
                if stage_id is None:
                    continue
                if stage_id in contracts:
                    logger.warning(
                        "Overriding contract for stage %s: %s replaces %s",
                        stage_id,
                        cls.name,
                        contracts[stage_id].source_class,
                    )
                contracts[stage_id] = StageContract(
                    stage_id=stage_id,
                    input_files=_extract_list_literal(cls, "input_files", tree),
                    output_files=_extract_list_literal(cls, "output_files", tree),
                    source_class=cls.name,
                )
            for var_name, dict_expr in find_dict_mapping_assignments(tree):
                if not isinstance(dict_expr, ast.Dict):
                    continue
                if "CONTRACT" not in var_name.upper():
                    continue
                parsed = parse_contracts_dict(
                    dict_expr,
                    ctx.enum_class_names,
                    ctx.member_to_value,
                )
                for stage_id, (inp, out, call_name, dod) in parsed.items():
                    contracts[stage_id] = StageContract(
                        stage_id=stage_id,
                        input_files=inp,
                        output_files=out,
                        dod=dod,
                        source_class=call_name,
                    )
        return contracts


def _stages_from_directory(source_dir: Path) -> list[ExtractedStage]:
    stages: list[ExtractedStage] = []
    order = 1
    for dir_name in _STAGE_DIR_NAMES:
        stage_dir = source_dir / dir_name
        if not stage_dir.is_dir():
            continue
        for py_file in sorted(stage_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            stem = py_file.stem
            tree = parse_ast(py_file)
            desc = ""
            if tree:
                desc = extract_docstring_first_para(tree)
            stages.append(
                ExtractedStage(
                    id=order,
                    name=to_kebab(stem),
                    label=stem.upper(),
                    file_path=str(py_file.relative_to(source_dir)),
                    description=desc,
                    inferred=True,
                )
            )
            order += 1
        if stages:
            break
    return stages


def _stages_from_files_coarse(source_dir: Path) -> list[ExtractedStage]:
    stages: list[ExtractedStage] = []
    order = 1
    for py_file in walk_py_files(source_dir):
        if any(part in py_file.parts for part in _STAGE_DIR_NAMES):
            continue
        if py_file.name.startswith("_"):
            continue
        stem = py_file.stem
        stages.append(
            ExtractedStage(
                id=order,
                name=to_kebab(stem),
                label=stem,
                file_path=str(py_file.relative_to(source_dir)),
                description="",
                inferred=True,
            )
        )
        order += 1
    return stages


def _parse_decision_func(
    func: ast.FunctionDef,
    member_to_value: dict[str, int],
    enum_class_names: set[str],
) -> dict:
    outcomes: dict[str, OutcomeSpec] = {}
    inferred = True
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            outcome_name, next_stage = _parse_return_outcome(
                node.value,
                member_to_value,
                enum_class_names,
            )
            if outcome_name:
                outcomes[outcome_name] = OutcomeSpec(
                    next_stage=next_stage,
                    max_times=_INFERRED_DECISION_MAX_TIMES,
                )
                if next_stage is not None:
                    inferred = False
    return {"outcomes": outcomes, "inferred": inferred}


def _parse_return_outcome(
    node: ast.expr,
    member_to_value: dict[str, int],
    enum_class_names: set[str],
) -> tuple[str | None, int | None]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, None
    ref = resolve_enum_member(node, enum_class_names, member_to_value)
    if ref:
        return ref[0], ref[1]  # use actual enum member name as outcome
    if isinstance(node, ast.Tuple) and node.elts:
        first = node.elts[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            second = node.elts[1] if len(node.elts) > 1 else None
            if second is not None:
                ref2 = resolve_enum_member(second, enum_class_names, member_to_value)
                if ref2:
                    return first.value, ref2[1]
    return None, None


def _guess_decision_stage(func_name: str, member_to_value: dict[str, int]) -> int | None:
    """Match the decision function to the most specific stage member.

    When several enum members share a substring with the function name, the
    longest match wins so the result does not depend on dict insertion order.
    """
    func_lower = func_name.lower()
    matches = [(member, value) for member, value in member_to_value.items() if member.lower() in func_lower]
    if not matches:
        return None
    return max(matches, key=lambda m: len(m[0]))[1]


def _match_stage_id(class_name: str, member_to_value: dict[str, int]) -> int | None:
    upper = class_name.upper()
    for member, value in member_to_value.items():
        if member in upper:
            return value
    return None


def _extract_list_literal(cls: ast.ClassDef, field_name: str, tree: ast.Module) -> list[str]:
    """Best-effort: find class attribute assignment for field defaults in module."""
    # Prefer the class body so a module-level variable with the same name
    # cannot shadow the annotated class attribute.
    for item in cls.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if item.target.id == field_name and item.value is not None:
                return _const_str_list(item.value)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == field_name:
                return _const_str_list(node.value)
    return []


def _const_str_list(node: ast.expr) -> list[str]:
    if isinstance(node, (ast.List, ast.Tuple)):
        result: list[str] = []
        for el in node.elts:
            if isinstance(el, ast.Constant) and isinstance(el.value, str):
                result.append(el.value)
        return result
    return []
