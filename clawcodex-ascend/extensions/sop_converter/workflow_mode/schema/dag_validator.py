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

"""Compile-side DAG validation."""

# pylint: disable=too-many-nested-blocks
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DagValidationResult:
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _kahn_order(stage_ids: set[int], depends_on: dict[int, list[int]]) -> tuple[list[int] | None, str | None]:
    in_degree = {sid: len(depends_on.get(sid, [])) for sid in stage_ids}
    adj: dict[int, list[int]] = {sid: [] for sid in stage_ids}
    for sid, deps in depends_on.items():
        for dep in deps:
            if dep in adj:
                adj[dep].append(sid)

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    order: list[int] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(stage_ids):
        return None, "Workflow DAG contains a cycle"
    return order, None


def validate_workflow_dict(data: dict) -> DagValidationResult:
    """Validate emitted workflow.yaml shape."""
    result = DagValidationResult()
    stages_raw = data.get("stages", [])
    if not isinstance(stages_raw, list):
        result.errors.append("'stages' must be a list")
        return result

    stage_ids: set[int] = set()
    depends_on: dict[int, list[int]] = {}

    for i, raw in enumerate(stages_raw):
        if not isinstance(raw, dict):
            result.errors.append(f"stages[{i}] must be a mapping")
            continue
        sid = raw.get("id")
        if sid is None:
            result.errors.append(f"stages[{i}] missing id")
            continue
        try:
            sid_int = int(sid)
        except (TypeError, ValueError):
            result.errors.append(f"stages[{i}] id must be int")
            continue
        if sid_int in stage_ids:
            result.errors.append(f"duplicate stage id {sid_int}")
        stage_ids.add(sid_int)

        deps = raw.get("depends_on", [])
        if not isinstance(deps, list):
            result.errors.append(f"stage {sid_int}: depends_on must be a list")
            deps = []
        dep_ints: list[int] = []
        for d in deps:
            try:
                dep_ints.append(int(d))
            except (TypeError, ValueError):
                result.warnings.append(f"stage {sid_int}: non-int depends_on entry {d!r}")
        depends_on[sid_int] = dep_ints

    for sid, deps in depends_on.items():
        for dep in deps:
            if dep not in stage_ids:
                result.errors.append(f"stage {sid}: depends_on references unknown stage {dep}")

    order, cycle_err = _kahn_order(stage_ids, depends_on)
    if cycle_err:
        result.errors.append(cycle_err)
    elif order is None:
        result.errors.append("DAG sort failed")

    id_set = stage_ids
    for raw in stages_raw:
        if not isinstance(raw, dict):
            continue
        sid_val = raw.get("id")
        if sid_val is None:
            continue
        try:
            sid = int(sid_val)
        except (TypeError, ValueError):
            continue
        rollback = raw.get("gate_rollback_to")
        if rollback is not None:
            try:
                rb = int(rollback)
            except (TypeError, ValueError):
                result.warnings.append(f"stage {sid}: invalid gate_rollback_to")
                continue
            if rb not in id_set:
                result.errors.append(f"stage {sid}: gate_rollback_to references unknown stage {rb}")

        outcomes = raw.get("decision_outcomes", {})
        if isinstance(outcomes, dict):
            for outcome, spec in outcomes.items():
                if not isinstance(spec, dict):
                    continue
                for key in ("next", "rollback_to"):
                    ref = spec.get(key)
                    if ref is not None:
                        try:
                            ref_int = int(ref)
                        except (TypeError, ValueError):
                            result.warnings.append(f"stage {sid} outcome {outcome}: invalid {key}")
                            continue
                        if ref_int not in id_set:
                            result.errors.append(
                                f"stage {sid} outcome {outcome}: {key} references unknown stage {ref_int}"
                            )

    return result
