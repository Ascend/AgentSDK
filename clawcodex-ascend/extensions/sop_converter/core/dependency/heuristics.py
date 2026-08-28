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

"""Heuristics for dependency detection.

Two pure functions:

* :func:`pair_build_invoke` — given a list of lifecycle-classified ops,
  return the candidate ``(build_op, invoke_op)`` pairs using the
  following rules:

  * the build op's name returns a Dict containing an id-shaped field
  * the invoke op's required parameter name ends with ``_id`` / ``id``
  * the build op's id field name (after stripping ``_id``) overlaps
    with the invoke op's id parameter (i.e. the shared param
    contract is plausible).

* :func:`extract_shared_params` — for a candidate pair, return the
  list of parameter names that semantically match between
  ``build_op`` and ``invoke_op`` (the *id* and any auxiliaries that
  share a normalised name).

Both functions are pure predicates so the detector (and tests) can
build the dependency graph deterministically.
"""

from __future__ import annotations

from typing import Iterable

from extensions.sop_converter.core.heuristics.lifecycle import LifecycleKind, infer_lifecycle_kind
from extensions.sop_converter.core.source_parser import SourceOperation

# Field name suffixes we treat as "id-shaped" when an op returns a Dict.
_ID_FIELD_SUFFIXES = ("_id", "id", "Id")


def _normalise_param(name: str) -> str:
    """Strip ``_id`` / ``Id`` suffix and lowercase for cross-op matching."""
    stripped = name
    for suf in _ID_FIELD_SUFFIXES:
        if stripped.endswith(suf) and len(stripped) > len(suf):
            stripped = stripped[: -len(suf)]
            break
    return stripped.lower()


def _looks_like_id_field(name: str) -> bool:
    return any(name.endswith(suf) for suf in _ID_FIELD_SUFFIXES)


def _id_param_names(op: SourceOperation) -> list[str]:
    """Return required parameter names that look like an id."""
    return [p.name for p in op.parameters if p.required and _looks_like_id_field(p.name)]


def _dict_id_fields(op: SourceOperation) -> list[str]:
    """Infer the *return* Dict's id field name(s) from the op's name.

    Convention: ``build_agent`` / ``create_team_session`` return a dict
    keyed by ``agent_id`` / ``session_id`` (i.e. the op's name with
    the verb stripped, suffixed by ``_id``).  We use the most likely
    candidate and let the detector pair it with the invoke op.

    All six create-style prefixes are recognised, including
    ``load_`` (the prepare phase of a load→run chain).
    """
    name = (op.name or "").lower()
    for prefix in (
        "build_",
        "create_",
        "init_",
        "register_",
        "ensure_",
        "load_",
    ):
        if name.startswith(prefix):
            stem = name[len(prefix) :]
            if stem:
                return [f"{stem}_id"]
    return []


def _lifecycle(op: SourceOperation) -> LifecycleKind:
    return infer_lifecycle_kind(op)


def _op_key(op: SourceOperation, comp_name: str = "") -> str:
    """Stable kebab-case-ish key for an op, used as ``from`` / ``to`` in YAML.

    The format mirrors the Agent tool's registered name
    (``{comp_name}.{op_name}`` from ``register_component_tools``) so the
    graph keys line up with tool specs the agent actually sees.
    """
    comp = (comp_name or op.class_name or "").lower().replace("_", "-")
    op_part = (op.name or "").lower().replace("_", "-")
    if comp:
        return f"{comp}.{op_part}"
    return op_part


def pair_build_invoke(
    ops: Iterable[SourceOperation],
) -> list[tuple[SourceOperation, SourceOperation, list[str]]]:
    """Find all candidate ``(build_op, invoke_op, shared_params)`` triples.

    The result is a flat list — the caller is responsible for
    deduping or sorting.  The function is intentionally generous
    (returns *all* plausible candidates) so the detector can apply
    quality filters later.
    """
    ops_list = list(ops)
    builds = [op for op in ops_list if _lifecycle(op) == "create"]
    invokes = [op for op in ops_list if _lifecycle(op) == "invoke"]
    if not builds or not invokes:
        return []

    pairs: list[tuple[SourceOperation, SourceOperation, list[str]]] = []
    for b in builds:
        build_keys = {_normalise_param(k): k for k in _dict_id_fields(b)}
        # Also accept any parameter on the build side that looks like
        # it carries the future id (rare, but cheap to check).
        for p in b.parameters:
            if p.required and _looks_like_id_field(p.name):
                build_keys.setdefault(_normalise_param(p.name), p.name)
        if not build_keys:
            continue
        for inv in invokes:
            inv_id_params = _id_param_names(inv)
            if not inv_id_params:
                continue
            shared: list[str] = []
            for inv_param in inv_id_params:
                norm = _normalise_param(inv_param)
                if norm in build_keys:
                    shared.append(inv_param)
            if not shared:
                continue
            pairs.append((b, inv, shared))
    return pairs


def extract_shared_params(
    build_op: SourceOperation,
    invoke_op: SourceOperation,
) -> list[str]:
    """Return the list of shared parameter names for a single pair.

    Falls back to the union of normalised id param names when there
    is no direct name match — this keeps the YAML useful even when
    the build op's return type is ``Dict[str, Any]`` with no
    inferable key.
    """
    direct = pair_build_invoke([build_op, invoke_op])
    if direct:
        return direct[0][2]
    build_keys = {_normalise_param(k): k for k in _dict_id_fields(build_op)}
    inv_norm = {_normalise_param(p.name) for p in invoke_op.parameters if p.required}
    return [k for k, v in build_keys.items() if k in inv_norm] or list(build_keys)


__all__ = [
    "pair_build_invoke",
    "extract_shared_params",
    "_op_key",  # exported for the detector
]
