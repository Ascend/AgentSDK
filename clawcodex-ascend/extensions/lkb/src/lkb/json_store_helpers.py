#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# AgentSDK validates this split-package import in the complete ordered migration state.
# pylint: disable=relative-beyond-top-level
"""Pure record projections shared by JSON board-store persistence paths."""

from __future__ import annotations

import copy
from typing import Any

from .commands import CommandResult
from .graph_types import RevisionVector
from .json_store_models import BoardEnvelope, now_iso, parse_ref


def build_command_audit_fields(
    *,
    board_id: str,
    command_id: str,
    result: CommandResult,
    actor: str,
    store_revision: int,
    revision_vector: RevisionVector,
    input_snapshot_hash: str,
    validation_run_id: str | None,
    subject_ref: str,
    affected_refs: list[str],
    rule: str,
) -> dict[str, Any]:
    """Return fields shared by received and executed command audit events."""
    return {
        "board_id": board_id,
        "command_id": command_id,
        "decision": result.decision,
        "actor": actor,
        "timestamp": now_iso(),
        "store_revision": store_revision,
        "revision_vector": revision_vector.to_dict(),
        "input_snapshot_hash": input_snapshot_hash,
        "validation_run_id": validation_run_id,
        "subject_ref": subject_ref,
        "affected_refs": affected_refs,
        "rule": rule,
    }


def build_graph_content(envelope: BoardEnvelope) -> dict[str, dict[str, Any]]:
    """Build an exact, revision-free content projection for each graph."""
    content: dict[str, dict[str, Any]] = {}
    for graph_id, graph in envelope.graphs.items():
        graph_data = copy.deepcopy(graph)
        graph_data.pop("revision", None)
        content[graph_id] = {
            "graph": graph_data,
            "nodes": {},
            "edges": {},
            "claims": {},
        }

    for record_id, record in envelope.nodes.items():
        graph_id = _node_graph(record)
        if graph_id in content:
            content[graph_id]["nodes"][record_id] = copy.deepcopy(record)
    for collection_name in ("edges", "claims"):
        collection = getattr(envelope, collection_name)
        for record_id, record in collection.items():
            for graph_id in _record_graph_ids(record):
                if graph_id in content:
                    content[graph_id][collection_name][record_id] = copy.deepcopy(record)
    return content


def _node_graph(node_dict: dict[str, Any]) -> str:
    """Extract a graph id from a node record's reference string."""
    return str(node_dict.get("ref", "")).split(":", 2)[0]


def _record_graph_ids(record: dict[str, Any]) -> set[str]:
    """Return every graph explicitly owned or touched by a domain record."""
    result: set[str] = set()
    graph = record.get("graph") or record.get("graph_id")
    if isinstance(graph, str) and graph:
        result.add(graph)
    for key in (
        "ref",
        "task_ref",
        "subject_ref",
        "subjectRef",
        "node_ref",
        "target_ref",
        "source_ref",
    ):
        value = record.get(key)
        if isinstance(value, str):
            result.add(parse_ref(value, location=key).graph)
    affected = record.get("affected_refs", record.get("affectedRefs"))
    if affected is not None:
        if not isinstance(affected, list):
            raise ValueError("affected_refs must be a list")
        for value in affected:
            result.add(parse_ref(value, location="affected_refs[]").graph)
    return result
