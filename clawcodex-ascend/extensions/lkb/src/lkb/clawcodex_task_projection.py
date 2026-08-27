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

"""Pure Task-v2 projection helpers kept independent from the routing adapter."""

from __future__ import annotations

from typing import Any


def tool_action(
    action: str,
    tool: str,
    tool_input: dict[str, Any],
    description: str,
) -> dict[str, Any]:
    """Build a model-executable recovery or next-action descriptor."""
    return {
        "action": action,
        "tool": tool,
        "input": tool_input,
        "description": description,
    }


def _blocker_active(snap: Any, graph_id: str, task_id: str) -> bool:
    from lkb.graph_types import NodeRef

    ref = NodeRef(graph_id, "task", task_id)
    node = snap.nodes.get(ref)
    if node is None:
        return False
    if node.state != "completed":
        return True
    payload = node.payload if isinstance(node.payload, dict) else {}
    return str(payload.get("derived_status", "") or "") in ("needs_recheck", "needs_review")


def _task_lkb_summary(
    snap: Any,
    plan: Any,
    ref: Any,
    node: Any,
    blocked_by: list[str],
    validation_run: dict[str, Any] | None,
    *,
    self_owner_sentinel: str,
) -> dict[str, Any]:
    payload = node.payload if hasattr(node, "payload") else {}
    if not isinstance(payload, dict):
        payload = {}
    derived_raw = str(payload.get("derived_status", "") or "")
    state = str(node.state or "pending")
    # Derived status: needs_recheck/needs_review win over base state.
    if derived_raw in ("needs_recheck", "needs_review"):
        derived = derived_raw
    elif state == "completed":
        derived = "verified"
    elif ref in plan.blocked_ids:
        derived = "blocked"
    elif state == "in_progress":
        derived = "running"
    else:
        derived = "ready"
    active_blockers = [blocker for blocker in blocked_by if _blocker_active(snap, ref.graph, blocker)]
    claimable = derived == "ready" and not (node.owner or None)
    if validation_run is not None:
        validation = {
            "status": "validated",
            "result": validation_run.get("result", ""),
            "runId": validation_run.get("validationRunId", validation_run.get("validation_run_id", "")),
        }
    else:
        validation = {"status": "unvalidated"}
    consistency_state = "stale" if derived == "needs_recheck" else "clean"
    next_actions: list[str] = []
    next_action_commands: list[dict[str, Any]] = []
    if derived == "ready":
        if not (node.owner or None):
            next_actions.append("claim_task")
            next_action_commands.append(
                tool_action(
                    "claim_task",
                    "TaskUpdate",
                    {"taskId": ref.id, "owner": self_owner_sentinel},
                    "Claim this task as the current Agent.",
                )
            )
        else:
            next_actions.append("start_task")
            next_action_commands.append(
                tool_action(
                    "start_task",
                    "TaskUpdate",
                    {"taskId": ref.id, "status": "in_progress"},
                    "Start the task after it has been claimed.",
                )
            )
    elif derived == "running":
        next_actions.append("complete_task")
        next_action_commands.append(
            tool_action(
                "complete_task",
                "TaskUpdate",
                {"taskId": ref.id, "status": "completed"},
                "Complete the running task.",
            )
        )
    elif derived == "needs_recheck":
        next_actions.append("revalidate_task")
        next_action_commands.append(
            tool_action(
                "revalidate_task",
                "Lkb",
                {"action": "revalidate", "taskId": ref.id},
                "Revalidate this task after its upstream work is current.",
            )
        )
    elif derived == "blocked":
        next_actions.append("complete_prerequisite")
        next_action_commands.append(
            tool_action(
                "explain_blockers",
                "Lkb",
                {"action": "explain", "taskId": ref.id},
                "Inspect the active prerequisite chain before retrying.",
            )
        )
    return {
        "derivedStatus": derived,
        "claimable": claimable,
        "activeBlockers": active_blockers,
        "validation": validation,
        "consistency": {
            "state": consistency_state,
            "issueCount": 1 if derived == "needs_recheck" else 0,
        },
        "nextActions": next_actions,
        "nextActionCommands": next_action_commands,
    }


def build_projection(
    snap: Any,
    plan_id: str,
    *,
    self_owner_sentinel: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project dependency edges and LKB summaries into Task-v2 task records."""
    from lkb.graph_types import PlanSnapshot

    plan = PlanSnapshot.from_graph(snap)
    blocked_by_map: dict[str, list[str]] = {}
    blocks_map: dict[str, list[str]] = {}
    for edge in snap.edges.values():
        if edge.graph != plan_id or edge.type != "depends_on":
            continue
        dependent = edge.source.id
        prerequisite = edge.target.id
        blocked_by_map.setdefault(dependent, []).append(prerequisite)
        blocks_map.setdefault(prerequisite, []).append(dependent)
    validation_by_ref: dict[str, dict[str, Any]] = {}
    for validation_run in snap.validation_runs.values() if hasattr(snap, "validation_runs") else []:
        if not isinstance(validation_run, dict):
            continue
        subject = validation_run.get("subjectRef") or validation_run.get("subject_ref")
        if isinstance(subject, dict):
            subject = f"{subject.get('graph')}:{subject.get('kind')}:{subject.get('id')}"
        if isinstance(subject, str) and subject:
            validation_by_ref[subject] = validation_run

    tasks: dict[str, Any] = {}
    counts = {"ready": 0, "running": 0, "blocked": 0, "needsRecheck": 0}
    for ref, node in snap.nodes.items():
        if ref.graph != plan_id or ref.kind != "task":
            continue
        payload = node.payload if hasattr(node, "payload") else {}
        if not isinstance(payload, dict):
            payload = {}
        blocked_by = sorted(blocked_by_map.get(ref.id, []))
        blocks = sorted(blocks_map.get(ref.id, []))
        lkb = _task_lkb_summary(
            snap,
            plan,
            ref,
            node,
            blocked_by,
            validation_by_ref.get(ref.to_str()),
            self_owner_sentinel=self_owner_sentinel,
        )
        tasks[ref.id] = {
            "id": ref.id,
            "subject": node.title,
            "description": str(payload.get("description", "")),
            "activeForm": str(payload.get("activeForm", node.title)),
            "status": str(node.state or "pending"),
            "owner": node.owner,
            "blocks": blocks,
            "blockedBy": blocked_by,
            "metadata": dict(payload.get("metadata", {}) or {}),
            "output": str(payload.get("output", "")),
            "lkb": lkb,
        }
        derived = lkb["derivedStatus"]
        if derived == "ready":
            counts["ready"] += 1
        elif derived == "running":
            counts["running"] += 1
        elif derived == "blocked":
            counts["blocked"] += 1
        elif derived == "needs_recheck":
            counts["needsRecheck"] += 1
    graph = snap.graphs.get(plan_id)
    metadata = graph.metadata if graph is not None else {}
    lkb_board = {
        "boardId": snap.board_id,
        "revision": snap.store_revision,
        "planId": plan_id,
        "planTitle": str(metadata.get("title") or plan_id),
        "planState": str(metadata.get("state") or "active"),
        "planRevision": graph.revision if graph is not None else 0,
        "counts": counts,
    }
    return tasks, lkb_board


__all__ = ["build_projection", "tool_action"]
